"""
sac_agent.py — Soft Actor-Critic with Dirichlet Policy (v2.1)
=============================================================
Fixes from v2:
    - Alpha tuning uses Dirichlet ENTROPY (not log_prob) — prevents explosion
    - log_alpha clamped to [-5, 2] (alpha range: 0.007 to 7.4)
    - Target entropy calibrated for Dirichlet (positive, ~log(K))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import deque
from typing import Optional

from .networks import DirichletActor, Critic

EPSILON = 1e-6


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state_dict, weights, reward, next_state_dict, done, n_tradable):
        self.buffer.append({
            "asset_features": state_dict["asset_features"].copy(),
            "global_features": state_dict["global_features"].copy(),
            "weights": state_dict["weights"].copy(),
            "action_weights": weights.copy(),
            "reward": reward,
            "next_asset_features": next_state_dict["asset_features"].copy(),
            "next_global_features": next_state_dict["global_features"].copy(),
            "next_weights": next_state_dict["weights"].copy(),
            "done": done,
            "n_tradable": n_tradable,
        })

    def sample(self, batch_size, device):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        groups = {}
        for item in batch:
            n = item["n_tradable"]
            if n not in groups:
                groups[n] = []
            groups[n].append(item)
        return groups

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


def _batch_group(items, device):
    n = items[0]["n_tradable"]
    return {
        "state": {
            "asset_features": torch.FloatTensor(np.stack([it["asset_features"] for it in items])).to(device),
            "global_features": torch.FloatTensor(np.stack([it["global_features"] for it in items])).to(device),
            "weights": torch.FloatTensor(np.stack([it["weights"] for it in items])).to(device),
            "n_tradable": n,
        },
        "action_weights": torch.FloatTensor(np.stack([it["action_weights"] for it in items])).to(device),
        "rewards": torch.FloatTensor([it["reward"] for it in items]).unsqueeze(1).to(device),
        "next_state": {
            "asset_features": torch.FloatTensor(np.stack([it["next_asset_features"] for it in items])).to(device),
            "global_features": torch.FloatTensor(np.stack([it["next_global_features"] for it in items])).to(device),
            "weights": torch.FloatTensor(np.stack([it["next_weights"] for it in items])).to(device),
            "n_tradable": n,
        },
        "dones": torch.FloatTensor([it["done"] for it in items]).unsqueeze(1).to(device),
        "n_tradable": n,
    }


class SACAgent:
    DEFAULT_CONFIG = {
        "n_asset_features": 7,
        "n_global_features": 5,
        "lstm_hidden": 64,
        "embed_dim": 64,
        "n_attn_heads": 4,
        "scorer_hidden": 128,
        "critic_hidden": 256,
        "lr_actor": 1e-4,
        "lr_critic": 3e-4,
        "lr_alpha": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha_init": 0.01,
        "auto_alpha": True,
        "buffer_capacity": 500000,
        "batch_size": 256,
        "gradient_steps": 1,
        "warmup_steps": 256,
        "device": "auto",
    }

    def __init__(self, config=None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config

        if c["device"] == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(c["device"])
        print(f"  SAC Agent using device: {self.device}")

        self.actor = DirichletActor(
            n_asset_features=c["n_asset_features"], n_global_features=c["n_global_features"],
            lstm_hidden=c["lstm_hidden"], embed_dim=c["embed_dim"],
            n_attn_heads=c["n_attn_heads"], scorer_hidden=c["scorer_hidden"],
        ).to(self.device)

        self.critic = Critic(
            n_asset_features=c["n_asset_features"], n_global_features=c["n_global_features"],
            lstm_hidden=c["lstm_hidden"], embed_dim=c["embed_dim"],
            n_attn_heads=c["n_attn_heads"], critic_hidden=c["critic_hidden"],
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=c["lr_actor"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c["lr_critic"])

        self.auto_alpha = c["auto_alpha"]
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(max(c["alpha_init"], 1e-4)), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=c["lr_alpha"])
        else:
            self.log_alpha = torch.tensor(
                np.log(max(c["alpha_init"], 1e-4)), dtype=torch.float32, device=self.device,
            )

        self.buffer = ReplayBuffer(c["buffer_capacity"])
        self.total_steps = 0
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state_dict, deterministic=False):
        with torch.no_grad():
            torch_state = {
                "asset_features": torch.FloatTensor(state_dict["asset_features"]).unsqueeze(0).to(self.device),
                "global_features": torch.FloatTensor(state_dict["global_features"]).unsqueeze(0).to(self.device),
                "weights": torch.FloatTensor(state_dict["weights"]).unsqueeze(0).to(self.device),
                "n_tradable": state_dict["n_tradable"],
            }
            if deterministic:
                weights = self.actor.get_deterministic_action(torch_state)
            else:
                weights, _, _ = self.actor.sample(torch_state)
            return weights.cpu().numpy().flatten()

    def store_transition(self, state_dict, action_weights, reward, next_state_dict, done, n_tradable):
        self.buffer.push(state_dict, action_weights, reward, next_state_dict, done, n_tradable)
        self.total_steps += 1

    def update(self):
        c = self.config
        if len(self.buffer) < c["warmup_steps"]:
            return {}

        groups = self.buffer.sample(c["batch_size"], self.device)
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        total_count = 0

        for n_t, items in groups.items():
            if len(items) < 2:
                continue
            batch = _batch_group(items, self.device)
            B = len(items)
            K = n_t + 1  # action dimension (stocks + cash)

            # --- Critic ---
            with torch.no_grad():
                next_w, next_lp, _ = self.actor.sample(batch["next_state"])
                q1t, q2t = self.critic_target(batch["next_state"], next_w)
                q_tgt = torch.min(q1t, q2t) - self.alpha * next_lp
                td_target = batch["rewards"] + c["gamma"] * (1 - batch["dones"]) * q_tgt

            q1, q2 = self.critic(batch["state"], batch["action_weights"])
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # --- Actor ---
            new_w, log_prob, _ = self.actor.sample(batch["state"])
            q1n, q2n = self.critic(batch["state"], new_w)
            actor_loss = (self.alpha.detach() * log_prob - torch.min(q1n, q2n)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # --- Alpha (FIXED: use entropy, not log_prob) ---
            if self.auto_alpha:
                # Dirichlet target entropy: log(K) is entropy of uniform Dir(1,1,...,1)
                # We target slightly below uniform for mild concentration
                target_ent = np.log(K) * 0.8  # ~3.7 for K=101
                # Compute actual Dirichlet entropy
                with torch.no_grad():
                    actual_entropy = self.actor.entropy(batch["state"])  # (B, 1)
                # If entropy < target: increase alpha (encourage exploration)
                # If entropy > target: decrease alpha (allow exploitation)
                alpha_loss = (self.log_alpha * (actual_entropy - target_ent).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                # CLAMP log_alpha to prevent explosion
                with torch.no_grad():
                    self.log_alpha.clamp_(-5.0, 2.0)  # alpha in [0.007, 7.4]
                total_alpha_loss += alpha_loss.item() * B

            total_critic_loss += critic_loss.item() * B
            total_actor_loss += actor_loss.item() * B
            total_count += B

        self._soft_update()
        self.update_count += 1
        if total_count == 0:
            return {}
        return {
            "critic_loss": total_critic_loss / total_count,
            "actor_loss": total_actor_loss / total_count,
            "alpha_loss": total_alpha_loss / total_count if self.auto_alpha else 0,
            "alpha": self.alpha.item(),
            "buffer_size": len(self.buffer),
        }

    def _soft_update(self):
        tau = self.config["tau"]
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha, "config": self.config,
            "total_steps": self.total_steps, "update_count": self.update_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.log_alpha = ckpt["log_alpha"]
        self.total_steps = ckpt["total_steps"]
        self.update_count = ckpt["update_count"]

    def param_count(self):
        def c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"actor": c(self.actor), "critic": c(self.critic), "total": c(self.actor) + c(self.critic)}

    def reset_for_fine_tune(self):
        self.buffer.clear()
        self.total_steps = 0
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(max(self.config["alpha_init"], 1e-4)), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config["lr_alpha"])
