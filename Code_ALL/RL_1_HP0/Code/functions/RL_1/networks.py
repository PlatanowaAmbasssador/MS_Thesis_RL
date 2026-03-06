"""
networks.py — Neural Network Architectures for SAC Portfolio Agent (v2)
=========================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Architecture (literature-aligned):
    Per-asset LSTM temporal encoder (W=60 lookback) →
    Cross-sectional multi-head attention with global token →
    Dirichlet policy head (N+1 assets incl. cash) / Twin critics

Key design choices:
    - LSTM encodes 60-day feature history per asset (Markov approximation)
    - Multi-head attention captures inter-asset dependencies
    - Dirichlet distribution: unbiased gradients on simplex (Ye et al. 2022)
    - Cash position allows risk-off allocation
    - Shared encoder handles dynamic universe (variable n_tradable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-6


# =============================================================================
# PER-ASSET LSTM TEMPORAL ENCODER
# =============================================================================

class AssetTemporalEncoder(nn.Module):
    """
    Shared LSTM encoder that processes each asset's lookback window.
    Input:  (batch, n_tradable, W, F) — W timesteps, F features per asset
    Output: (batch, n_tradable, embed_dim) — one embedding per asset
    """

    def __init__(self, n_features: int = 7, hidden_dim: int = 64,
                 embed_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim) if hidden_dim != embed_dim else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_assets, W, F = x.shape
        x_flat = x.reshape(batch * n_assets, W, F)
        _, (h_n, _) = self.lstm(x_flat)
        h_last = h_n[-1]  # (batch*n_assets, hidden)
        embeds = self.proj(h_last).reshape(batch, n_assets, -1)
        return self.norm(embeds)


# =============================================================================
# CROSS-SECTIONAL MULTI-HEAD ATTENTION
# =============================================================================

class CrossSectionalAttention(nn.Module):
    """
    Multi-head attention over asset embeddings with a learnable global token.
    Input:  (batch, n_tradable, embed_dim)
    Output: (batch, output_dim) fixed-size global + (batch, n_tradable, embed_dim) per-asset
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4,
                 n_global_features: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            batch_first=True, dropout=0.1,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(n_global_features, embed_dim), nn.ReLU(),
        )
        self.weight_stats_dim = 4
        self.weight_proj = nn.Sequential(
            nn.Linear(self.weight_stats_dim, embed_dim // 2), nn.ReLU(),
        )
        self.output_dim = embed_dim + embed_dim + embed_dim // 2
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, asset_embeds, global_features, weight_stats):
        batch = asset_embeds.shape[0]
        global_tok = self.global_token.expand(batch, -1, -1)
        tokens = torch.cat([global_tok, asset_embeds], dim=1)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        global_out = tokens[:, 0, :]
        asset_out = tokens[:, 1:, :]
        global_feat_emb = self.global_proj(global_features)
        weight_emb = self.weight_proj(weight_stats)
        global_repr = self.output_norm(
            torch.cat([global_out, global_feat_emb, weight_emb], dim=1)
        )
        return global_repr, asset_out


# =============================================================================
# STATE PROCESSOR v2
# =============================================================================

class StateProcessorV2(nn.Module):
    """Parses structured state dict → (global_repr, per-asset embeds)."""

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4):
        super().__init__()
        self.temporal_encoder = AssetTemporalEncoder(
            n_features=n_asset_features, hidden_dim=lstm_hidden,
            embed_dim=embed_dim,
        )
        self.attention = CrossSectionalAttention(
            embed_dim=embed_dim, n_heads=n_attn_heads,
            n_global_features=n_global_features,
        )
        self.output_dim = self.attention.output_dim

    def forward(self, state_dict: dict):
        asset_feats = state_dict['asset_features']
        global_feats = state_dict['global_features']
        weights = state_dict['weights']
        n_tradable = state_dict['n_tradable']

        asset_embeds = self.temporal_encoder(asset_feats)

        stock_w = weights[:, :n_tradable]
        w_mean = stock_w.mean(dim=1, keepdim=True)
        w_std = stock_w.std(dim=1, keepdim=True).clamp(min=1e-8)
        w_max = stock_w.max(dim=1, keepdim=True).values
        w_safe = stock_w.clamp(min=1e-8)
        w_entropy = -(w_safe * w_safe.log()).sum(dim=1, keepdim=True)
        weight_stats = torch.cat([w_mean, w_std, w_max, w_entropy], dim=1)

        global_repr, asset_embeds_upd = self.attention(
            asset_embeds, global_feats, weight_stats,
        )
        return global_repr, asset_embeds_upd


# =============================================================================
# DIRICHLET ACTOR
# =============================================================================

class DirichletActor(nn.Module):
    """
    SAC actor with Dirichlet policy. Outputs N+1 weights (N stocks + cash).
    Concentration params via softplus ensure valid Dirichlet parameters.
    """

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 scorer_hidden=128, min_concentration=0.01):
        super().__init__()
        self.state_processor = StateProcessorV2(
            n_asset_features, n_global_features,
            lstm_hidden, embed_dim, n_attn_heads,
        )
        self.embed_dim = embed_dim
        self.min_concentration = min_concentration
        state_dim = self.state_processor.output_dim

        self.scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, 1),
        )
        self.cash_scorer = nn.Sequential(
            nn.Linear(state_dim, scorer_hidden // 2), nn.ReLU(),
            nn.Linear(scorer_hidden // 2, 1),
        )

    def _get_concentrations(self, state_dict):
        global_repr, asset_embeds = self.state_processor(state_dict)
        batch, n_assets, _ = asset_embeds.shape
        global_exp = global_repr.unsqueeze(1).expand(-1, n_assets, -1)
        combined = torch.cat([global_exp, asset_embeds], dim=2)
        asset_scores = self.scorer(combined).squeeze(-1)
        cash_score = self.cash_scorer(global_repr).squeeze(-1)
        all_scores = torch.cat([asset_scores, cash_score.unsqueeze(1)], dim=1)
        alphas = F.softplus(all_scores) + self.min_concentration
        return alphas, global_repr, asset_embeds

    def forward(self, state_dict):
        alphas, _, _ = self._get_concentrations(state_dict)
        return alphas

    def sample(self, state_dict):
        alphas, _, _ = self._get_concentrations(state_dict)
        gamma_samples = torch._standard_gamma(alphas).clamp(min=EPSILON)
        weights = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        weights = weights.clamp(min=EPSILON, max=1.0 - EPSILON)
        log_prob = self._dirichlet_log_prob(weights, alphas)
        mean_w = alphas / alphas.sum(dim=1, keepdim=True)
        return weights, log_prob, mean_w

    def get_deterministic_action(self, state_dict):
        alphas, _, _ = self._get_concentrations(state_dict)
        return alphas / alphas.sum(dim=1, keepdim=True)

    @staticmethod
    def _dirichlet_log_prob(x, alpha):
        alpha_sum = alpha.sum(dim=1, keepdim=True)
        log_B = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
        return -log_B + ((alpha - 1) * torch.log(x.clamp(min=EPSILON))).sum(dim=1, keepdim=True)

    def entropy(self, state_dict):
        alphas, _, _ = self._get_concentrations(state_dict)
        K = alphas.shape[1]
        alpha_sum = alphas.sum(dim=1, keepdim=True)
        log_B = torch.lgamma(alphas).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
        return log_B + (alpha_sum - K) * torch.digamma(alpha_sum) \
               - ((alphas - 1) * torch.digamma(alphas)).sum(dim=1, keepdim=True)


# =============================================================================
# CRITIC — Twin Q-Networks
# =============================================================================

class Critic(nn.Module):
    """Twin critics: state + weight stats → Q-values."""

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 critic_hidden=256, action_stats_dim=6):
        super().__init__()
        self.state_processor = StateProcessorV2(
            n_asset_features, n_global_features,
            lstm_hidden, embed_dim, n_attn_heads,
        )
        self.action_stats_dim = action_stats_dim
        input_dim = self.state_processor.output_dim + action_stats_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, critic_hidden), nn.LayerNorm(critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, critic_hidden), nn.LayerNorm(critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, critic_hidden), nn.LayerNorm(critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, critic_hidden), nn.LayerNorm(critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

    def _action_stats(self, weights, n_tradable):
        stock_w = weights[:, :n_tradable]
        cash_w = weights[:, n_tradable:]
        a_mean = stock_w.mean(dim=1, keepdim=True)
        a_std = stock_w.std(dim=1, keepdim=True).clamp(min=1e-8)
        a_min = stock_w.min(dim=1, keepdim=True).values
        a_max = stock_w.max(dim=1, keepdim=True).values
        w_safe = stock_w.clamp(min=1e-8)
        a_ent = -(w_safe * w_safe.log()).sum(dim=1, keepdim=True)
        return torch.cat([a_mean, a_std, a_min, a_max, a_ent, cash_w], dim=1)

    def forward(self, state_dict, weights):
        global_repr, _ = self.state_processor(state_dict)
        action_repr = self._action_stats(weights, state_dict['n_tradable'])
        x = torch.cat([global_repr, action_repr], dim=1)
        return self.q1(x), self.q2(x)
