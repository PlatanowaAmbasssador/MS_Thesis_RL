"""
networks.py — Neural Network Architectures for HRA-SAC Portfolio Agent
=========================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Run 10 ADDITION: GaussianActor for long-short-cash strategy
    - Outputs unconstrained weights via Gaussian (mu, log_std per asset)
    - Positive weight = long, negative = short
    - Cash always non-negative via softplus
    - Gross exposure normalized to 1.0 (|longs| + |shorts| + cash = 1)
    
Existing DirichletActor and Critic are UNCHANGED for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-6


# =============================================================================
# PER-ASSET LSTM TEMPORAL ENCODER (unchanged)
# =============================================================================

class AssetTemporalEncoder(nn.Module):
    """
    Shared LSTM encoder that processes each asset's lookback window.
    Input:  (batch, n_tradable, W, F) — W timesteps, F features per asset
    Output: (batch, n_tradable, embed_dim) — one embedding per asset
    """

    def __init__(self, n_features: int = 7, hidden_dim: int = 64,
                 embed_dim: int = 64, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_dim, embed_dim) if hidden_dim != embed_dim else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_assets, W, F = x.shape
        x_flat = x.reshape(batch * n_assets, W, F)
        _, (h_n, _) = self.lstm(x_flat)
        h_last = h_n[-1]  # (batch*n_assets, hidden)
        h_last = self.dropout(h_last)
        embeds = self.proj(h_last).reshape(batch, n_assets, -1)
        return self.norm(embeds)


# =============================================================================
# CROSS-SECTIONAL MULTI-HEAD ATTENTION (unchanged)
# =============================================================================

class CrossSectionalAttention(nn.Module):
    """
    Multi-head attention over per-asset embeddings + learnable global token.
    Input: per-asset embeds (batch, N, embed_dim), global features, weight stats
    Output: global_repr (batch, output_dim), updated_asset_embeds (batch, N, embed_dim)
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4,
                 n_global_features: int = 5, n_weight_stats: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.global_proj = nn.Sequential(
            nn.Linear(n_global_features, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(),
        )
        self.weight_proj = nn.Sequential(
            nn.Linear(n_weight_stats, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(),
        )
        self.output_norm = nn.LayerNorm(embed_dim * 3)
        self.output_dim = embed_dim * 3

    def forward(self, asset_embeds, global_features, weight_stats):
        batch = asset_embeds.shape[0]
        global_tok = self.global_token.expand(batch, -1, -1)
        tokens = torch.cat([global_tok, asset_embeds], dim=1)
        tokens, _ = self.attn(tokens, tokens, tokens)
        global_out = tokens[:, 0, :]
        asset_out = tokens[:, 1:, :]
        global_feat_emb = self.global_proj(global_features)
        weight_emb = self.weight_proj(weight_stats)
        global_repr = self.output_norm(
            torch.cat([global_out, global_feat_emb, weight_emb], dim=1)
        )
        return global_repr, asset_out


# =============================================================================
# STATE PROCESSOR v2 (unchanged)
# =============================================================================

class StateProcessorV2(nn.Module):
    """Parses structured state dict → (global_repr, per-asset embeds)."""

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4, dropout=0.0):
        super().__init__()
        self.temporal_encoder = AssetTemporalEncoder(
            n_features=n_asset_features, hidden_dim=lstm_hidden,
            embed_dim=embed_dim, dropout=dropout,
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
# CASH TIMING HEAD (unchanged — used by hierarchical DirichletActor)
# =============================================================================

class CashTimingHead(nn.Module):
    """
    Learns equity fraction via Gaussian + sigmoid squashing.
    Input:  global_repr (batch, state_dim)
    Output: equity_fraction ∈ [min_equity, max_equity]
    """

    def __init__(self, input_dim, hidden_dim=64, min_equity=0.1, max_equity=1.0):
        super().__init__()
        self.min_equity = min_equity
        self.max_equity = max_equity
        self.range = max_equity - min_equity
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.log_std_head = nn.Linear(hidden_dim // 2, 1)
        # Bias mu toward high equity (invest, don't hide in cash)
        nn.init.constant_(self.mu_head.bias, 2.0)

    def forward(self, global_repr):
        h = self.net(global_repr)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-5, 1)
        return mu, log_std

    def sample(self, global_repr):
        mu, log_std = self.forward(global_repr)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        z = mu + std * eps
        sigmoid_z = torch.sigmoid(z)
        equity_frac = self.min_equity + self.range * sigmoid_z
        log_prob = -0.5 * ((z - mu) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob - torch.log(sigmoid_z * (1 - sigmoid_z) + 1e-6)
        log_prob = log_prob - np.log(self.range)
        return equity_frac, log_prob

    def get_deterministic(self, global_repr):
        mu, _ = self.forward(global_repr)
        sigmoid_mu = torch.sigmoid(mu)
        return self.min_equity + self.range * sigmoid_mu

    def entropy(self, global_repr):
        _, log_std = self.forward(global_repr)
        return 0.5 * (1.0 + 2 * log_std + np.log(2 * np.pi))


# =============================================================================
# DIRICHLET ACTOR — Long-only (unchanged from Run 9)
# =============================================================================

class DirichletActor(nn.Module):
    """
    HRA-SAC actor with hierarchical policy:
        Level 1: CashTimingHead → equity_fraction ∈ [0.1, 1.0]
        Level 2: Dirichlet-N → stock weights (N stocks, no cash)

    Combined output: [equity_frac * stock_w, 1 - equity_frac]  (N+1 dim)
    When hierarchical=False, falls back to flat Dirichlet-(N+1).
    """

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 scorer_hidden=128, min_concentration=0.01,
                 hierarchical=True, cash_head_hidden=64,
                 min_equity=0.1, max_equity=1.0, dropout=0.0):
        super().__init__()
        self.state_processor = StateProcessorV2(
            n_asset_features, n_global_features,
            lstm_hidden, embed_dim, n_attn_heads, dropout=dropout,
        )
        self.embed_dim = embed_dim
        self.min_concentration = min_concentration
        self.hierarchical = hierarchical
        state_dim = self.state_processor.output_dim

        self.scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, scorer_hidden // 2),
            nn.LayerNorm(scorer_hidden // 2), nn.ReLU(),
            nn.Linear(scorer_hidden // 2, 1),
        )

        if self.hierarchical:
            self.cash_timing = CashTimingHead(
                input_dim=state_dim,
                hidden_dim=cash_head_hidden,
                min_equity=min_equity,
                max_equity=max_equity,
            )
        else:
            self.cash_scorer = nn.Sequential(
                nn.Linear(state_dim, scorer_hidden // 2), nn.ReLU(),
                nn.Linear(scorer_hidden // 2, 1),
            )

    def _get_stock_concentrations(self, state_dict):
        global_repr, asset_embeds = self.state_processor(state_dict)
        batch, n_assets, _ = asset_embeds.shape
        global_exp = global_repr.unsqueeze(1).expand(-1, n_assets, -1)
        combined = torch.cat([global_exp, asset_embeds], dim=2)
        asset_scores = self.scorer(combined).squeeze(-1)
        alphas = F.softplus(asset_scores) + self.min_concentration
        return alphas, global_repr, asset_embeds

    def _get_flat_concentrations(self, state_dict):
        alphas, global_repr, asset_embeds = self._get_stock_concentrations(state_dict)
        cash_score = self.cash_scorer(global_repr).squeeze(-1)
        cash_alpha = F.softplus(cash_score) + self.min_concentration
        all_alphas = torch.cat([alphas, cash_alpha.unsqueeze(1)], dim=1)
        return all_alphas, global_repr, asset_embeds

    def forward(self, state_dict):
        if self.hierarchical:
            alphas, global_repr, _ = self._get_stock_concentrations(state_dict)
            return alphas
        else:
            alphas, _, _ = self._get_flat_concentrations(state_dict)
            return alphas

    def sample(self, state_dict):
        if self.hierarchical:
            return self._sample_hierarchical(state_dict)
        else:
            return self._sample_flat(state_dict)

    def _sample_hierarchical(self, state_dict):
        stock_alphas, global_repr, _ = self._get_stock_concentrations(state_dict)
        equity_frac, timing_log_prob = self.cash_timing.sample(global_repr)
        gamma_samples = torch._standard_gamma(stock_alphas).clamp(min=EPSILON)
        stock_weights = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        stock_weights = stock_weights.clamp(min=EPSILON, max=1.0 - EPSILON)
        stock_log_prob = self._dirichlet_log_prob(stock_weights, stock_alphas)
        cash_frac = 1.0 - equity_frac
        combined_weights = torch.cat([
            equity_frac * stock_weights, cash_frac,
        ], dim=1)
        log_prob = timing_log_prob + stock_log_prob
        mean_stock = stock_alphas / stock_alphas.sum(dim=1, keepdim=True)
        mean_equity = self.cash_timing.get_deterministic(global_repr)
        mean_weights = torch.cat([
            mean_equity * mean_stock, 1.0 - mean_equity,
        ], dim=1)
        return combined_weights, log_prob, mean_weights

    def _sample_flat(self, state_dict):
        alphas, _, _ = self._get_flat_concentrations(state_dict)
        gamma_samples = torch._standard_gamma(alphas).clamp(min=EPSILON)
        weights = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        weights = weights.clamp(min=EPSILON, max=1.0 - EPSILON)
        log_prob = self._dirichlet_log_prob(weights, alphas)
        mean_w = alphas / alphas.sum(dim=1, keepdim=True)
        return weights, log_prob, mean_w

    def get_deterministic_action(self, state_dict):
        if self.hierarchical:
            stock_alphas, global_repr, _ = self._get_stock_concentrations(state_dict)
            mean_stock = stock_alphas / stock_alphas.sum(dim=1, keepdim=True)
            equity_frac = self.cash_timing.get_deterministic(global_repr)
            return torch.cat([equity_frac * mean_stock, 1.0 - equity_frac], dim=1)
        else:
            alphas, _, _ = self._get_flat_concentrations(state_dict)
            return alphas / alphas.sum(dim=1, keepdim=True)

    @staticmethod
    def _dirichlet_log_prob(x, alpha):
        alpha_sum = alpha.sum(dim=1, keepdim=True)
        log_B = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
        return -log_B + ((alpha - 1) * torch.log(x.clamp(min=EPSILON))).sum(dim=1, keepdim=True)

    def entropy(self, state_dict):
        if self.hierarchical:
            stock_alphas, global_repr, _ = self._get_stock_concentrations(state_dict)
            K = stock_alphas.shape[1]
            alpha_sum = stock_alphas.sum(dim=1, keepdim=True)
            log_B = torch.lgamma(stock_alphas).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
            dir_entropy = log_B + (alpha_sum - K) * torch.digamma(alpha_sum) \
                          - ((stock_alphas - 1) * torch.digamma(stock_alphas)).sum(dim=1, keepdim=True)
            timing_entropy = self.cash_timing.entropy(global_repr)
            return dir_entropy + timing_entropy
        else:
            alphas, _, _ = self._get_flat_concentrations(state_dict)
            K = alphas.shape[1]
            alpha_sum = alphas.sum(dim=1, keepdim=True)
            log_B = torch.lgamma(alphas).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
            return log_B + (alpha_sum - K) * torch.digamma(alpha_sum) \
                   - ((alphas - 1) * torch.digamma(alphas)).sum(dim=1, keepdim=True)


# =============================================================================
# GAUSSIAN ACTOR — Long-Short-Cash (NEW for Run 10)
# =============================================================================

class GaussianActor(nn.Module):
    """
    Long-short-cash actor using Gaussian policy.
    
    Architecture:
        Same LSTM + attention backbone as DirichletActor.
        Per-asset scorer outputs mu (signal) instead of concentration.
        Separate log_std head (shared across assets for stability).
        Cash head outputs non-negative cash weight via softplus.
    
    Action space:
        raw[:N] ~ N(mu, std)  — positive=long, negative=short
        raw[N]  = softplus(cash_score) — always non-negative
    
    Normalization to portfolio weights:
        gross = |raw[:N]|.sum() + cash
        stock_w = raw[:N] / gross   (can be negative!)
        cash_w  = cash / gross      (always positive)
        Result: |stock_w|.sum() + cash_w = 1.0
    
    This allows the agent to learn long-short strategies without
    any margin/borrow modeling — shorts are just negative weights.
    The portfolio return is Σ(w_i * r_i) which naturally handles shorts:
        Short INTC: w=-0.05, r=-0.02 → contribution = +0.001 (profit)
    """
    
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0  # std ∈ [0.007, 2.7]
    
    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 scorer_hidden=128, dropout=0.0, **kwargs):
        super().__init__()
        # Same backbone as Dirichlet
        self.state_processor = StateProcessorV2(
            n_asset_features, n_global_features,
            lstm_hidden, embed_dim, n_attn_heads, dropout=dropout,
        )
        self.embed_dim = embed_dim
        state_dim = self.state_processor.output_dim
        
        # Per-asset mu head: global_repr + asset_embed → mu_i
        self.mu_scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, scorer_hidden // 2),
            nn.LayerNorm(scorer_hidden // 2), nn.ReLU(),
            nn.Linear(scorer_hidden // 2, 1),
        )
        
        # Shared log_std: learned from global state (not per-asset)
        # This keeps exploration uniform across assets → more stable
        self.log_std_head = nn.Sequential(
            nn.Linear(state_dim, scorer_hidden // 2), nn.ReLU(),
            nn.Linear(scorer_hidden // 2, 1),
        )
        
        # Cash head: global_repr → cash_score (softplus → always ≥ 0)
        self.cash_head = nn.Sequential(
            nn.Linear(state_dim, scorer_hidden // 2),
            nn.LayerNorm(scorer_hidden // 2), nn.ReLU(),
            nn.Linear(scorer_hidden // 2, 1),
        )
        # Initialize cash head bias so cash starts at ~10%
        nn.init.constant_(self.cash_head[-1].bias, -1.0)
    
    def _get_mu_logstd_cash(self, state_dict):
        """Compute per-asset mu, shared log_std, and cash score."""
        global_repr, asset_embeds = self.state_processor(state_dict)
        batch, n_assets, _ = asset_embeds.shape
        
        # Per-asset mu
        global_exp = global_repr.unsqueeze(1).expand(-1, n_assets, -1)
        combined = torch.cat([global_exp, asset_embeds], dim=2)
        mu = self.mu_scorer(combined).squeeze(-1)  # (B, N)
        
        # Shared log_std (broadcast to all assets)
        log_std = self.log_std_head(global_repr)  # (B, 1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        log_std = log_std.expand(-1, n_assets)  # (B, N)
        
        # Cash score
        cash_score = self.cash_head(global_repr).squeeze(-1)  # (B,)
        
        return mu, log_std, cash_score, global_repr
    
    def _normalize_weights(self, stock_raw, cash_raw):
        """
        Normalize raw outputs to portfolio weights.
        stock_raw can be negative (short), cash_raw is always ≥ 0.
        Result: |stock_w|.sum() + cash_w = 1.0
        """
        cash = F.softplus(cash_raw).unsqueeze(-1) + EPSILON  # (B, 1)
        gross_stock = stock_raw.abs().sum(dim=1, keepdim=True) + EPSILON  # (B, 1)
        gross_total = gross_stock + cash  # (B, 1)
        
        stock_w = stock_raw / gross_total   # (B, N) — can be negative!
        cash_w = cash / gross_total         # (B, 1) — always positive
        
        # Concatenate: [stock_w, cash_w] = (B, N+1)
        weights = torch.cat([stock_w, cash_w], dim=1)
        return weights
    
    def sample(self, state_dict):
        """Sample action via reparameterization trick."""
        mu, log_std, cash_score, _ = self._get_mu_logstd_cash(state_dict)
        std = log_std.exp()
        
        # Reparameterized sample
        eps = torch.randn_like(mu)
        stock_raw = mu + std * eps  # (B, N)
        
        # Log-prob of Gaussian
        log_prob_per_asset = -0.5 * ((stock_raw - mu) / (std + 1e-8)) ** 2 \
                             - log_std - 0.5 * np.log(2 * np.pi)
        # Sum over assets for total log-prob
        log_prob = log_prob_per_asset.sum(dim=1, keepdim=True)  # (B, 1)
        
        # Normalize to portfolio weights
        weights = self._normalize_weights(stock_raw, cash_score)
        
        # Mean weights for logging
        mean_weights = self._normalize_weights(mu, cash_score)
        
        return weights, log_prob, mean_weights
    
    def get_deterministic_action(self, state_dict):
        """Mean action (no sampling) — use at test time."""
        mu, _, cash_score, _ = self._get_mu_logstd_cash(state_dict)
        return self._normalize_weights(mu, cash_score)
    
    def entropy(self, state_dict):
        """
        Gaussian entropy: H = 0.5 * N * (1 + log(2π)) + Σ log_std_i
        Used for SAC alpha tuning.
        """
        _, log_std, _, _ = self._get_mu_logstd_cash(state_dict)
        N = log_std.shape[1]
        # Entropy per dimension: 0.5 * (1 + log(2π)) + log_std
        per_dim = 0.5 * (1.0 + np.log(2 * np.pi)) + log_std
        return per_dim.sum(dim=1, keepdim=True)  # (B, 1)


# =============================================================================
# CRITIC — Twin Q-Networks (updated to handle negative weights)
# =============================================================================

class Critic(nn.Module):
    """Twin critics: state + action stats → Q-values.
    
    Run 10: action_stats updated to handle long-short weights:
        - Uses absolute weights for some stats (HHI, concentration)
        - Adds net_exposure and gross_exposure as features
    """

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 critic_hidden=256, action_stats_dim=7, dropout=0.0):
        super().__init__()
        self.state_processor = StateProcessorV2(
            n_asset_features, n_global_features,
            lstm_hidden, embed_dim, n_attn_heads, dropout=dropout,
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
        
        # These stats work for both long-only and long-short:
        a_mean = stock_w.mean(dim=1, keepdim=True)
        a_std = stock_w.std(dim=1, keepdim=True).clamp(min=1e-8)
        a_min = stock_w.min(dim=1, keepdim=True).values
        a_max = stock_w.max(dim=1, keepdim=True).values
        
        # Use absolute weights for entropy (works for L/S)
        abs_w = stock_w.abs().clamp(min=1e-8)
        abs_sum = abs_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_norm = abs_w / abs_sum
        a_ent = -(w_norm * w_norm.log()).sum(dim=1, keepdim=True)
        
        # Equity fraction = 1 - cash (net exposure for L/S)
        equity_frac = 1.0 - cash_w
        
        return torch.cat([a_mean, a_std, a_min, a_max, a_ent, cash_w, equity_frac], dim=1)

    def forward(self, state_dict, weights):
        global_repr, _ = self.state_processor(state_dict)
        action_repr = self._action_stats(weights, state_dict['n_tradable'])
        x = torch.cat([global_repr, action_repr], dim=1)
        return self.q1(x), self.q2(x)
