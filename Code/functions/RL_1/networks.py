"""
networks.py — Neural Network Architectures for HRA-SAC Portfolio Agent
=========================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Architecture (Hierarchical Risk-Aware SAC):
    Per-asset LSTM temporal encoder (W=20 lookback) →
    Cross-sectional multi-head attention with global token →
    TWO-LEVEL policy heads:
        Level 1: Cash timing head (Gaussian+sigmoid → equity fraction)
        Level 2: Stock selection head (Dirichlet-N → within-equity weights)
    Twin critics with hierarchical action representation

Key novelty:
    - Hierarchical action decomposition separates timing from selection
    - Cash timing is 1D (easy to learn from macro signals)
    - Stock selection is Dirichlet-N (no cash component — cleaner simplex)
    - Gradient separation: timing head learns from global features,
      selection head learns from per-asset features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-6


# =============================================================================
# PER-ASSET LSTM TEMPORAL ENCODER (unchanged from v2)
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
# CROSS-SECTIONAL MULTI-HEAD ATTENTION (unchanged from v2)
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
# STATE PROCESSOR v2 (unchanged)
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
# CASH TIMING HEAD (NEW — Level 1 of hierarchical policy)
# =============================================================================

class CashTimingHead(nn.Module):
    """
    Learns equity fraction via Gaussian + sigmoid squashing.
    Input:  global_repr (batch, state_dim)
    Output: equity_fraction ∈ [min_equity, max_equity]

    Uses reparameterized Gaussian → sigmoid → affine rescaling.
    Log-prob includes Jacobian correction for the sigmoid transform.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 min_equity: float = 0.1, max_equity: float = 1.0):
        super().__init__()
        self.min_equity = min_equity
        self.max_equity = max_equity
        self.range = max_equity - min_equity

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.log_std_head = nn.Linear(hidden_dim // 2, 1)

        # Initialize mu bias so sigmoid(2.0)=0.88 → equity_frac ≈ 0.89
        # This starts the agent mostly invested (sensible default)
        # and lets it learn to reduce equity when conditions warrant.
        nn.init.constant_(self.mu_head.bias, 2.0)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # start with low variance

    def forward(self, global_repr):
        """Returns (mu, log_std) for the pre-sigmoid Gaussian."""
        h = self.net(global_repr)
        mu = self.mu_head(h)          # (batch, 1)
        log_std = self.log_std_head(h).clamp(-5, 2)  # (batch, 1)
        return mu, log_std

    def sample(self, global_repr):
        """Sample equity_fraction with reparameterized gradients."""
        mu, log_std = self.forward(global_repr)
        std = log_std.exp()

        # Reparameterized sample: z = mu + std * eps
        eps = torch.randn_like(mu)
        z = mu + std * eps

        # Sigmoid squash to [0, 1]
        sigmoid_z = torch.sigmoid(z)
        # Rescale to [min_equity, max_equity]
        equity_frac = self.min_equity + self.range * sigmoid_z

        # Log-prob with Jacobian corrections
        # log p(z) for Gaussian
        log_prob = -0.5 * ((z - mu) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        # Jacobian of sigmoid: |d(sigmoid)/dz| = sigmoid * (1 - sigmoid)
        log_prob = log_prob - torch.log(sigmoid_z * (1 - sigmoid_z) + 1e-6)
        # Jacobian of affine rescaling: |d(equity)/d(sigmoid)| = range
        log_prob = log_prob - np.log(self.range)

        return equity_frac, log_prob  # both (batch, 1)

    def get_deterministic(self, global_repr):
        """Mean action (no sampling)."""
        mu, _ = self.forward(global_repr)
        sigmoid_mu = torch.sigmoid(mu)
        return self.min_equity + self.range * sigmoid_mu

    def entropy(self, global_repr):
        """Gaussian entropy (pre-transform, used for alpha tuning)."""
        _, log_std = self.forward(global_repr)
        # Entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2) = 0.5 + log_std + 0.5*log(2*pi)
        return 0.5 * (1.0 + 2 * log_std + np.log(2 * np.pi))  # (batch, 1)


# =============================================================================
# DIRICHLET ACTOR — Hierarchical version
# =============================================================================

class DirichletActor(nn.Module):
    """
    HRA-SAC actor with hierarchical policy:
        Level 1: CashTimingHead → equity_fraction ∈ [0.1, 1.0]
        Level 2: Dirichlet-N → stock weights (N stocks, no cash)

    Combined output: [equity_frac * stock_w, 1 - equity_frac]  (N+1 dim)
    This maintains backward compatibility with the environment interface.

    When hierarchical=False, falls back to flat Dirichlet-(N+1) (for ablation).
    """

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 scorer_hidden=128, min_concentration=0.01,
                 hierarchical=True, cash_head_hidden=64,
                 min_equity=0.1, max_equity=1.0):
        super().__init__()
        self.state_processor = StateProcessorV2(
            n_asset_features, n_global_features,
            lstm_hidden, embed_dim, n_attn_heads,
        )
        self.embed_dim = embed_dim
        self.min_concentration = min_concentration
        self.hierarchical = hierarchical
        state_dim = self.state_processor.output_dim

        # Stock selection scorer (per-asset Dirichlet concentrations)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, scorer_hidden),
            nn.LayerNorm(scorer_hidden), nn.ReLU(),
            nn.Linear(scorer_hidden, 1),
        )

        if self.hierarchical:
            # Level 1: Cash timing head
            self.cash_timing = CashTimingHead(
                input_dim=state_dim,
                hidden_dim=cash_head_hidden,
                min_equity=min_equity,
                max_equity=max_equity,
            )
        else:
            # Flat mode: cash is the (N+1)th Dirichlet component
            self.cash_scorer = nn.Sequential(
                nn.Linear(state_dim, scorer_hidden // 2), nn.ReLU(),
                nn.Linear(scorer_hidden // 2, 1),
            )

    def _get_stock_concentrations(self, state_dict):
        """Get Dirichlet concentration params for N stocks only."""
        global_repr, asset_embeds = self.state_processor(state_dict)
        batch, n_assets, _ = asset_embeds.shape
        global_exp = global_repr.unsqueeze(1).expand(-1, n_assets, -1)
        combined = torch.cat([global_exp, asset_embeds], dim=2)
        asset_scores = self.scorer(combined).squeeze(-1)  # (batch, n_assets)
        alphas = F.softplus(asset_scores) + self.min_concentration
        return alphas, global_repr, asset_embeds

    def _get_flat_concentrations(self, state_dict):
        """Get Dirichlet concentration params for N+1 (stocks + cash)."""
        alphas, global_repr, asset_embeds = self._get_stock_concentrations(state_dict)
        cash_score = self.cash_scorer(global_repr).squeeze(-1)  # (batch,)
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
        """Hierarchical: separate timing + selection sampling."""
        stock_alphas, global_repr, _ = self._get_stock_concentrations(state_dict)

        # Level 1: Cash timing
        equity_frac, timing_log_prob = self.cash_timing.sample(global_repr)  # (B,1)

        # Level 2: Stock selection via Dirichlet-N
        gamma_samples = torch._standard_gamma(stock_alphas).clamp(min=EPSILON)
        stock_weights = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        stock_weights = stock_weights.clamp(min=EPSILON, max=1.0 - EPSILON)
        stock_log_prob = self._dirichlet_log_prob(stock_weights, stock_alphas)  # (B,1)

        # Combine: [equity_frac * stock_w, 1 - equity_frac]
        cash_frac = 1.0 - equity_frac  # (B, 1)
        combined_weights = torch.cat([
            equity_frac * stock_weights,   # (B, N)
            cash_frac,                     # (B, 1)
        ], dim=1)

        # Combined log-prob (independent components)
        log_prob = timing_log_prob + stock_log_prob  # (B, 1)

        # Mean weights (for logging)
        mean_stock = stock_alphas / stock_alphas.sum(dim=1, keepdim=True)
        mean_equity = self.cash_timing.get_deterministic(global_repr)
        mean_weights = torch.cat([
            mean_equity * mean_stock,
            1.0 - mean_equity,
        ], dim=1)

        return combined_weights, log_prob, mean_weights

    def _sample_flat(self, state_dict):
        """Flat Dirichlet-(N+1) sampling (for ablation)."""
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
            equity_frac = self.cash_timing.get_deterministic(global_repr)  # (B,1)
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
        """Combined entropy for alpha tuning."""
        if self.hierarchical:
            stock_alphas, global_repr, _ = self._get_stock_concentrations(state_dict)
            # Dirichlet entropy over N stocks
            K = stock_alphas.shape[1]
            alpha_sum = stock_alphas.sum(dim=1, keepdim=True)
            log_B = torch.lgamma(stock_alphas).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
            dir_entropy = log_B + (alpha_sum - K) * torch.digamma(alpha_sum) \
                          - ((stock_alphas - 1) * torch.digamma(stock_alphas)).sum(dim=1, keepdim=True)
            # Timing entropy
            timing_entropy = self.cash_timing.entropy(global_repr)  # (B, 1)
            return dir_entropy + timing_entropy
        else:
            alphas, _, _ = self._get_flat_concentrations(state_dict)
            K = alphas.shape[1]
            alpha_sum = alphas.sum(dim=1, keepdim=True)
            log_B = torch.lgamma(alphas).sum(dim=1, keepdim=True) - torch.lgamma(alpha_sum)
            return log_B + (alpha_sum - K) * torch.digamma(alpha_sum) \
                   - ((alphas - 1) * torch.digamma(alphas)).sum(dim=1, keepdim=True)


# =============================================================================
# CRITIC — Twin Q-Networks (minor update: equity_frac in action_stats)
# =============================================================================

class Critic(nn.Module):
    """Twin critics: state + action stats → Q-values."""

    def __init__(self, n_asset_features=7, n_global_features=5,
                 lstm_hidden=64, embed_dim=64, n_attn_heads=4,
                 critic_hidden=256, action_stats_dim=7):
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
        # Equity fraction = 1 - cash
        equity_frac = 1.0 - cash_w
        a_mean = stock_w.mean(dim=1, keepdim=True)
        a_std = stock_w.std(dim=1, keepdim=True).clamp(min=1e-8)
        a_min = stock_w.min(dim=1, keepdim=True).values
        a_max = stock_w.max(dim=1, keepdim=True).values
        w_safe = stock_w.clamp(min=1e-8)
        a_ent = -(w_safe * w_safe.log()).sum(dim=1, keepdim=True)
        return torch.cat([a_mean, a_std, a_min, a_max, a_ent, cash_w, equity_frac], dim=1)

    def forward(self, state_dict, weights):
        global_repr, _ = self.state_processor(state_dict)
        action_repr = self._action_stats(weights, state_dict['n_tradable'])
        x = torch.cat([global_repr, action_repr], dim=1)
        return self.q1(x), self.q2(x)
