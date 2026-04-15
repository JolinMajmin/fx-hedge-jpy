"""
rl_hedge_training_jpy.py — Stage 3 of USD/JPY Hedge Timing Pipeline
=====================================================================
Reads:
    usdjpy_features.parquet   (from sabr_calibration_jpy.py)

Produces:
    rl_metrics.csv, rl_test_results.parquet, benchmark_results.parquet,
    rl_overall_metrics.json, rl_performance.png, model_window_*.zip,
    optuna_best_w*.json

P&L reporting:
    daily_pnl      — hedge error net of TC (continuity-based, no inception premium).
                     Used for he_vol_reduction metric and reward.
    full_daily_pnl — complete economic P&L including inception premium and expiry
                     settlement. Use this for total strategy P&L reporting.

Tokyo UTC equivalents (JST = UTC+9, Japan has NO DST):
    9am Tokyo  (open)  = 00:00 UTC
    3pm Tokyo  (close) = 06:00 UTC

Usage:
    python -u rl_hedge_training_jpy.py C:\\BackbookData_jpy
    python -u rl_hedge_training_jpy.py C:\\BackbookData_jpy 150000 0
        (data_dir, total_timesteps, n_optuna_trials)

Checkpoint management — delete these to re-run from scratch:
    del model_window_*.zip
    del rl_metrics_w*.json
    del optuna_best_w*.json
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import entropy as calc_entropy
import time, sys, os, json, warnings, logging
from collections import deque

warnings.filterwarnings('ignore')
logging.getLogger('optuna').setLevel(logging.WARNING)

log = logging.getLogger(__name__)

def logprint(msg):
    print(msg, flush=True)
    log.info(msg)


# ════════════════════════════════════════════════════════════════
# SECTION 1: FEATURE SELECTION & NORMALIZATION
# ════════════════════════════════════════════════════════════════

STATE_FEATURES = [
    'atm_1m', 'atm_1m_chg', 'term_slope_3m_1w',
    'rr25_1m', 'rr25_1m_chg', 'bf25_1m',
    'spot_ret', 'carry_3m', 'spread_bps',
    'rho_sv', 'beta_up',
    'h00_intraday_zscore', 'h06_intraday_zscore', 'h16_intraday_zscore',
    'h00_ret_4h', 'h06_ret_4h', 'h16_ret_4h',
    'h00_rv_yz_21d',
    'vr_minus1_21d', 'range_pos_21d',
    'vol_ma_alignment', 'vol_regime_63d',
    'spot_trend_strength', 'h00_to_close_lag1',
]


class FeatureNormalizer:
    def __init__(self, window=252):
        self.window = window
        self.means  = None
        self.stds   = None

    def fit(self, df):
        self.means = df.rolling(self.window, min_periods=20).mean().iloc[-1]
        self.stds  = df.rolling(self.window, min_periods=20).std().iloc[-1].replace(0, 1.0)
        return self

    def transform(self, df):
        return ((df - self.means) / self.stds).clip(-5, 5).fillna(0)


# ════════════════════════════════════════════════════════════════
# SECTION 2: ACTION SPACE
# ════════════════════════════════════════════════════════════════

ACTION_MAP = {
    # ── Single venue: 00 UTC (cheapest) ──────────────────────────────
    0: {'name': 'h100_00',        'f00': 1.0,  'f06': 0.0,  'f16': 0.0},
    1: {'name': 'h75_00',         'f00': 0.75, 'f06': 0.0,  'f16': 0.0},
    # ── Single venue: 06 UTC ─────────────────────────────────────────
    2: {'name': 'h100_06',        'f00': 0.0,  'f06': 1.0,  'f16': 0.0},
    # ── Splits across venue pairs ────────────────────────────────────
    3: {'name': 'split_50_00_16', 'f00': 0.5,  'f06': 0.0,  'f16': 0.5},
    4: {'name': 'split_50_06_16', 'f00': 0.0,  'f06': 0.5,  'f16': 1.0},
    # ── Single venue: 16 UTC (benchmark) ─────────────────────────────
    5: {'name': 'h100_16',        'f00': 0.0,  'f06': 0.0,  'f16': 1.0},
}

# Transaction costs per hedge window (basis points)
TC_BPS = {
    0:  1.6,   # 00 UTC — Tokyo open: tightest JPY spreads
    6:  1.8,   # 06 UTC — Tokyo PM / London pre-open
    16: 1.8,   # 16 UTC — London afternoon / NY overlap
}


# ════════════════════════════════════════════════════════════════
# SECTION 3: HEDGING ENVIRONMENT
# ════════════════════════════════════════════════════════════════

class FXHedgingEnv(gym.Env):

    def __init__(self,
                 state_features,
                 delta_close, gamma_close, price_close,
                 delta_00, delta_06, delta_16,
                 spot_00, spot_06, spot_16, spot_close,
                 continuity_pnl=None,
                 full_pnl=None,
                 reward_scaling=1000.0):
        super().__init__()

        self.state_feat    = state_features.astype(np.float32)
        self.delta_close   = delta_close.astype(np.float64)
        self.gamma_close   = gamma_close.astype(np.float64)
        self.price_close   = price_close.astype(np.float64)
        self.delta_00      = delta_00.astype(np.float64)
        self.delta_06      = delta_06.astype(np.float64)
        self.delta_16      = delta_16.astype(np.float64)
        self.spot_00       = spot_00.astype(np.float64)
        self.spot_06       = spot_06.astype(np.float64)
        self.spot_16       = spot_16.astype(np.float64)
        self.spot_close    = spot_close.astype(np.float64)
        # continuity_pnl: MTM change of continuing legs — used for reward (clean gamma signal)
        self.continuity_pnl = continuity_pnl.astype(np.float64) if continuity_pnl is not None else None
        # full_pnl: complete economic P&L including inception premium — used for reporting only
        self.full_pnl       = full_pnl.astype(np.float64)       if full_pnl       is not None else None

        self.reward_scaling = reward_scaling
        self.n_steps        = len(state_features)

        n_extra = 4
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(state_features.shape[1] + n_extra,),
            dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_MAP))

        self.step_idx    = 0
        self.hedge_pos   = 0.0
        self.steps_since = 0
        self.prev_price  = 0.0
        self.vol_window  = deque(maxlen=21)

    def _obs(self):
        base  = self.state_feat[self.step_idx]
        extra = np.array([
            self.hedge_pos,
            min(self.steps_since / 10.0, 1.0),
            self.delta_close[self.step_idx],
            self.gamma_close[self.step_idx],
        ], dtype=np.float32)
        return np.concatenate([base, extra])

    def step(self, action):
        action = int(np.asarray(action).item())
        cfg    = ACTION_MAP[action]
        i      = self.step_idx

        d00, d06, d16 = self.delta_00[i], self.delta_06[i], self.delta_16[i]
        s00, s06, s16 = self.spot_00[i], self.spot_06[i], self.spot_16[i]
        sc      = self.spot_close[i]
        sc_prev = self.spot_close[i-1] if i > 0 else sc
        H0      = self.hedge_pos

        f00, f06, f16 = cfg['f00'], cfg['f06'], cfg['f16']

        # Cascade: each time slot targets remaining delta mismatch
        mismatch_00 = (-d00) - H0
        dH00        = f00 * mismatch_00
        H_after_00  = H0 + dH00

        mismatch_06 = (-d06) - H_after_00
        dH06        = f06 * mismatch_06
        H_after_06  = H_after_00 + dH06

        mismatch_16 = (-d16) - H_after_06
        dH16        = f16 * mismatch_16

        # option_pnl for REWARD: continuity_pnl = pure gamma P&L, no inception premium
        if self.continuity_pnl is not None:
            option_pnl = self.continuity_pnl[i]
        else:
            option_pnl = self.price_close[i] - self.prev_price if i > 0 else 0.0

        # full_option_pnl for REPORTING: includes inception premium + expiry settlement
        if self.full_pnl is not None:
            full_option_pnl = self.full_pnl[i]
        else:
            full_option_pnl = option_pnl  # fallback: same as continuity

        # Hedge P&L: each component divided by its execution spot to convert JPY → USD.
        # dH × (sc - s_exec) is in JPY; dividing by s_exec gives USD return on delta units.
        # carry P&L: H0 × (sc - sc_prev) / sc_prev — return on carried position in USD.
        carry_pnl   = H0 * (sc - sc_prev) / sc_prev if i > 0 else 0.0
        trade00_pnl = dH00 * (sc - s00) / s00 if s00 > 0 else 0.0
        trade06_pnl = dH06 * (sc - s06) / s06 if s06 > 0 else 0.0
        trade16_pnl = dH16 * (sc - s16) / s16 if s16 > 0 else 0.0
        hedge_pnl   = carry_pnl + trade00_pnl + trade06_pnl + trade16_pnl

        # Transaction costs in USD: bps × delta_units (spot cancels: JPY/10000 / JPY per USD)
        tc00 = abs(dH00) * TC_BPS[0]  / 10000.0
        tc06 = abs(dH06) * TC_BPS[6]  / 10000.0
        tc16 = abs(dH16) * TC_BPS[16] / 10000.0
        tc   = tc00 + tc06 + tc16

        # hedge_error and daily_pnl use continuity_pnl (for HE vol metric and reward)
        hedge_error = option_pnl + hedge_pnl
        daily_pnl   = hedge_error - tc

        # full_daily_pnl uses full_pnl (complete economic P&L for reporting)
        full_daily_pnl = full_option_pnl + hedge_pnl - tc

        # Reward: minimise squared hedge error + TC penalty
        reward = float(-np.log1p(hedge_error**2 * self.reward_scaling) - tc * 10)

        self.hedge_pos   = H0 + dH00 + dH06 + dH16
        self.prev_price  = self.price_close[i]
        total_traded     = abs(dH00) + abs(dH06) + abs(dH16)
        self.steps_since = 0 if total_traded > 1e-10 else self.steps_since + 1
        self.step_idx   += 1
        done = self.step_idx >= self.n_steps - 1

        obs = self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            'daily_pnl':      daily_pnl,       # hedge error net TC — used for he_vol metric
            'full_daily_pnl': full_daily_pnl,  # complete economic P&L incl. inception premium
            'hedge_error':    hedge_error,
            'option_pnl':     option_pnl,
            'carry_pnl':      carry_pnl,
            'trade00_pnl':    trade00_pnl,
            'trade06_pnl':    trade06_pnl,
            'trade16_pnl':    trade16_pnl,
            'hedge_pnl':      hedge_pnl,
            'tc':             tc,
            'tc00':           tc00,
            'tc06':           tc06,
            'tc16':           tc16,
            'dH00':           dH00,
            'dH06':           dH06,
            'dH16':           dH16,
            'delta_00':       d00,
            'delta_06':       d06,
            'delta_16':       d16,
            'delta_close':    self.delta_close[i],
            'spot_00':        s00,
            'spot_06':        s06,
            'spot_16':        s16,
            'spot_close':     sc,
            'action_name':    cfg['name'],
        }
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx    = 0
        self.hedge_pos   = 0.0
        self.steps_since = 0
        self.prev_price  = self.price_close[0]
        self.vol_window  = deque(maxlen=21)
        return self._obs(), {}


# ════════════════════════════════════════════════════════════════
# SECTION 4: BENCHMARK  (always hedge at 16 UTC)
# ════════════════════════════════════════════════════════════════

def run_benchmark(delta_16, price_close, spot_16, spot_close,
                  continuity_pnl=None, full_pnl=None):
    """
    Benchmark: hedge 100% of delta at 16 UTC every day.
    daily_pnl      — hedge error net TC (continuity-based, for he_vol metric)
    full_daily_pnl — complete economic P&L including inception premium
    """
    n, H, prev_price = len(spot_close), 0.0, price_close[0]
    records = []
    for i in range(n):
        sc      = spot_close[i]
        sc_prev = spot_close[i-1] if i > 0 else sc

        opt_pnl = continuity_pnl[i] if continuity_pnl is not None else (
            price_close[i] - prev_price if i > 0 else 0.0)

        full_opt_pnl = full_pnl[i] if full_pnl is not None else opt_pnl

        s16_i     = spot_16[i]
        carry_pnl = H * (sc - sc_prev) / sc_prev if i > 0 else 0.0
        target    = -delta_16[i]
        dH        = target - H
        trade_pnl = dH * (sc - s16_i) / s16_i if s16_i > 0 else 0.0
        tc        = abs(dH) * TC_BPS[16] / 10000.0   # spot cancels in JPY/USD
        hedge_pnl = carry_pnl + trade_pnl
        hedge_error    = opt_pnl + hedge_pnl
        daily_pnl      = hedge_error - tc
        full_daily_pnl = full_opt_pnl + hedge_pnl - tc

        H, prev_price = target, price_close[i]
        records.append({
            'daily_pnl':      daily_pnl,
            'full_daily_pnl': full_daily_pnl,
            'hedge_error':    hedge_error,
            'carry_pnl':      carry_pnl,
            'trade_pnl':      trade_pnl,
            'tc':             tc,
            'trade_size':     dH,
            'delta_close':    delta_16[i],
        })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════
# SECTION 5: EVALUATION METRICS
# ════════════════════════════════════════════════════════════════

def evaluate(rl_pnl, bm_pnl):
    rl, bm  = np.array(rl_pnl), np.array(bm_pnl)
    ml      = min(len(rl), len(bm)); rl, bm = rl[:ml], bm[:ml]
    rs, bs  = max(np.std(rl,ddof=1),1e-10), max(np.std(bm,ddof=1),1e-10)
    rl_cum  = np.cumsum(rl); bm_cum = np.cumsum(bm)
    act     = rl - bm; ast = max(np.std(act,ddof=1),1e-10)
    from scipy.stats import f as f_dist
    F_stat      = (bs**2) / (rs**2)
    var_p_value = 1 - f_dist.cdf(F_stat, ml-1, ml-1)
    return {
        'rl_he_vol':        float(rs),
        'bm_he_vol':        float(bs),
        'he_vol_reduction': float(1-rs/bs),
        'rl_mean_abs':      float(np.mean(np.abs(rl))),
        'bm_mean_abs':      float(np.mean(np.abs(bm))),
        'rl_max_loss':      float(np.min(rl)),
        'bm_max_loss':      float(np.min(bm)),
        'rl_total_pnl':     float(np.sum(rl)),
        'bm_total_pnl':     float(np.sum(bm)),
        'rl_max_dd':        float(np.min(rl_cum - np.maximum.accumulate(rl_cum))),
        'bm_max_dd':        float(np.min(bm_cum - np.maximum.accumulate(bm_cum))),
        'information_ratio':float(np.mean(act)/ast*np.sqrt(252)),
        'F_stat':           float(F_stat),
        'var_p_value':      float(var_p_value),
        'n_days':           ml,
    }


# ════════════════════════════════════════════════════════════════
# SECTION 6: WALK-FORWARD SPLITS
# ════════════════════════════════════════════════════════════════

def create_splits(dates):
    sy = dates.min().year; ey = dates.max().year
    splits = []
    for ty in range(sy+6, ey):
        vy = ty-1; te = vy-1
        tr = dates[dates.year <= te]
        va = dates[dates.year == vy]
        ts = dates[dates.year == ty]
        if len(tr)<252 or len(va)<100 or len(ts)<100: continue
        splits.append({'w': len(splits)+1, 'tr_yrs': f"{sy}-{te}",
                        'vy': vy, 'ty': ty, 'tr': tr, 'va': va, 'ts': ts})
    return splits


# ════════════════════════════════════════════════════════════════
# SECTION 7: BUILD ENVIRONMENT FROM FEATURE FILE
# ════════════════════════════════════════════════════════════════

def make_env(feat_df, dates, normalizer, feature_cols, reward_scaling=1000.0):
    f     = feat_df.loc[dates]
    state = normalizer.transform(f[feature_cols]).values
    cont  = f['port_continuity_pnl'].values if 'port_continuity_pnl' in f.columns else None
    full  = f['port_full_pnl'].values        if 'port_full_pnl'       in f.columns else None

    return FXHedgingEnv(
        state_features = state,
        delta_close    = f['port_delta_close'].values,
        gamma_close    = f['port_gamma_close'].values,
        price_close    = f['port_price_close'].values,
        delta_00       = f['port_delta_00'].values,
        delta_06       = f['port_delta_06'].values,
        delta_16       = f['port_delta_16'].values,
        spot_00        = f['spot_00'].values,
        spot_06        = f['spot_06'].values,
        spot_16        = f['spot_16'].values,
        spot_close     = f['spot_close'].values,
        continuity_pnl = cont,
        full_pnl       = full,
        reward_scaling = reward_scaling,
    )


def run_episode(env, model=None):
    obs, _      = env.reset()
    records     = []; done = False
    lstm_states = None; ep_start = np.ones((1,), dtype=bool)
    while not done:
        if model is not None:
            try:
                action, lstm_states = model.predict(obs, state=lstm_states,
                                                     episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
            except TypeError:
                action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        info['action'] = int(np.asarray(action).item())
        records.append(info)
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════
# SECTION 8: OPTUNA HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════

def build_model(env, params, use_recurrent):
    n_steps = min(params.get('n_steps', 256), env.n_steps - 2)
    SEED    = 42
    if use_recurrent:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO(
            "MlpLstmPolicy", env,
            learning_rate  = params['learning_rate'],
            seed           = SEED,
            n_steps        = n_steps,
            batch_size     = 64,
            n_epochs       = params.get('n_epochs', 4),
            gamma          = params.get('gamma', 0.99),
            gae_lambda     = params.get('gae_lambda', 0.95),
            clip_range     = params.get('clip_range', 0.2),
            ent_coef       = params.get('ent_coef', 0.05),
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            policy_kwargs  = {'lstm_hidden_size': params.get('lstm_hidden_size', 48),
                              'n_lstm_layers': 1},
            device='cpu', verbose=0)
    else:
        from stable_baselines3 import PPO
        hs = params.get('lstm_hidden_size', 48)
        return PPO(
            "MlpPolicy", env,
            learning_rate  = params['learning_rate'],
            seed           = SEED,
            n_steps        = n_steps,
            batch_size     = 64,
            n_epochs       = params.get('n_epochs', 4),
            gamma          = params.get('gamma', 0.99),
            gae_lambda     = params.get('gae_lambda', 0.95),
            clip_range     = params.get('clip_range', 0.2),
            ent_coef       = params.get('ent_coef', 0.05),  # consistent with RecurrentPPO
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            policy_kwargs  = {'net_arch': dict(pi=[hs, hs], vf=[hs, hs])},
            device='cpu', verbose=0)


# ent_coef=0.05: higher than ZAR (0.023) because JPY lower vol → greater entropy collapse risk
DEFAULT_PARAMS = {
    'learning_rate':    3e-4,
    'ent_coef':         0.05,
    'clip_range':       0.2,
    'gae_lambda':       0.95,
    'gamma':            0.99,
    'n_steps':          256,
    'n_epochs':         4,
    'reward_scaling':   10.0,
    'lstm_hidden_size': 56,
}


def tune_window(feat_df, train_dates, val_dates, feature_cols,
                use_recurrent, n_trials=20):
    import optuna

    norm_tune = FeatureNormalizer(252)
    norm_tune.fit(feat_df.loc[train_dates, feature_cols])

    vf       = feat_df.loc[val_dates]
    has_cont = 'port_continuity_pnl' in vf.columns
    has_full = 'port_full_pnl' in vf.columns
    val_cont = vf['port_continuity_pnl'].values if has_cont else None
    val_full = vf['port_full_pnl'].values        if has_full else None
    val_bm   = run_benchmark(vf['port_delta_16'].values, vf['port_price_close'].values,
                              vf['spot_16'].values, vf['spot_close'].values,
                              val_cont, val_full)
    bm_pnl   = val_bm['daily_pnl'].values  # use continuity-based for Optuna objective

    def objective(trial):
        params = {
            'learning_rate':    trial.suggest_float('learning_rate', 8e-5, 4e-4, log=True),
            'clip_range':       trial.suggest_float('clip_range', 0.15, 0.25),
            'gae_lambda':       trial.suggest_float('gae_lambda', 0.9, 0.99),
            'gamma':            trial.suggest_float('gamma', 0.95, 0.999),
            'n_steps':          trial.suggest_categorical('n_steps', [128, 256, 512]),
            'n_epochs':         trial.suggest_int('n_epochs', 3, 8),
            'reward_scaling':   trial.suggest_float('reward_scaling', 3, 50, log=True),
            'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [32, 48, 64]),
            'ent_coef':         DEFAULT_PARAMS['ent_coef'],  # fixed
        }
        train_env = make_env(feat_df, train_dates, norm_tune, feature_cols, params['reward_scaling'])
        val_env   = make_env(feat_df, val_dates,   norm_tune, feature_cols, params['reward_scaling'])
        model     = build_model(train_env, params, use_recurrent)

        total = 150_000; check_every = 30_000
        best_reduction = -999.0; no_improve = 0

        for block in range(0, total, check_every):
            model.learn(total_timesteps=check_every, reset_num_timesteps=False, progress_bar=False)
            val_rl    = run_episode(val_env, model)
            m         = evaluate(val_rl['daily_pnl'].values, bm_pnl)
            reduction = m['he_vol_reduction']
            trial.report(reduction, block // check_every)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if reduction > best_reduction:
                best_reduction = reduction; no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 3: break

        return best_reduction

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**DEFAULT_PARAMS, **study.best_params}
    best['best_val_reduction'] = study.best_value
    best['n_trials_completed'] = len(study.trials)
    return best


# ════════════════════════════════════════════════════════════════
# SECTION 9: ACTION ANALYSIS
# ════════════════════════════════════════════════════════════════

def analyse_actions(rl, feat_df, data_dir):
    rl = rl.copy()
    rl['date'] = pd.to_datetime(rl['date'])
    feat_test  = feat_df.loc[feat_df.index.isin(rl['date'])]

    rl['vol_regime']        = feat_test['vol_regime_63d'].values
    rl['vr_minus1']         = feat_test['vr_minus1_21d'].values
    rl['vol_ma_alignment']  = feat_test['vol_ma_alignment'].values
    rl['h00_zscore']        = feat_test['h00_intraday_zscore'].values
    rl['h06_zscore']        = feat_test['h06_intraday_zscore'].values
    rl['h00_to_close_lag1'] = feat_test['h00_to_close_lag1'].values
    rl['range_pos']         = feat_test['range_pos_21d'].values
    rl['action_name']       = rl['action'].map({k: v['name'] for k, v in ACTION_MAP.items()})
    rl['year']              = rl['date'].dt.year

    # 6-action map: 0=h100_00, 1=h75_00, 2=h100_06, 3=split_50_00_16, 4=split_50_06_16, 5=h100_16
    rl['is_00utc'] = rl['action'].isin([0, 1]).astype(int)
    rl['is_06utc'] = rl['action'].isin([2]).astype(int)
    rl['is_16utc'] = rl['action'].isin([5]).astype(int)
    rl['is_split'] = rl['action'].isin([3, 4]).astype(int)

    lines = []
    lines.append("="*70)
    lines.append("ACTION DISTRIBUTION ANALYSIS (JPY — 3 hedge windows)")
    lines.append("="*70)

    lines.append("\nOverall action distribution:")
    dist = (rl['action_name'].value_counts(normalize=True)*100).round(1)
    for k, v in dist.items():
        lines.append(f"  {k:<22} {v:.1f}%")

    lines.append("\nVenue weights and entropy by test year:")
    max_entropy = np.log(len(ACTION_MAP))
    lines.append(f"  {'year':<6} {'00utc%':>8} {'06utc%':>8} {'16utc%':>8} "
                 f"{'split%':>8} {'entropy':>10}  (max={max_entropy:.3f})")
    for year, grp in rl.groupby('year'):
        counts = grp['action'].value_counts(normalize=True)
        ent    = calc_entropy(counts)
        lines.append(
            f"  {year:<6}"
            f"{grp['is_00utc'].mean()*100:>8.1f}%"
            f"{grp['is_06utc'].mean()*100:>8.1f}%"
            f"{grp['is_16utc'].mean()*100:>8.1f}%"
            f"{grp['is_split'].mean()*100:>8.1f}%"
            f"{ent:>10.3f}"
        )

    # Conditioning diagnostics: pure 00 vs 06 vs 16 UTC actions
    h00 = rl['action'].isin([0, 1])
    h06 = rl['action'].isin([2])
    h16 = rl['action'].isin([5])
    n00, n06, n16 = h00.sum(), h06.sum(), h16.sum()
    lines.append(f"\nKey conditioning diagnostics  "
                 f"(n: 00UTC={n00}  06UTC={n06}  16UTC={n16})")
    lines.append(f"  {'metric':<25} {'00UTC':>10} {'06UTC':>10} {'16UTC':>10}")
    for col in ['vol_regime','vr_minus1','h00_zscore','h06_zscore',
                'h00_to_close_lag1','range_pos']:
        if col in rl.columns:
            m00 = rl.loc[h00, col].mean() if n00 > 0 else float('nan')
            m06 = rl.loc[h06, col].mean() if n06 > 0 else float('nan')
            m16 = rl.loc[h16, col].mean() if n16 > 0 else float('nan')
            lines.append(f"  {col:<25} {m00:>10.3f} {m06:>10.3f} {m16:>10.3f}")

    rl['alignment_bucket'] = pd.cut(
        rl['vol_ma_alignment'],
        bins=[-1.1,-0.3,0.3,1.1],
        labels=['bear_vol','flat_vol','bull_vol'])
    lines.append("\nAction % by vol MA alignment:")
    ct = pd.crosstab(rl['action_name'], rl['alignment_bucket'], normalize='columns') * 100
    lines.append(ct.round(1).to_string())

    for line in lines:
        logprint(line)

    rl[['date','year','action','action_name','is_00utc','is_06utc','is_16utc','is_split',
        'vol_regime','vr_minus1','h00_zscore','h06_zscore','h00_to_close_lag1']
       ].to_csv(f'{data_dir}/action_analysis.csv', index=False)
    logprint(f"\n  Saved: action_analysis.csv")


# ════════════════════════════════════════════════════════════════
# SECTION 10: MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════

def run_training(data_dir='.', total_timesteps=200_000, n_optuna_trials=20):

    logging.basicConfig(
        filename=os.path.join(data_dir, 'rl_training_jpy.log'),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logprint("="*70)
    logprint("STAGE 3 (JPY): RL HEDGE TIMING — 3 INTRADAY WINDOWS")
    logprint(f"  Hedge hours UTC: 00 (TC={TC_BPS[0]}bps), "
             f"06 (TC={TC_BPS[6]}bps), 16 (TC={TC_BPS[16]}bps)")
    logprint("="*70)

    import torch
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logprint("\n[1/5] Loading features...")
    feat_df = pd.read_parquet(f'{data_dir}/usdjpy_features.parquet')
    dates   = feat_df.index

    feature_cols = [c for c in STATE_FEATURES if c in feat_df.columns]
    missing      = [c for c in STATE_FEATURES if c not in feat_df.columns]
    if missing:
        logprint(f"  Warning: missing features: {missing}")
    logprint(f"  {len(feature_cols)} features, {len(dates)} days")

    required = ['port_delta_close','port_delta_00','port_delta_06','port_delta_16',
                'port_gamma_close','port_price_close',
                'spot_00','spot_06','spot_16','spot_close']
    for r in required:
        if r not in feat_df.columns:
            raise ValueError(f"Missing column '{r}' — run sabr_calibration_jpy.py first")

    has_cont = 'port_continuity_pnl' in feat_df.columns
    has_full = 'port_full_pnl' in feat_df.columns
    logprint(f"  Continuity P&L: {'YES' if has_cont else 'NO'}")
    logprint(f"  Full P&L:       {'YES' if has_full else 'NO'}")
    logprint(f"  Portfolio: avg |delta|={feat_df['port_delta_close'].abs().mean():.4f}  "
             f"avg legs={feat_df['port_n_legs'].mean():.0f}")

    logprint("\n[2/5] Walk-forward splits...")
    splits = create_splits(dates)
    for s in splits:
        logprint(f"  W{s['w']}: Train {s['tr_yrs']} → Val {s['vy']} → Test {s['ty']}")

    logprint("\n[3/5] RL setup...")
    use_recurrent = False
    try:
        from sb3_contrib import RecurrentPPO
        use_recurrent = True
        logprint("  RecurrentPPO (LSTM)")
    except ImportError:
        logprint("  Standard PPO (MLP)")
    from stable_baselines3 import PPO

    has_optuna = False
    try:
        import optuna
        has_optuna = True
        logprint(f"  Optuna: YES ({n_optuna_trials} trials per window)")
    except ImportError:
        logprint("  Optuna: NOT INSTALLED — using default hyperparameters")

    logprint(f"\n[4/5] Training ({total_timesteps} steps per window)...")
    all_test, all_bench, all_metrics = [], [], []

    for split in splits:
        w            = split['w']
        model_path   = f'{data_dir}/model_window_{w}'
        optuna_path  = f'{data_dir}/optuna_best_w{w}.json'
        metrics_path = f'{data_dir}/rl_metrics_w{w}.json'

        # ── CHECKPOINT: skip if fully done ──
        if os.path.exists(model_path + '.zip') and os.path.exists(metrics_path):
            logprint(f"\n  -- Window {w}: Test {split['ty']} -- SKIPPING (checkpoint exists)")
            with open(metrics_path) as mf: m = json.load(mf)
            all_metrics.append(m)

            norm = FeatureNormalizer(252)
            norm.fit(feat_df.loc[split['tr'], feature_cols])
            best_params = DEFAULT_PARAMS.copy()
            if os.path.exists(optuna_path):
                with open(optuna_path) as pf: best_params = json.load(pf)
                src = "optuna"
            else:
                src = "default"
            logprint(f"    Params ({src}): lr={best_params['learning_rate']:.6f}  "
                     f"ent_coef={best_params['ent_coef']:.4f}  "
                     f"lstm_hidden={best_params['lstm_hidden_size']}  "
                     f"reward_scaling={best_params.get('reward_scaling', 1000.0):.1f}")

            test_env = make_env(feat_df, split['ts'], norm, feature_cols,
                                best_params.get('reward_scaling', 1000.0))
            if use_recurrent:
                loaded = RecurrentPPO.load(model_path, env=test_env, device='cpu')
            else:
                loaded = PPO.load(model_path, env=test_env, device='cpu')
            test_rl = run_episode(test_env, loaded)
            tf      = feat_df.loc[split['ts']]
            test_cont = tf['port_continuity_pnl'].values if has_cont else None
            test_full = tf['port_full_pnl'].values        if has_full else None
            test_bm = run_benchmark(tf['port_delta_16'].values, tf['port_price_close'].values,
                                     tf['spot_16'].values, tf['spot_close'].values,
                                     test_cont, test_full)
            test_rl['date'] = split['ts'][:len(test_rl)]
            test_bm['date'] = split['ts'][:len(test_bm)]
            all_test.append(test_rl); all_bench.append(test_bm)
            logprint(f"    Loaded: he_vol_reduction={m['he_vol_reduction']:.1%}")
            continue

        logprint(f"\n  -- Window {w}: Test {split['ty']} --")

        # ── Tune ──
        if os.path.exists(optuna_path):
            logprint(f"    Loading saved hyperparameters...")
            with open(optuna_path) as pf: best_params = json.load(pf)
            logprint(f"    Best val reduction: {best_params.get('best_val_reduction','N/A')}")
        elif has_optuna and n_optuna_trials > 0:
            logprint(f"    Tuning ({n_optuna_trials} trials)...")
            t_tune = time.time()
            best_params = tune_window(feat_df, split['tr'], split['va'], feature_cols,
                                       use_recurrent, n_optuna_trials)
            logprint(f"    Done ({time.time()-t_tune:.0f}s, "
                     f"{best_params['n_trials_completed']} trials)")
            logprint(f"    Best val reduction: {best_params['best_val_reduction']:.1%}")
            logprint(f"    lr={best_params['learning_rate']:.5f}  "
                     f"lstm={best_params['lstm_hidden_size']}  "
                     f"reward_scale={best_params['reward_scaling']:.0f}")
            with open(optuna_path, 'w') as pf: json.dump(best_params, pf, indent=2)
        else:
            logprint("    Using default hyperparameters")
            best_params = DEFAULT_PARAMS.copy()

        logprint(f"    Hyperparameters for W{w} (Test {split['ty']}):")
        logprint(f"      lr={best_params['learning_rate']:.6f}  "
                 f"ent_coef={best_params['ent_coef']:.4f}  "
                 f"clip_range={best_params['clip_range']:.3f}")
        logprint(f"      gae_lambda={best_params['gae_lambda']:.3f}  "
                 f"gamma={best_params['gamma']:.4f}  "
                 f"n_steps={best_params['n_steps']}  "
                 f"n_epochs={best_params['n_epochs']}")
        logprint(f"      lstm_hidden={best_params['lstm_hidden_size']}  "
                 f"reward_scaling={best_params['reward_scaling']:.1f}")

        # ── Train ──
        norm = FeatureNormalizer(252)
        norm.fit(feat_df.loc[split['tr'], feature_cols])
        rs = best_params.get('reward_scaling', 10.0)

        train_env = make_env(feat_df, split['tr'], norm, feature_cols, rs)
        val_env   = make_env(feat_df, split['va'], norm, feature_cols, rs)
        test_env  = make_env(feat_df, split['ts'], norm, feature_cols, rs)

        t0    = time.time()
        model = build_model(train_env, best_params, use_recurrent)
        logprint(f"    Training ({total_timesteps} steps)...")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        elapsed = time.time() - t0
        logprint(f"    Done ({elapsed:.0f}s)")
        model.save(model_path)

        # ── Validation ──
        val_rl   = run_episode(val_env, model)
        vf       = feat_df.loc[split['va']]
        val_cont = vf['port_continuity_pnl'].values if has_cont else None
        val_full = vf['port_full_pnl'].values        if has_full else None
        val_bm   = run_benchmark(vf['port_delta_16'].values, vf['port_price_close'].values,
                                  vf['spot_16'].values, vf['spot_close'].values,
                                  val_cont, val_full)
        vm = evaluate(val_rl['daily_pnl'].values, val_bm['daily_pnl'].values)
        logprint(f"    Val: HE_vol RL={vm['rl_he_vol']:.6f} BM={vm['bm_he_vol']:.6f} "
                 f"reduction={vm['he_vol_reduction']:.1%}")

        # ── Test ──
        test_rl   = run_episode(test_env, model)
        tf        = feat_df.loc[split['ts']]
        test_cont = tf['port_continuity_pnl'].values if has_cont else None
        test_full = tf['port_full_pnl'].values        if has_full else None
        test_bm   = run_benchmark(tf['port_delta_16'].values, tf['port_price_close'].values,
                                   tf['spot_16'].values, tf['spot_close'].values,
                                   test_cont, test_full)
        m = evaluate(test_rl['daily_pnl'].values, test_bm['daily_pnl'].values)
        m['window'] = w; m['test_year'] = split['ty']; m['train_time'] = elapsed

        # Full P&L totals for reporting (includes inception premium)
        rl_full_total = test_rl['full_daily_pnl'].sum() if 'full_daily_pnl' in test_rl.columns else float('nan')
        bm_full_total = test_bm['full_daily_pnl'].sum() if 'full_daily_pnl' in test_bm.columns else float('nan')

        logprint(f"    Test {split['ty']}:")
        logprint(f"      RL  HE_vol={m['rl_he_vol']:.6f}  MaxLoss={m['rl_max_loss']:.6f}  "
                 f"PnL(hedge)={m['rl_total_pnl']:.4f}  PnL(full)={rl_full_total:.4f}")
        logprint(f"      BM  HE_vol={m['bm_he_vol']:.6f}  MaxLoss={m['bm_max_loss']:.6f}  "
                 f"PnL(hedge)={m['bm_total_pnl']:.4f}  PnL(full)={bm_full_total:.4f}")
        logprint(f"      HE_vol_reduction={m['he_vol_reduction']:.1%}  "
                 f"F={m['F_stat']:.3f}  p={m['var_p_value']:.4f}")

        acts     = test_rl['action'].value_counts().sort_index()
        act_str  = '  '.join(f"{ACTION_MAP[a]['name']}={c/len(test_rl)*100:.0f}%"
                             for a, c in acts.items())
        logprint(f"    Actions: {act_str}")
        logprint(f"    Policy entropy: {calc_entropy(test_rl['action'].value_counts(normalize=True)):.3f} "
                 f"(max={np.log(len(ACTION_MAP)):.3f})")

        test_rl['date'] = split['ts'][:len(test_rl)]
        test_bm['date'] = split['ts'][:len(test_bm)]
        all_test.append(test_rl); all_bench.append(test_bm); all_metrics.append(m)

        with open(metrics_path, 'w') as mf: json.dump(m, mf, indent=2)
        logprint(f"    Checkpoint saved")

    # ── Aggregate ─────────────────────────────────────────────────────
    logprint(f"\n{'='*70}")
    logprint("[5/5] AGGREGATE RESULTS")
    logprint(f"{'='*70}")

    all_rl  = pd.concat([r['daily_pnl']      for r in all_test])
    all_bm  = pd.concat([r['daily_pnl']      for r in all_bench])
    overall = evaluate(all_rl.values, all_bm.values)

    logprint(f"\n  Out-of-sample ({overall['n_days']} days):")
    logprint(f"    RL:  HE_vol={overall['rl_he_vol']:.6f}  "
             f"Mean|PnL|={overall['rl_mean_abs']:.6f}  MaxLoss={overall['rl_max_loss']:.6f}")
    logprint(f"    BM:  HE_vol={overall['bm_he_vol']:.6f}  "
             f"Mean|PnL|={overall['bm_mean_abs']:.6f}  MaxLoss={overall['bm_max_loss']:.6f}")
    logprint(f"    HE_vol_reduction={overall['he_vol_reduction']:.1%}  "
             f"F={overall['F_stat']:.3f}  p={overall['var_p_value']:.4f}")

    # Full P&L aggregate (economic, including inception premium)
    if 'full_daily_pnl' in pd.concat(all_test).columns:
        rl_full_agg = pd.concat([r['full_daily_pnl'] for r in all_test]).sum()
        bm_full_agg = pd.concat([r['full_daily_pnl'] for r in all_bench]).sum()
        logprint(f"\n  Full economic P&L in USD (hedge + inception premium):")
        logprint(f"    RL cumulative: {rl_full_agg:.4f} USD per USD notional")
        logprint(f"    BM cumulative: {bm_full_agg:.4f} USD per USD notional")

    mdf = pd.DataFrame(all_metrics)
    logprint(f"\n  Per-window:")
    logprint(mdf[['test_year','rl_he_vol','bm_he_vol','he_vol_reduction',
                  'F_stat','var_p_value']].to_string(index=False))

    logprint(f"\n  Tuned hyperparameters per window:")
    for split in splits:
        op = f'{data_dir}/optuna_best_w{split["w"]}.json'
        if os.path.exists(op):
            with open(op) as pf: p = json.load(pf)
            logprint(f"    W{split['w']} ({split['ty']}): lr={p['learning_rate']:.5f}  "
                     f"lstm={p['lstm_hidden_size']}  "
                     f"reward_scale={p['reward_scaling']:.0f}  "
                     f"val_red={p.get('best_val_reduction','N/A')}")

    mdf.to_csv(f'{data_dir}/rl_metrics.csv', index=False)
    pd.concat(all_test).to_parquet(f'{data_dir}/rl_test_results.parquet', index=False)
    pd.concat(all_bench).to_parquet(f'{data_dir}/benchmark_results.parquet', index=False)
    with open(f'{data_dir}/rl_overall_metrics.json', 'w') as f:
        json.dump(overall, f, indent=2)
    logprint(f"\n  Saved: rl_metrics.csv, rl_test_results.parquet, "
             f"benchmark_results.parquet, rl_overall_metrics.json")

    try:
        all_rl_df = pd.concat(all_test)
        analyse_actions(all_rl_df, feat_df, data_dir)
    except Exception as e:
        logprint(f"  Action analysis skipped: {e}")

    try:
        plot_results(data_dir)
    except Exception as e:
        logprint(f"  Plotting skipped: {e}")

    logprint(f"\n{'='*70}")
    logprint("STAGE 3 (JPY) COMPLETE")
    logprint(f"{'='*70}")
    return {'metrics': mdf, 'overall': overall}


# ════════════════════════════════════════════════════════════════
# SECTION 11: PLOTTING
# ════════════════════════════════════════════════════════════════

def plot_results(data_dir='.'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    rl = pd.read_parquet(f'{data_dir}/rl_test_results.parquet')
    bm = pd.read_parquet(f'{data_dir}/benchmark_results.parquet')
    rl_dates = pd.to_datetime(rl['date'])
    bm_dates = pd.to_datetime(bm['date'])

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    # Hedge error P&L (continuity-based)
    ax = axes[0]
    ax.plot(rl_dates, rl['daily_pnl'].cumsum(), label='RL Agent', color='blue', lw=1)
    ax.plot(bm_dates, bm['daily_pnl'].cumsum(), label='Benchmark (16UTC)', color='red', lw=1)
    ax.set_ylabel('Cum P&L (hedge error)')
    ax.set_title('USD/JPY Out-of-Sample: RL Agent vs Benchmark')
    ax.legend(); ax.axhline(0, color='gray', ls='--', alpha=0.5); ax.grid(alpha=0.3)

    # Full economic P&L (includes inception premium)
    ax = axes[1]
    if 'full_daily_pnl' in rl.columns and 'full_daily_pnl' in bm.columns:
        ax.plot(rl_dates, rl['full_daily_pnl'].cumsum(), label='RL (full)', color='blue', lw=1)
        ax.plot(bm_dates, bm['full_daily_pnl'].cumsum(), label='BM (full)', color='red', lw=1)
        ax.set_ylabel('Cum P&L (full, incl. premium)')
        ax.legend(); ax.axhline(0, color='gray', ls='--', alpha=0.5); ax.grid(alpha=0.3)
    else:
        axes[1].set_visible(False)

    # Rolling HE vol
    ax = axes[2]
    w = 63
    ax.plot(rl_dates, rl['daily_pnl'].rolling(w).std()*np.sqrt(252), label='RL', color='blue', lw=1)
    ax.plot(bm_dates, bm['daily_pnl'].rolling(w).std()*np.sqrt(252), label='Benchmark', color='red', lw=1)
    ax.set_ylabel('Rolling 3M HE Vol (ann.)')
    ax.legend(); ax.grid(alpha=0.3)

    # Action distribution
    ax = axes[3]
    if 'action' in rl.columns:
        for a_id in sorted(ACTION_MAP.keys()):
            mask = (rl['action'] == a_id).astype(float)
            rpct = mask.rolling(63, min_periods=10).mean() * 100
            if rpct.max() > 1:
                ax.plot(rl_dates, rpct.values, label=ACTION_MAP[a_id]['name'], lw=1)
        ax.set_ylabel('Action %'); ax.set_title('Rolling 3M Action Distribution')
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'{data_dir}/rl_performance.png', dpi=150, bbox_inches='tight')
    logprint("  Saved: rl_performance.png")
    plt.close()


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    d  = sys.argv[1] if len(sys.argv) > 1 else '.'
    ts = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    run_training(d, ts, nt)
