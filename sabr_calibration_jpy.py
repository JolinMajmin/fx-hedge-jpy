"""
sabr_calibration_jpy.py — Stage 2 of USD/JPY Hedge Timing Pipeline
====================================================================
Reads:
    usdjpy_pipeline_ready.parquet   (from usdjpy_pipeline_creator.py)
    usdjpy_hourly_ohlc.csv          (from FX_daily_hourly_create.py)

Produces:
    usdjpy_sabr_params.parquet      (daily SABR params per tenor, smoothed)
    usdjpy_features.parquet         (complete feature matrix for RL)

P&L columns produced:
    port_continuity_pnl  — daily MTM change of CONTINUING legs only (no inception/expiry).
                           Used by RL reward: pure gamma P&L the agent can minimise.
    port_inception_pnl   — premium paid (long) or received (short) when a new cohort starts.
    port_expiry_pnl      — settlement minus last MTM when a cohort expires (usually ~0).
    port_full_pnl        — complete economic daily P&L = continuity + inception + expiry.
                           Used for strategy P&L reporting.

Tokyo UTC equivalents (JST = UTC+9, Japan has NO daylight saving time):
    9am Tokyo  = 00:00 UTC
    3pm Tokyo  = 06:00 UTC

Usage:  python -u sabr_calibration_jpy.py C:\\BackbookData_jpy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import norm
import time, sys, warnings
from numba import jit

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

TENOR_YEARS = {'3M': 0.25, '2M': 2/12, '1M': 1/12, '3W': 3/52,
               '2W': 2/52, '1W': 1/52, 'ON': 1/365}
TENOR_ORDER = ['3M', '2M', '1M', '3W', '2W', '1W', 'ON']
CALIBRATION_TENORS = ['3M', '2M', '1M', '1W']
CAL_TENOR_YEARS = sorted([(TENOR_YEARS[t], t) for t in CALIBRATION_TENORS], reverse=True)

HEDGE_HOURS = [0, 6, 16]   # UTC: Tokyo open, Tokyo afternoon, London/NY

BASE_NOTIONAL   = 1_000_000
VOL_SCALING     = 0.5
EXPIRY_BDAYS    = 63
ROLL_BDAYS      = 21
SPOT_VOL_WINDOW = 130
SPOT_VOL_DECAY  = 0.98


# ════════════════════════════════════════════════════════════════
# PART A: SABR MODEL (beta=1)
# ════════════════════════════════════════════════════════════════

def sabr_vol_beta1(K, F, T, alpha, rho, nu):
    if T <= 1e-10 or alpha <= 0:
        return alpha if alpha > 0 else 0.15
    if abs(F - K) < 1e-10:
        return alpha * (1 + ((2-3*rho**2)*nu**2/24 + rho*nu*alpha/4 + alpha**2/24) * T)
    logFK = np.log(F / K)
    FK = F * K
    zeta = (nu / alpha) * logFK
    sqrt_term = np.sqrt(1 - 2*rho*zeta + zeta**2)
    x_zeta = np.log((sqrt_term + zeta - rho) / (1 - rho))
    ratio = zeta / x_zeta if abs(x_zeta) > 1e-10 else 1.0
    correction = 1 + (alpha**2/(24*FK) + rho*nu*alpha/(4*np.sqrt(FK)) + (2-3*rho**2)*nu**2/24) * T
    return alpha * ratio * correction

def alpha_from_atm(sigma_atm, F, T, rho, nu):
    if sigma_atm <= 0 or T <= 1e-10:
        return max(sigma_atm, 0.01)
    def obj(a):
        return (sabr_vol_beta1(F, F, T, a, rho, nu) - sigma_atm)**2 if a > 0 else 1e6
    return minimize_scalar(obj, bounds=(0.001, 5.0), method='bounded').x

def calibrate_sabr(S, F, T, sigma_atm, vol_25c, vol_25p, vol_10c, vol_10p):
    K_25c = strike_from_pa_delta(0.25, S, F, T, vol_25c, True)
    K_25p = strike_from_pa_delta(0.25, S, F, T, vol_25p, False)
    K_10c = strike_from_pa_delta(0.10, S, F, T, vol_10c, True)
    K_10p = strike_from_pa_delta(0.10, S, F, T, vol_10p, False)
    strikes  = [K_10p, K_25p, K_25c, K_10c]
    mkt_vols = [vol_10p, vol_25p, vol_25c, vol_10c]

    def objective(params):
        rho = np.clip(params[0], -0.999, 0.999)
        nu  = np.clip(params[1], 0.001, 5.0)
        try:
            a = alpha_from_atm(sigma_atm, F, T, rho, nu)
            if a <= 0 or a > 5: return 1e6
            return sum((sabr_vol_beta1(K, F, T, a, rho, nu) - mv)**2
                       for K, mv in zip(strikes, mkt_vols))
        except: return 1e6

    best_obj, best_res = 1e6, None
    for rho0, nu0 in [(0.3,0.5),(0.5,0.3),(0.1,0.8),(0.6,1.0),(-0.1,0.5)]:
        try:
            res = minimize(objective, [rho0,nu0], bounds=[(-0.999,0.999),(0.001,5.0)],
                           method='L-BFGS-B', options={'maxiter':200,'ftol':1e-12})
            if res.fun < best_obj: best_obj, best_res = res.fun, res
        except: continue

    if best_res is None:
        return sigma_atm, 0.3, 0.5, 0.0, 99.0, False
    rho_f   = np.clip(best_res.x[0], -0.999, 0.999)
    nu_f    = np.clip(best_res.x[1], 0.001, 5.0)
    alpha_f = alpha_from_atm(sigma_atm, F, T, rho_f, nu_f)
    atm_err = sabr_vol_beta1(F, F, T, alpha_f, rho_f, nu_f) - sigma_atm
    rmse    = np.sqrt(best_obj / 4)
    return alpha_f, rho_f, nu_f, atm_err, rmse, True


def smooth_sabr_params(sabr_df, ew_alpha=0.8, spike_threshold=2.0):
    for tenor in sabr_df['tenor'].unique():
        mask = sabr_df['tenor'] == tenor
        for param in ['rho', 'nu']:
            s = sabr_df.loc[mask, param].copy()
            s = s.interpolate(method='linear').ffill().bfill()
            roll_mean = s.rolling(21, min_periods=5, center=True).mean()
            roll_std  = s.rolling(21, min_periods=5, center=True).std().replace(0, 0.01)
            is_spike  = (s - roll_mean).abs() > spike_threshold * roll_std
            s[is_spike] = np.nan
            s = s.interpolate(method='linear').ffill().bfill()
            sabr_df.loc[mask, param] = s.ewm(alpha=1-ew_alpha, min_periods=1).mean().values
    return sabr_df


def build_sabr_lookup(sabr_df, dates):
    lookup = {}
    for tenor in CALIBRATION_TENORS:
        subset = sabr_df[sabr_df['tenor']==tenor].copy()
        subset['date'] = pd.to_datetime(subset['date'])
        subset = subset.set_index('date').sort_index().reindex(dates)
        for col in ['alpha','rho','nu']:
            subset[col] = subset[col].interpolate().ffill().bfill()
        lookup[tenor] = {
            'alpha': subset['alpha'].values,
            'rho':   subset['rho'].values,
            'nu':    subset['nu'].values,
        }
    return lookup


def get_sabr_params(T_rem, day_idx, sabr_lookup):
    if T_rem >= CAL_TENOR_YEARS[0][0]:
        t = CAL_TENOR_YEARS[0][1]
        return sabr_lookup[t]['alpha'][day_idx], sabr_lookup[t]['rho'][day_idx], sabr_lookup[t]['nu'][day_idx]
    if T_rem <= CAL_TENOR_YEARS[-1][0]:
        t = CAL_TENOR_YEARS[-1][1]
        return sabr_lookup[t]['alpha'][day_idx], sabr_lookup[t]['rho'][day_idx], sabr_lookup[t]['nu'][day_idx]
    for i in range(len(CAL_TENOR_YEARS)-1):
        T_long, t_long   = CAL_TENOR_YEARS[i]
        T_short, t_short = CAL_TENOR_YEARS[i+1]
        if T_short <= T_rem <= T_long:
            w = (T_rem - T_short) / (T_long - T_short)
            a = sabr_lookup[t_short]['alpha'][day_idx]*(1-w) + sabr_lookup[t_long]['alpha'][day_idx]*w
            r = sabr_lookup[t_short]['rho'][day_idx]*(1-w)   + sabr_lookup[t_long]['rho'][day_idx]*w
            v = sabr_lookup[t_short]['nu'][day_idx]*(1-w)    + sabr_lookup[t_long]['nu'][day_idx]*w
            return a, r, v
    t = CAL_TENOR_YEARS[-1][1]
    return sabr_lookup[t]['alpha'][day_idx], sabr_lookup[t]['rho'][day_idx], sabr_lookup[t]['nu'][day_idx]


# ════════════════════════════════════════════════════════════════
# PART B: FX DELTA CONVENTIONS & GK PRICING
# ════════════════════════════════════════════════════════════════

def gk_d1_d2(F, K, T, sigma):
    if T <= 1e-10 or sigma <= 1e-10: return 0.0, 0.0
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    return d1, d1 - sigma*np.sqrt(T)

def pa_spot_delta(S, K, F, T, sigma, is_call):
    phi = 1.0 if is_call else -1.0
    _, d2 = gk_d1_d2(F, K, T, sigma)
    return (K/F) * phi * norm.cdf(phi * d2)

def strike_from_pa_delta(delta_tgt, S, F, T, sigma, is_call):
    def obj(K): return pa_spot_delta(S, K, F, T, sigma, is_call) - (delta_tgt if is_call else -delta_tgt)
    try:    return brentq(obj, F*0.3, F*3.0, xtol=1e-8)
    except: return brentq(obj, F*0.1, F*5.0, xtol=1e-8)

def gk_price(S, K, F, T, sigma, is_call, r_f=0.05):
    if T <= 1e-10:
        phi = 1 if is_call else -1
        return max(phi*(S-K), 0)
    r_d = r_f + np.log(F/S)/T
    d1, d2 = gk_d1_d2(F, K, T, sigma)
    phi = 1 if is_call else -1
    return phi*(S*np.exp(-r_f*T)*norm.cdf(phi*d1) - K*np.exp(-r_d*T)*norm.cdf(phi*d2))

def gk_delta(S, K, F, T, sigma, is_call, r_f=0.05):
    if T <= 1e-10:
        phi = 1 if is_call else -1
        return phi*1.0 if phi*(S-K) > 0 else 0.0
    d1, _ = gk_d1_d2(F, K, T, sigma)
    phi = 1 if is_call else -1
    return phi * np.exp(-r_f*T) * norm.cdf(phi*d1)

def gk_gamma(S, K, F, T, sigma, r_f=0.05):
    if T <= 1e-10: return 0.0
    d1, _ = gk_d1_d2(F, K, T, sigma)
    return np.exp(-r_f*T) * norm.pdf(d1) / (S*sigma*np.sqrt(T))

def gk_vega(S, K, F, T, sigma, r_f=0.05):
    if T <= 1e-10: return 0.0
    d1, _ = gk_d1_d2(F, K, T, sigma)
    return S * np.exp(-r_f*T) * np.sqrt(T) * norm.pdf(d1)


# ════════════════════════════════════════════════════════════════
# PART C: VOL SURFACE INTERPOLATION  (inception only)
# ════════════════════════════════════════════════════════════════

def interpolate_vol(T_target, vols_by_tenor):
    tenors = sorted([(TENOR_YEARS[t], v) for t, v in vols_by_tenor.items()
                     if not np.isnan(v)], reverse=True)
    if not tenors: return 0.10
    if T_target >= tenors[0][0]: return tenors[0][1]
    if T_target <= tenors[-1][0]: return tenors[-1][1]
    for i in range(len(tenors)-1):
        Tl, vl = tenors[i]; Ts, vs = tenors[i+1]
        if Ts <= T_target <= Tl:
            varl, vars_ = vl**2*Tl, vs**2*Ts
            vt = vars_ + (varl-vars_)/(Tl-Ts)*(T_target-Ts)
            return np.sqrt(max(vt,0)/T_target) if vt > 0 else vs
    return tenors[-1][1]

def get_vol(opt_type, T, row):
    prefix = {'ATM':'ATM', '25C':'VOL_25C', '25P':'VOL_25P'}[opt_type]
    vols = {t: row[f"{prefix}_{t}"]/100.0 for t in TENOR_ORDER if f"{prefix}_{t}" in row.index}
    return interpolate_vol(T, vols)

def get_forward(T, S, row):
    fwds = {t: row[f'FWD_{t}'] for t in TENOR_ORDER if f'FWD_{t}' in row.index}
    tenors = sorted([(TENOR_YEARS[t], fwds[t]) for t in TENOR_ORDER if t in fwds], reverse=True)
    if not tenors: return S
    if T >= tenors[0][0]: return S + tenors[0][1]
    if T <= tenors[-1][0]: return S + tenors[-1][1]
    for i in range(len(tenors)-1):
        Tl, fl = tenors[i]; Ts, fs = tenors[i+1]
        if Ts <= T <= Tl:
            return S + fs + (fl-fs)/(Tl-Ts)*(T-Ts)
    return S + tenors[-1][1]


# ════════════════════════════════════════════════════════════════
# PART D: SPOT-VOL BETA
# ════════════════════════════════════════════════════════════════

def estimate_spot_vol_betas(spot, atm_vol):
    spot_ret = np.log(spot/spot.shift(1))
    vol_chg  = atm_vol.diff()
    r_up, r_down = spot_ret.clip(lower=0), spot_ret.clip(upper=0)
    bu  = pd.Series(np.nan, index=spot.index)
    bd  = pd.Series(np.nan, index=spot.index)
    rho = pd.Series(np.nan, index=spot.index)
    for i in range(SPOT_VOL_WINDOW, len(spot)):
        s, e = i-SPOT_VOL_WINDOW, i
        y  = vol_chg.iloc[s:e].values
        xu, xd, xs = r_up.iloc[s:e].values, r_down.iloc[s:e].values, spot_ret.iloc[s:e].values
        w  = np.array([SPOT_VOL_DECAY**(e-1-j) for j in range(s, e)])
        mask = ~(np.isnan(y)|np.isnan(xu)|np.isnan(xd))
        if mask.sum() < 20: continue
        yw = y[mask]*np.sqrt(w[mask])
        X  = np.column_stack([np.ones(mask.sum()), xu[mask], xd[mask]])
        Xw = X * np.sqrt(w[mask])[:,None]
        try:
            b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            bu.iloc[i], bd.iloc[i] = b[1], b[2]
        except: pass
        try: rho.iloc[i] = np.corrcoef(xs[mask], y[mask])[0,1]
        except: pass
    return bu.ffill().bfill(), bd.ffill().bfill(), rho.ffill().bfill()


# ════════════════════════════════════════════════════════════════
# PART E: AGING PORTFOLIO
# ════════════════════════════════════════════════════════════════

class Leg:
    __slots__ = ['K','is_call','notional','inc','exp','opt_type','cohort']
    def __init__(self, K, is_call, notional, inc, exp, opt_type, cohort):
        self.K, self.is_call, self.notional = K, is_call, notional
        self.inc, self.exp, self.opt_type, self.cohort = inc, exp, opt_type, cohort
    def alive(self, d): return self.inc <= d < self.exp
    def T_rem(self, d): return max((self.exp-d)/365.0, 1/365)

def build_cohorts(pipe_df, dates):
    n = len(dates)
    median_atm = pipe_df.loc[dates, 'ATM_1M'].median()
    legs, cid = [], 0
    for inc in range(0, n, ROLL_BDAYS):
        exp = min(inc+EXPIRY_BDAYS, n)
        if exp-inc < 5: continue
        row = pipe_df.iloc[inc]
        S   = row['SPOT']
        notional = BASE_NOTIONAL * (1 + VOL_SCALING*(row['ATM_1M']/median_atm - 1))
        T   = EXPIRY_BDAYS/365.0
        F   = get_forward(T, S, row)
        sa, v25c, v25p = row['ATM_3M']/100, row['VOL_25C_3M']/100, row['VOL_25P_3M']/100
        K_atm = F * np.exp(0.5*sa**2*T)
        K_25c = strike_from_pa_delta(0.25, S, F, T, v25c, True)
        K_25p = strike_from_pa_delta(0.25, S, F, T, v25p, False)
        legs += [
            Leg(K_atm, True,  +notional,   inc, exp, 'ATM', cid),
            Leg(K_atm, False, +notional,   inc, exp, 'ATM', cid),
            Leg(K_25c, True,  -2*notional, inc, exp, '25C', cid),
            Leg(K_25p, False, +notional,   inc, exp, '25P', cid),
        ]
        cid += 1
    return legs, cid


# ════════════════════════════════════════════════════════════════
# PART F: HOURLY FEATURES
# ════════════════════════════════════════════════════════════════

def yang_zhang_vol(ohlc, window=21, ann=252):
    lho  = np.log(ohlc['high']/ohlc['open'])
    llo  = np.log(ohlc['low']/ohlc['open'])
    lco  = np.log(ohlc['close']/ohlc['open'])
    loc_ = np.log(ohlc['open']/ohlc['close'].shift(1))
    rs   = lho*(lho-lco) + llo*(llo-lco)
    k    = 0.34/(1.34+(window+1)/(window-1))
    return np.sqrt(ann*((loc_**2).rolling(window).mean() +
                        k*(lco**2).rolling(window).mean() +
                        (1-k)*rs.rolling(window).mean()))

def compute_hourly_features(hourly_df):
    df = hourly_df.copy()
    dt = pd.to_datetime(df['hour_utc'])
    df = df.set_index(dt).sort_index()
    f  = pd.DataFrame(index=df.index)
    f['mid']     = df['close']
    f['ret_1h']  = np.log(df['close']/df['close'].shift(1))
    f['ret_4h']  = np.log(df['close']/df['close'].shift(4))
    f['ret_24h'] = np.log(df['close']/df['close'].shift(24))
    daily = df.resample('B').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    for w in [5, 10, 21, 63]:
        rv = yang_zhang_vol(daily, w); rv.name = f'rv_yz_{w}d'
        f = f.join(rv, how='left'); f[f'rv_yz_{w}d'] = f[f'rv_yz_{w}d'].ffill()
    ds = f['ret_1h'].resample('B').std().rolling(20).mean()
    ds.name = 'daily_ret_std'
    f = f.join(ds, how='left'); f['daily_ret_std'] = f['daily_ret_std'].ffill()
    f['date'] = f.index.date
    f['intraday_cumret'] = f.groupby('date')['ret_1h'].cumsum()
    f['intraday_zscore'] = f['intraday_cumret'] / f['daily_ret_std'].replace(0, np.nan)
    f['hourly_range']    = (df['high']-df['low'])/df['close']
    f['spread']          = df['avg_spread']
    return f

@jit(nopython=True)
def calculate_slope_numba_standardized(arr, window):
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    x = np.arange(window, dtype=np.float64)
    x_mean = np.mean(x)
    x = (x - x_mean) / np.sqrt(np.sum((x - x_mean)**2))
    for i in range(window-1, n):
        y = arr[i-window+1:i+1].astype(np.float64)
        y_mean = np.mean(y)
        result[i] = np.sum(x * (y - y_mean))
    return result


# ════════════════════════════════════════════════════════════════
# PART G: MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pipeline(data_dir='.'):
    print("="*70, flush=True)
    print("STAGE 2 (JPY): SABR CALIBRATION + AGING PORTFOLIO + FEATURES", flush=True)
    print(f"  Hedge hours (UTC): {HEDGE_HOURS}  "
          f"[{HEDGE_HOURS[0]:02d}=Tokyo open, {HEDGE_HOURS[1]:02d}=Tokyo PM, {HEDGE_HOURS[2]:02d}=London/NY]",
          flush=True)
    print("="*70, flush=True)

    # ── [1/7] Load ──────────────────────────────────────────────────────
    print("\n[1/7] Loading data...", flush=True)
    pipe_df = pd.read_parquet(f'{data_dir}/usdjpy_pipeline_ready.parquet')
    hourly  = pd.read_csv(f'{data_dir}/usdjpy_hourly_ohlc.csv')
    dates   = pipe_df.index
    n       = len(dates)
    print(f"  {n} days: {dates[0].date()} to {dates[-1].date()}", flush=True)

    # ── [2/7] SABR calibration ──────────────────────────────────────────
    print("\n[2/7] SABR calibration...", flush=True)
    sabr_records = []
    t0 = time.time()
    for i, (dt, row) in enumerate(pipe_df.iterrows()):
        if (i+1)%500 == 0:
            el = time.time()-t0; rem = (n-i-1)/(i+1)*el
            print(f"    {i+1}/{n} ({el:.0f}s, ~{rem:.0f}s left)", flush=True)
        S = row['SPOT']
        for tenor in CALIBRATION_TENORS:
            T    = TENOR_YEARS[tenor]
            F    = row.get(f'F_{tenor}', S)
            sa   = row[f'ATM_{tenor}']/100
            v25c = row[f'VOL_25C_{tenor}']/100
            v25p = row[f'VOL_25P_{tenor}']/100
            v10c = row[f'VOL_10C_{tenor}']/100
            v10p = row[f'VOL_10P_{tenor}']/100
            if any(v<=0 or v>2 or np.isnan(v) for v in [sa,v25c,v25p,v10c,v10p]):
                sabr_records.append({'date':dt,'tenor':tenor,'alpha':np.nan,'rho':np.nan,
                                     'nu':np.nan,'atm_error':np.nan,'wing_rmse':np.nan,'ok':False})
                continue
            a, rho, nu, ae, rmse, ok = calibrate_sabr(S, F, T, sa, v25c, v25p, v10c, v10p)
            sabr_records.append({'date':dt,'tenor':tenor,'alpha':a,'rho':rho,
                                 'nu':nu,'atm_error':ae,'wing_rmse':rmse,'ok':ok})

    sabr_df = pd.DataFrame(sabr_records)
    print("  Smoothing SABR parameters...", flush=True)
    sabr_df = smooth_sabr_params(sabr_df, ew_alpha=0.8, spike_threshold=2.0)
    sabr_df.to_parquet(f'{data_dir}/usdjpy_sabr_params.parquet', index=False)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    for t in CALIBRATION_TENORS:
        m = sabr_df[sabr_df['tenor']==t]
        print(f"    {t}: {m['ok'].mean()*100:.1f}% ok  RMSE={m['wing_rmse'].mean()*100:.3f}vp  "
              f"rho={m['rho'].mean():.3f}  nu={m['nu'].mean():.3f}", flush=True)

    sabr_lookup = build_sabr_lookup(sabr_df, dates)
    print("  SABR lookup arrays built", flush=True)

    # ── [3/7] Spot-vol betas ────────────────────────────────────────────
    print("\n[3/7] Spot-vol betas...", flush=True)
    beta_up, beta_down, rho_sv = estimate_spot_vol_betas(pipe_df['SPOT'], pipe_df['ATM_1M'])
    print(f"  beta_up={beta_up.mean():.1f}  beta_down={beta_down.mean():.1f}  "
          f"rho={rho_sv.mean():.3f}", flush=True)

    # ── [4/7] Build aging portfolio ─────────────────────────────────────
    print("\n[4/7] Building aging portfolio...", flush=True)
    legs, n_cohorts = build_cohorts(pipe_df, dates)
    print(f"  {len(legs)} legs, {n_cohorts} cohorts", flush=True)
    print(f"  Book: long ATM straddle, short 2x 25D call, long 1x 25D put", flush=True)

    # ── [5/7] Pre-compute Greeks at close + 3 intraday snapshots ────────
    print(f"\n[5/7] Pre-computing Greeks at close + UTC hours {HEDGE_HOURS}...", flush=True)

    hfeat = compute_hourly_features(hourly)

    snaps = {}
    for hr in HEDGE_HOURS:
        s = hfeat[hfeat.index.hour == hr].copy()
        s.index = pd.to_datetime(s['date'])
        snaps[hr] = s

    spot_c = pipe_df['SPOT'].values
    spot_h = {hr: snaps[hr].reindex(dates)['mid'].ffill().bfill().values for hr in HEDGE_HOURS}

    bu_v  = beta_up.values
    bd_v  = beta_down.values
    dstd  = np.log(pipe_df['SPOT']/pipe_df['SPOT'].shift(1)).rolling(20,min_periods=5).std().fillna(0.01).values

    spot_open = np.roll(spot_c, 1); spot_open[0] = spot_c[0]
    vbump_h = {}
    for hr in HEDGE_HOURS:
        ret_h    = np.log(spot_h[hr] / spot_open)
        sig_mask = np.abs(ret_h) > 0.7 * dstd
        beta_arr = np.where(ret_h > 0, bu_v, bd_v)
        vbump_h[hr] = np.where(sig_mask, np.clip(beta_arr * ret_h / 100, -0.05, 0.05), 0.0)

    # Output arrays
    # continuity_pnl : MTM change of continuing legs only  → RL reward signal
    # inception_pnl  : premium paid/received at cohort inception
    # expiry_pnl     : settlement minus last MTM at cohort expiry (~0)
    # full_pnl       : complete economic P&L = continuity + inception + expiry
    cols_float = (
        ['delta_close','gamma_close','vega_close','price_close',
         'continuity_pnl','inception_pnl','expiry_pnl','full_pnl']
        + [f'delta_{hr:02d}' for hr in HEDGE_HOURS]
        + [f'gamma_{hr:02d}' for hr in HEDGE_HOURS]
        + [f'price_{hr:02d}' for hr in HEDGE_HOURS]
        + [f'vol_bump_{hr:02d}' for hr in HEDGE_HOURS]
    )
    cols_int = ['n_legs', 'n_cohorts']
    out = {c: np.zeros(n) for c in cols_float}
    out.update({c: np.zeros(n, dtype=int) for c in cols_int})

    # Pre-index legs by expiry day for O(1) expiry lookup
    legs_expiring = {}
    for l in legs:
        legs_expiring.setdefault(l.exp, []).append(l)

    prev_leg_prices = {}
    t0 = time.time()

    for d in range(n):
        if (d+1)%500 == 0:
            el = time.time()-t0; rem = (n-d-1)/(d+1)*el
            print(f"    {d+1}/{n} ({el:.0f}s, ~{rem:.0f}s left)", flush=True)

        row = pipe_df.iloc[d]
        Sc  = spot_c[d]

        for hr in HEDGE_HOURS:
            out[f'vol_bump_{hr:02d}'][d] = vbump_h[hr][d]

        curr_leg_prices = {}
        dc, gc, vc, pc = 0., 0., 0., 0.
        nl = 0
        cont_pnl      = 0.0
        inception_pnl = 0.0   # premium paid (long) or received (short) today

        delta_h = {hr: 0. for hr in HEDGE_HOURS}
        gamma_h = {hr: 0. for hr in HEDGE_HOURS}
        price_h = {hr: 0. for hr in HEDGE_HOURS}
        bumped_alpha_cache = {hr: {} for hr in HEDGE_HOURS}

        for l in legs:
            if not l.alive(d): continue
            T    = l.T_rem(d)
            sign = np.sign(l.notional)
            w    = abs(l.notional) / BASE_NOTIONAL
            leg_key = (l.cohort, l.K, l.is_call)

            alpha_d, rho_d, nu_d = get_sabr_params(T, d, sabr_lookup)

            F_c    = get_forward(T, Sc, row)
            sig_c  = max(sabr_vol_beta1(l.K, F_c, T, alpha_d, rho_d, nu_d), 0.01)
            lp     = sign * w * gk_price(Sc, l.K, F_c, T, sig_c, l.is_call)
            dc    += sign * w * gk_delta(Sc, l.K, F_c, T, sig_c, l.is_call)
            gc    += sign * w * gk_gamma(Sc, l.K, F_c, T, sig_c)
            vc    += sign * w * gk_vega(Sc, l.K, F_c, T, sig_c)
            pc    += lp
            curr_leg_prices[leg_key] = lp
            nl    += 1

            if leg_key in prev_leg_prices:
                # Continuing leg: daily MTM change — pure gamma P&L for RL reward
                cont_pnl += lp - prev_leg_prices[leg_key]
            else:
                # New leg incepted today: record inception premium cash flow.
                # Long leg  (lp > 0): pay premium  → inception_pnl -= lp  (negative)
                # Short leg (lp < 0): receive premium → inception_pnl -= lp (positive)
                inception_pnl -= lp

            for hr in HEDGE_HOURS:
                S_hr = spot_h[hr][d]
                b_hr = vbump_h[hr][d]
                F_hr = get_forward(T, S_hr, row)
                if abs(b_hr) > 1e-6:
                    T_key = round(T, 6)
                    if T_key not in bumped_alpha_cache[hr]:
                        base_atm = sabr_vol_beta1(F_hr, F_hr, T, alpha_d, rho_d, nu_d)
                        bumped_alpha_cache[hr][T_key] = alpha_from_atm(
                            max(base_atm + b_hr, 0.01), F_hr, T, rho_d, nu_d)
                    alpha_hr = bumped_alpha_cache[hr][T_key]
                    sig_hr   = max(sabr_vol_beta1(l.K, F_hr, T, alpha_hr, rho_d, nu_d), 0.01)
                else:
                    sig_hr = max(sabr_vol_beta1(l.K, F_hr, T, alpha_d, rho_d, nu_d), 0.01)

                delta_h[hr] += sign * w * gk_delta(S_hr, l.K, F_hr, T, sig_hr, l.is_call)
                gamma_h[hr] += sign * w * gk_gamma(S_hr, l.K, F_hr, T, sig_hr)
                price_h[hr] += sign * w * gk_price(S_hr, l.K, F_hr, T, sig_hr, l.is_call)

        # ── Expiry settlement ────────────────────────────────────────────
        # Legs whose exp==d expired today. They were in prev_leg_prices but
        # are no longer alive, so not in curr_leg_prices.
        # settlement - last_MTM is typically near zero (last price ≈ intrinsic
        # at T_rem=1/365), but compute explicitly for correctness.
        expiry_pnl = 0.0
        for l in legs_expiring.get(d, []):
            leg_key = (l.cohort, l.K, l.is_call)
            if leg_key in prev_leg_prices:
                sign = np.sign(l.notional)
                w    = abs(l.notional) / BASE_NOTIONAL
                intrinsic  = max(Sc - l.K, 0.0) if l.is_call else max(l.K - Sc, 0.0)
                settlement = sign * w * intrinsic
                expiry_pnl += settlement - prev_leg_prices[leg_key]

        out['delta_close'][d]    = dc
        out['gamma_close'][d]    = gc
        out['vega_close'][d]     = vc
        out['price_close'][d]    = pc
        out['n_legs'][d]         = nl
        out['n_cohorts'][d]      = len(set(l.cohort for l in legs if l.alive(d)))
        out['continuity_pnl'][d] = cont_pnl
        out['inception_pnl'][d]  = inception_pnl
        out['expiry_pnl'][d]     = expiry_pnl
        out['full_pnl'][d]       = cont_pnl + inception_pnl + expiry_pnl

        for hr in HEDGE_HOURS:
            out[f'delta_{hr:02d}'][d] = delta_h[hr]
            out[f'gamma_{hr:02d}'][d] = gamma_h[hr]
            out[f'price_{hr:02d}'][d] = price_h[hr]

        prev_leg_prices = curr_leg_prices

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    raw_pnl = np.diff(out['price_close'], prepend=out['price_close'][0])
    cont    = out['continuity_pnl']
    n_big   = (np.abs(raw_pnl[1:] - cont[1:]) > 0.5).sum()
    print(f"  Continuity check: {n_big} cohort-roll days with gap>0.5", flush=True)

    inc_vals = out['inception_pnl'][out['inception_pnl'] != 0]
    exp_vals = out['expiry_pnl'][out['expiry_pnl'] != 0]
    print(f"  Inception days: {len(inc_vals)}  "
          f"mean|premium|={np.abs(inc_vals).mean():.4f}" if len(inc_vals) > 0
          else "  Inception days: 0", flush=True)
    print(f"  Expiry days:    {len(exp_vals)}  "
          f"mean|settlement_residual|={np.abs(exp_vals).mean():.6f}" if len(exp_vals) > 0
          else "  Expiry days: 0", flush=True)

    # ── [6/7] Build feature matrix ──────────────────────────────────────
    print("\n[6/7] Building feature matrix...", flush=True)
    feat = pd.DataFrame(index=dates)

    feat['atm_1m']           = pipe_df['ATM_1M']
    feat['atm_1m_chg']       = pipe_df['ATM_1M'].diff()
    feat['atm_3m']           = pipe_df['ATM_3M']
    feat['atm_3m_chg']       = pipe_df['ATM_3M'].diff()
    feat['term_slope_3m_1w'] = pipe_df['ATM_3M'] - pipe_df['ATM_1W']
    feat['rr25_1m']          = pipe_df['RR25_1M']
    feat['rr25_1m_chg']      = pipe_df['RR25_1M'].diff()
    feat['rr25_3m']          = pipe_df['RR25_3M']
    feat['bf25_1m']          = pipe_df['BF25_1M']
    feat['bf25_3m']          = pipe_df['BF25_3M']

    feat['spot_ret']         = np.log(pipe_df['SPOT']/pipe_df['SPOT'].shift(1))
    feat['carry_3m']         = pipe_df['FWD_3M'] / pipe_df['SPOT']
    feat['spread_bps']       = pipe_df['SPREAD_BPS']

    feat['beta_up']          = beta_up
    feat['beta_down']        = beta_down
    feat['rho_sv']           = rho_sv

    for c in cols_float + cols_int:
        feat[f'port_{c}'] = out[c]

    feat['spot_close'] = spot_c
    for hr in HEDGE_HOURS:
        feat[f'spot_{hr:02d}'] = spot_h[hr]

    hour_labels = {0: 'h00', 6: 'h06', 16: 'h16'}
    for hr in HEDGE_HOURS:
        pfx = hour_labels[hr]
        sr  = snaps[hr].reindex(dates)
        for col in ['intraday_zscore','intraday_cumret','ret_1h','ret_4h',
                    'ret_24h','hourly_range','spread','daily_ret_std']:
            if col in sr.columns:
                feat[f'{pfx}_{col}'] = sr[col].ffill().bfill()
        for w in [5, 10, 21, 63]:
            rc = f'rv_yz_{w}d'
            if rc in sr.columns:
                feat[f'{pfx}_{rc}'] = sr[rc].ffill().bfill()

    spot_log = np.log(pipe_df['SPOT'])

    for win in [21, 63]:
        log_ret_sq = (spot_log - spot_log.shift(win))**2
        rv_sum     = (spot_log.diff()**2).rolling(win).sum()
        feat[f'vr_minus1_{win}d'] = (log_ret_sq - rv_sum) / (rv_sum + 1e-12)

    for win in [21, 63]:
        roll_hi = pipe_df['SPOT'].rolling(win).max()
        roll_lo = pipe_df['SPOT'].rolling(win).min()
        feat[f'range_pos_{win}d'] = (pipe_df['SPOT'] - roll_lo) / (roll_hi - roll_lo + 1e-12)

    atm    = pipe_df['ATM_1M']
    ema_5  = atm.ewm(halflife=5).mean()
    ema_21 = atm.ewm(halflife=21).mean()
    ema_63 = atm.ewm(halflife=63).mean()
    feat['vol_ma_alignment'] = (np.sign(ema_5 - ema_21) + np.sign(ema_21 - ema_63)) / 2
    feat['vol_regime_63d']   = (atm / atm.rolling(63).mean() - 1).clip(-1, 1)

    spot_vol_21 = spot_log.diff().rolling(21).std() * np.sqrt(252)
    feat['regime_highvol']  = (spot_vol_21 > spot_vol_21.rolling(252).quantile(0.75)).astype(float)
    feat['regime_trending'] = (feat['vr_minus1_21d'] > 0.1).astype(float)
    feat['regime_mean_rev'] = (feat['vr_minus1_21d'] < -0.2).astype(float)

    ret_21 = spot_log.diff(21)
    feat['spot_trend_strength'] = ret_21 / (spot_log.diff().rolling(21).std() + 1e-12)

    feat['h00_to_close_lag1'] = (
        (pd.Series(spot_c, index=dates) - pd.Series(spot_h[0], index=dates))
        / (pd.Series(spot_h[0], index=dates) + 1e-12) * 10000
    ).shift(1)

    feat['tokyo_to_close_ret'] = (
        (pd.Series(spot_c, index=dates) - pd.Series(spot_h[0], index=dates))
        / (pd.Series(spot_h[0], index=dates) + 1e-12)
    ).shift(1)

    atm_arr  = pipe_df['ATM_1M'].values.astype(np.float64)
    spot_arr = np.log(pipe_df['SPOT']).values.astype(np.float64)
    rr_arr   = pipe_df['RR25_1M'].values.astype(np.float64)
    ts_arr   = (pipe_df['ATM_3M'] - pipe_df['ATM_1W']).values.astype(np.float64)
    feat['atm_slope_21d']        = calculate_slope_numba_standardized(atm_arr, 21)
    feat['spot_slope_21d']       = calculate_slope_numba_standardized(spot_arr, 21)
    feat['skew_slope_21d']       = calculate_slope_numba_standardized(rr_arr, 21)
    feat['term_slope_slope_21d'] = calculate_slope_numba_standardized(ts_arr, 21)

    feat = feat.ffill().bfill()
    feat.to_parquet(f'{data_dir}/usdjpy_features.parquet')

    # ── [7/7] Summary ───────────────────────────────────────────────────
    print("\n[7/7] Summary", flush=True)
    print(f"  usdjpy_sabr_params.parquet: {sabr_df.shape}", flush=True)
    print(f"  usdjpy_features.parquet:    {feat.shape}", flush=True)
    print(f"\n  Portfolio Greeks:", flush=True)
    print(f"    Avg delta (close): {feat['port_delta_close'].mean():.4f}", flush=True)
    print(f"    Avg |delta|:       {feat['port_delta_close'].abs().mean():.4f}", flush=True)
    print(f"    Avg gamma:         {feat['port_gamma_close'].mean():.6f}", flush=True)
    print(f"    Avg vega:          {feat['port_vega_close'].mean():.4f}", flush=True)
    print(f"    Avg live legs:     {feat['port_n_legs'].mean():.1f}", flush=True)
    for hr in HEDGE_HOURS:
        lbl = hour_labels[hr]
        print(f"    Avg |delta| {lbl}:  {feat[f'port_delta_{hr:02d}'].abs().mean():.4f}", flush=True)
    print(f"\n  P&L components:", flush=True)
    print(f"    Mean |continuity_pnl|: {feat['port_continuity_pnl'].abs().mean():.4f}", flush=True)
    print(f"    Mean |inception_pnl|:  {feat['port_inception_pnl'].abs().mean():.4f}", flush=True)
    print(f"    Mean |expiry_pnl|:     {feat['port_expiry_pnl'].abs().mean():.6f}", flush=True)
    print(f"    Mean |full_pnl|:       {feat['port_full_pnl'].abs().mean():.4f}", flush=True)
    print(f"    Cumulative full_pnl:   {feat['port_full_pnl'].sum():.4f}", flush=True)
    print(f"\n  Vol bumps active (days):", flush=True)
    for hr in HEDGE_HOURS:
        col = f'port_vol_bump_{hr:02d}'
        if col in feat.columns:
            print(f"    {hour_labels[hr]} UTC: {(feat[col]!=0).sum()} days", flush=True)
    print(f"\n{'='*70}", flush=True)
    print("STAGE 2 (JPY) COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    run_pipeline(d)
