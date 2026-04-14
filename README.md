# USD/JPY Delta Hedge Timing — RL Pipeline

A reinforcement learning system that learns **when** to execute spot FX hedges 
on a USD/JPY options portfolio, minimising hedge error volatility relative to a 
fixed benchmark.

## The Problem

A delta hedger must decide — at each intraday window — how much of the portfolio 
delta to hedge and at which time. The naive benchmark hedges 100% of delta at 
16:00 UTC daily. This pipeline trains an RL agent to exploit intraday spot and 
vol dynamics to do better.

## Portfolio Structure

- Long ATM straddles + short 25-delta calls (long gamma book)
- Weighted average maturity ~3 months
- Cohorts roll every 21 business days, expire at 63 business days
- Notional scaled by realised vol regime at inception

## Hedge Windows

Three intraday decision points per day:
- **00:00 UTC** — Tokyo open
- **06:00 UTC** — Tokyo afternoon
- **16:00 UTC** — London/NY overlap

## Pipeline
Stage 1 (external): usdjpy_pipeline_creator.py  →  usdjpy_pipeline_ready.parquet & usdjpy_hourly_ohlc.csv  [committed]
Stage 2:            sabr_calibration_jpy.py      →  usdjpy_features.parquet
Stage 3:            rl_hedge_training_jpy.py     →  rl_metrics.csv, trained models

**Stage 2** calibrates a SABR vol surface (beta=1) across tenors daily, builds 
an aging portfolio of GK-priced options, and computes pre-computed Greeks at 
each hedge window. Produces a full feature matrix for the RL agent.

**Stage 3** trains a RecurrentPPO (LSTM) agent using Stable-Baselines3 in a 
walk-forward validation framework. Optuna tunes hyperparameters per window. 
The reward signal penalises squared hedge error and transaction costs.

## Usage

```powershell
# Stage 2
python -u sabr_calibration_jpy.py C:\BackbookData_jpy ```or anyother filename```

# Stage 3
python -u rl_hedge_training_jpy.py C:\BackbookData_jpy 200000 20
# args: data_dir  total_timesteps  optuna_trials
```

## Key Design Choices

- **SABR at actual strikes**: as options age and spot moves, a leg struck at 
  25-delta at inception may drift to 10-delta. SABR prices each leg at its 
  actual strike daily rather than using the original delta label.
- **Reward scaling calibrated per currency**: JPY hedge error standard deviations 
  are larger than ZAR; reward_scaling is tuned via Optuna to avoid entropy collapse.
- **Continuity P&L**: cohort rolls (every 21 days) are handled cleanly — only 
  P&L from legs alive on both days is counted, eliminating artificial jump P&L.
- **Walk-forward validation**: 6 expanding windows (2020–2025), each with its 
  own Optuna tuning on the validation year before testing.

## Benchmark

Hedge 100% of portfolio delta at 16:00 UTC every day. The RL agent is evaluated 
against this on out-of-sample test years using hedge error volatility reduction 
as the primary metric.

## Requirements
Python 3.11, CPU only (no GPU required).

## Data

`usdjpy_pipeline_ready.parquet` contains daily vol surface data (ATM, 25-delta 
and 10-delta risk reversals and butterflies across tenors), forward points, spot, 
and bid/ask spreads. Source: Various
`usdjpy_hourly_ohlc.csv` contains hourly quote tick data of usdjpy OHLC metrics. Source: Quotes from Various
