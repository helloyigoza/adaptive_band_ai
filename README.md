# Adaptive Learning Band AI

A research tool that trains a small GRU model to predict an **asymmetric price band**
(`mid`, `upper_offset`, `lower_offset`) from OHLCV data. The model learns a balance
line and the typical positive/negative excursion around it for the next
`max_hold_bars`.

> Disclaimer: This is a research / analysis tool. **It is not financial advice.**

## Highlights

- Pulls OHLCV data from Yahoo Finance with on-disk pickle caching.
- Variable-length sequence training (`seq_min`..`seq_max` per sample).
- GRU model that outputs `mid`, `upper_offset`, `lower_offset` and a directional
  probability head.
- Asymmetric upper/lower bands - the model can learn a different shape for
  rallies and selloffs.
- No online learning: `train` and `indicator` are two separate modes.
- GPU is used automatically when available (CUDA / ROCm); falls back to CPU.
  All checkpoints are exported in a CPU-loadable form.
- On CPU, all logical cores are used by default (`OMP_NUM_THREADS`,
  `MKL_NUM_THREADS` are set).
- Resume / checkpoint system: re-running training for the same
  `symbol+interval` automatically resumes from `last.pt`.

## Feature engineering

The model intentionally avoids the classic "bag of lagging indicators". The
feature pipeline (`adaptive_band/features.py` + `advanced_features.py`) is
strictly causal and built from low-lag, microstructure-aware and regime-aware
families:

**Geometry / candle microstructure (zero-lag)**
- Causal rolling min-max normalised OHLC and volume.
- `body_ratio`, `upper_wick`, `lower_wick`.
- `tr_over_atr` for range expansion / shock detection.

**Momentum**
- Log-returns at multiple horizons (`ret_1`, `ret_3`, `ret_5`, `ret_10`).
- `ret_accel` (second derivative).
- `close_z_ema` (close vs short EMA, scaled by recent stdev).

**Trend quality / position**
- Kaufman efficiency ratio (`eff_ratio`).
- Donchian position (`donchian_pos`).

**Volume / order-flow proxy**
- Volume z-score and signed volume z-score.

**Regime**
- Realised volatility ratio (`rv_ratio`, short / long).

**Advanced (Ehlers DSP, OHLC vol, microstructure, complexity)** - toggled by
`FeatureConfig.enable_advanced` (on by default):

- Ehlers Super Smoother (2-pole Butterworth) deviation and slope.
- Ehlers High-Pass detrended cycle component (z-scored).
- Hilbert Homodyne Discriminator dominant cycle period.
- Hilbert phase `sin` / `cos` (turning-point indicators).
- Garman-Klass, Parkinson, Rogers-Satchell OHLC volatility estimators.
- Bipower variation ratio (jump detector).
- Amihud illiquidity, Roll effective spread, Kyle lambda proxy.
- Permutation entropy (Bandt-Pompe), rolling Hurst (R/S), realised
  skewness / kurtosis.

## Installation

```bash
cd adaptive_band_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick usage

Short Turkish-style commands:

```bash
python3 main.py --egitim btc 1h
python3 main.py --indikator btc 1h
```

Subcommand form:

```bash
python3 main.py train BTC-USD 1h --epochs 25
python3 main.py indicator BTC-USD 1h --bars 250
```

More controlled training:

```bash
python3 main.py train btc 1d \
  --start 2022-01-01 \
  --epochs 40 \
  --seq-min 48 \
  --seq-max 160 \
  --max-hold-bars 50 \
  --hidden 128
```

The quick-mode flags accept the same arguments:

```bash
python3 main.py --egitim btc 1h --batch-size 256 --num-workers 0 --seq-max 72 --epochs 10
```

## Resume / fresh start

- **Auto-resume**: re-running training for the same `symbol+interval` resumes
  from `models/<symbol>_<interval>/last.pt` if it exists.
- **Fresh start**: pass `--fresh` or `--resume none`.
- **Specific checkpoint**: `--resume models/BTC_USD_1H/best.pt`.
- If the feature columns differ between checkpoint and current code, resume is
  skipped automatically and training starts from scratch.

## CPU performance tips

GRU is sequential, so on CPU the timestep count is the main bottleneck.

- Lower `--seq-max` (64-72 is enough for `1h`).
- `--num-workers 0` is often fastest for small datasets (no fork overhead).
- Tune `--torch-threads N` - more is not always better; small matmuls are
  sometimes faster with 4 threads than 8.

Example fast combo:

```bash
python3 main.py --egitim btc 1h --num-workers 0 --batch-size 256 --seq-max 72 \
  --eval-seq-len 64 --torch-threads 4
```

## File layout

```text
adaptive_band_ai/
  main.py
  requirements.txt
  adaptive_band/
    cli.py
    config.py
    data.py
    features.py
    advanced_features.py
    dataset.py
    model.py
    train.py
    inference.py
    signals.py
    plotting.py
    utils.py
```

## Outputs

- Models: `models/<symbol>_<interval>/`
- Plots: `outputs/<symbol>_<interval>/`
- Metrics: saved as JSON next to the plots.

## Design decisions

1. Raw OHLCV is never fed to the model directly - every bar is normalised over
   a causal rolling window.
2. The model learns in normalised space; bands are mapped back to real prices
   for plotting / signals.
3. `upper_offset` and `lower_offset` are learned independently, so the band can
   be asymmetric.
4. Training labels never use future data as input. The future is only used to
   build the supervision signal.
5. `seq_len` is not fixed: each training sample uses a random length in
   `[seq_min, seq_max]`.

## Yahoo Finance notes

Yahoo limits historical depth on intraday timeframes:

- `1h` / `60m` / `90m`: ~last 730 days.
- `30m` and smaller: ~last 60 days.
- `1d` and larger: no practical limit.

Out-of-range requests are clamped automatically and a `[warn] Yahoo intraday limit ...`
message is printed. For deeper history use `1d` or higher.

## Data cache

Downloaded frames are pickled into `data_cache/` keyed by
`symbol+interval+start+end`. Re-running with the same arguments reads from disk
instead of hitting Yahoo.

- Intraday TTL: 30 minutes
- Daily TTL: 6 hours
- Manual purge: `rm -rf data_cache/` or a single file
  `rm data_cache/BTC-USD_1h_*.pkl`.

Console message: `[cache] BTC-USD_1h_2022-01-01_now.pkl used (age=120s, ttl=1800s)`.

## Resampling (custom timeframes)

Yahoo does not natively serve `4h`, `2h`, `8h`, etc. The code resamples them
from a base interval automatically:

| Requested | Base | Pandas rule |
|-----------|------|-------------|
| `2h`  | `1h` | `2h`  |
| `3h`  | `1h` | `3h`  |
| `4h`  | `1h` | `4h`  |
| `6h`  | `1h` | `6h`  |
| `8h`  | `1h` | `8h`  |
| `12h` | `1h` | `12h` |
| `2d`  | `1d` | `2D`  |
| `3d`  | `1d` | `3D`  |

OHLCV aggregation: `open` first, `high` max, `low` min, `close` last, `volume` sum.

```bash
python3 main.py --egitim btc 4h --epochs 15
```

Console message: `[resample] 1h -> 4h (12500 bars -> 3125 bars)`.

## Transfer learning (across timeframes)

You can warm-start a model on a different timeframe:

```bash
# Transfer a 1h checkpoint to start training on 4h.
python3 main.py train BTC-USD 4h --epochs 10 --resume models/BTC_USD_1H/best.pt
```

- The new model is saved under `models/BTC_USD_4H/`; the old `1h` model is not
  touched.
- If the feature columns differ, the resume is skipped and training starts
  fresh.
- This typically converges faster than a cold start.

## Training logs

Each epoch prints something like:

```text
epoch 003/10 | time=12.3s lr=2.00e-04 | train_loss=0.31475 val_loss=0.31525 | val_mid=0.04134 val_up=0.01411 val_down=0.01365 val_dir=0.05123 | gn=0.847 train_dir=0.04912
  [best] val_loss improved to 0.31525 - saving best.pt & cpu_model.pt
```

| Field | Meaning |
|-------|---------|
| `time` | Epoch wall-clock seconds |
| `lr` | Current learning rate |
| `train_loss` / `val_loss` | Composite total loss |
| `val_mid` / `val_up` / `val_down` | Raw SmoothL1 errors for mid line, upper, lower offsets |
| `val_dir` / `train_dir` | Direction (BCE) error |
| `gn` | Gradient norm (clipped at 1.0) |
| `[best]` | Validation improved - checkpoint written |
