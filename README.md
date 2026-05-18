# MS Thesis — Deep RL Portfolio Allocation (Daily)

## Objective

Train and evaluate a **soft actor–critic (SAC)** agent for **daily portfolio allocation** over a large equity universe (NASDAQ-100–style membership), using **per-asset LSTM encoders** on return and feature histories, plus global market features. Performance is assessed with **walk-forward optimization** (rolling train / validation / test), compared to **baselines** (e.g. QQQ buy-and-hold and other rules) and summarized in metrics, equity curves, and notebooks for plots and statistical tests.

## Repository layout

```
MS-Thesis-Deep-RL-KK/
├── Code/
│   ├── 01_Executor.ipynb          # orchestration / experiments
│   ├── 03_Train_RL_Daily.py       # main daily RL training entrypoint
│   ├── 04_Plots.ipynb             # figures, archiving to Code_ALL / Results by HP tag
│   └── functions/
│       ├── data_pipeline_daily.py # loads CSVs → tensors / features
│       ├── environment.py       # trading env
│       ├── baseline.py          # baselines vs RL
│       └── RL_1/
│           ├── train.py           # WFO training loop, metrics, CSV outputs
│           ├── sac_agent.py
│           └── networks.py        # LSTM policy / value nets
├── Data/
│   ├── QQQ/Outputs/Data/          # example: QQQ-universe processed CSVs (see below)
│   ├── NKSY/… , EUSTX/…          # other universes (same Outputs/Data pattern)
│   └── *.ipynb                    # data download / membership prep
├── Results_Daily/                 # RL outputs (metrics, folds, ensemble, stats)
├── Statistics_Ensemble/           # ensemble / significance notebooks
├── Code_ALL/                      # optional: archived Code per HP tag (e.g. QQQ_LSTM_DAILY_1)
└── README.md
```

Training writes under **`Results_Daily/`** (and may use subfolders per asset or experiment). **`Code_ALL/<HP_TAG>/Code`** holds snapshots of **`Code/`** for a named hyperparameter run (see `04_Plots.ipynb`).

## Requirements

- **Python** 3.10+ (code uses modern syntax; not Python 2.7).
- **PyTorch ≥ 2.7.1** (with CUDA if you use GPU).
- **NumPy**, **pandas**, and other imports used in `Code/functions/` and notebooks (e.g. **matplotlib**, **plotly** where notebooks need them).

## Run environment (reference)

Reported runs used:

- **Cloud:** Google Cloud Platform  
- **Machine type:** `g2-standard-8`  
- **GPU:** NVIDIA **L4**  
- **RAM:** **30 GB**

Other hardware (CPU-only, Apple MPS, different GPUs) may work but timings and memory use will differ.

## Data and experiment tag

- The pipeline expects a folder with at least:  
  `close_prices.csv`, `tradable_mask.csv`, `QQQ.csv`, `VIX.csv`, `risk_free_data.csv`  
  (see `Code/functions/data_pipeline_daily.py`).

- In **`Code/03_Train_RL_Daily.py`**, the default path is **`../Data/Outputs/Data`**. This repo ships **QQQ**-universe files under **`Data/QQQ/Outputs/Data`**. Copy or symlink that directory to **`Data/Outputs/Data`**, or change the `build_dataset(...)` path.

- Reproducing a specific configuration: match the **HP tag** (e.g. **`QQQ_LSTM_DAILY_1`**) in **`Code_ALL`** so code and hyperparameters align with that specification.

**Current default in this project:** **QQQ** data, tag **`QQQ_LSTM_DAILY_1`**.

## Run training

From the **`Code/`** directory:

```bash
python 03_Train_RL_Daily.py
```

Outputs go to **`Results_Daily/`** (see `functions/RL_1/train.py`).

Currently its set for QQQ (LSTM_1).
