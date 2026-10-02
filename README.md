# Automated Hyperparameter Search for Time-Series Forecasting

A Python research project that connects market-data preparation, LSTM training, automated hyperparameter optimization, and evaluation in original price units. Given a window of historical observations, the model predicts the next **100 closing-price values** in a single forward pass.

The main engineering work is in the experiment loop: constructing consistent training examples, matching a custom loss to the forecast horizon, comparing model configurations with Optuna, preserving study history, and saving the model together with its preprocessing state.

**Stack:** TensorFlow / Keras · TensorFlow Addons · Optuna · NumPy · pandas · scikit-learn

## What the project demonstrates

- **Dataset engineering:** convert timestamped price and volume arrays into technical indicators, calendar features, and supervised sequence windows.
- **Model experimentation:** search lookback length, hidden-layer size, batch size, and optional elastic-net regularization while keeping other settings fixed for a targeted experiment.
- **Custom objectives:** implement the same exponentially weighted RMSE calculation in TensorFlow for training and NumPy for evaluation.
- **Experiment persistence:** resume a named Optuna study through database-backed storage and record each trial's score and prediction time.
- **Training artifacts:** save the best validation checkpoint and fitted scaler for subsequent inference.

## Pipeline

```text
Historical price / volume data
            |
   Feature engineering
            |
   Chronological 95% / 5% split
            |
   Fit scaler on the earlier partition
            |
   Build lookback windows + 100-step targets
            |
   LSTM -> Dropout -> Dense(100)
            |
   Inverse scaling + weighted RMSE
            |
   Optuna trial history / model checkpoint
```

### Data and features

The repository includes `data/1y_data.pickle`: five aligned arrays containing timestamps in milliseconds, closing prices, high prices, low prices, and volume. It contains **105,353 observations** spanning January 9, 2023 to January 9, 2024 UTC. Observations are normally five minutes apart, with a gap in the series. The file does not include asset, exchange, or source metadata.

`helper_functions.py` derives indicators and drops rows with missing values. The training and search scripts use these 13 features, in this order:

```text
close, high, low, volume, EMA_5, EMA_15, RSI, MACD,
Signal_Line, mean1, mean2, hour, day_of_week
```

`mean1 = (upper_shadow / volume + lower_shadow / volume) / 2`. `mean2 = ((close - derived_open) / volume + volume) / 2`. The derived open is the previous day's final close.

`create_dataset()` produces inputs with shape `(samples, lookback, 13)` and targets with shape `(samples, 100)`. Targets use the first feature, `close`. The horizon is measured in observations rather than guaranteed elapsed time.

### Model and training

The current configuration uses a single unidirectional LSTM, dropout, and a 100-output dense layer. The model-building code also supports stacked or bidirectional LSTMs and optional L1/L2 regularization.

The default optimizer is RectifiedAdam wrapped in Lookahead, a Ranger-style combination. Training runs for up to 100 epochs, with early stopping after seven epochs without validation improvement and learning-rate reduction after three. Keras reserves 20% of the training windows for validation.

`train.py` contains a fixed experiment configuration: a 133-observation lookback, 949 LSTM units, and a batch size of 768. These settings are separate from the active search configuration; they are not automatically loaded from the best Optuna trial.

### Forecast objective

For each forecast window, the loss is:

```text
w[t] = exp(-0.001 * t), for t = 0, ..., 99
window_rmse = sqrt(sum(w[t] * (prediction[t] - actual[t])^2) / sum(w[t]))
score = mean(window_rmse across all windows)
```

This gives earlier forecast steps slightly more weight. Training computes the loss on scaled prices; evaluation inverse-transforms predictions and targets before calculating the score in original price units.

## Repository guide

| File | Purpose |
| --- | --- |
| [`hyper_search.py`](hyper_search.py) | Optuna objective, model construction, training, scoring, and study persistence. |
| [`train.py`](train.py) | Train a fixed configuration, save the best checkpoint and scaler, and print the evaluation score. |
| [`helper_functions.py`](helper_functions.py) | Data loading, feature engineering, sequence generation, and weighted RMSE implementations. |
| [`data_exploration.ipynb`](data_exploration.ipynb) | Inspect the dataset, engineered features, and correlations. |
| [`data/1y_data.pickle`](data/1y_data.pickle) | Historical dataset used by the training and search scripts. |
| [`miner.py`](miner.py) | Earlier Bittensor serving integration; requires external modules and alignment with the current model. |
| [`test_model.py`](test_model.py) | Earlier inference experiment; requires updates before use. |
| [`requirements.txt`](requirements.txt) | Dependency list. |

## Running the training workflow

Run commands from the repository root so relative data and artifact paths resolve correctly.

### 1. Prepare an environment

The code uses TensorFlow 2 / Keras 2 and TensorFlow Addons. Addons has ended development; its [upstream compatibility matrix](https://github.com/tensorflow/addons#python-op-compatibility-matrix) lists TensorFlow 2.14 with Addons 0.22 and Python 3.9–3.11. The example below uses that pairing as a starting point. The repository has no dependency lockfile, and a complete installation and training run have not been verified for this README.

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt optuna-integration \
  "tensorflow==2.14.*" "tensorflow-addons==0.22.*" \
  "numpy<2" "pandas<3"
```

`requirements.txt` includes `mysqlclient`, which can require system MySQL development libraries. SQLite can be used for experiment storage without a database server. The command also installs the [Optuna integration package](https://github.com/optuna/optuna-integration) to satisfy the search script's unconditional `TFKerasPruningCallback` import, even though the callback is currently unused.

### 2. Train the fixed configuration

```bash
mkdir -p trained_models
python train.py
```

The script reads the bundled dataset, trains the model, and writes:

| Artifact | Contents |
| --- | --- |
| `trained_models/formless-v2_2.h5` | Model checkpoint with the best validation loss. |
| `trained_models/formless-v2_2_scaler.pkl` | MinMaxScaler fitted during this run. |

The final console output reports weighted RMSE on the later 5% partition. Keep the model and scaler together. `trained_models/` is excluded from version control, and pretrained weights are not included.

## Running a hyperparameter search

### 1. Configure study storage

Create a `.env` file in the repository root:

```dotenv
DATABASE_URL=sqlite:///optuna_study.db
```

The committed configuration uses the study name `formless-v2-bigsearch-5-noprune` with `load_if_exists=True`. Using the same storage URL and study name resumes the existing study. Without a storage URL, Optuna uses in-memory storage.

### 2. Set a trial budget and launch

The committed `study.optimize(objective)` call has no trial or time limit. For a bounded experiment, change the final call in `hyper_search.py` before running:

```python
study.optimize(objective, n_trials=20)
```

```bash
python hyper_search.py
```

The active search space is:

| Parameter | Values |
| --- | --- |
| LSTM hidden units | Integer from 950 to 1,050 |
| Lookback length | Integer from 20 to 250 observations |
| Elastic-net regularization | Enabled or disabled |
| L1 coefficient | Float from `1e-6` to `1e-1` |
| L2 coefficient | Float from `1e-6` to `1e-1` |
| Batch size | 768, 1,024, 1,280, or 1,408 |

Other settings, including optimizer choice, learning rate, dropout, and layer count, are fixed in the current experiment. L1/L2 coefficients only affect the model when elastic-net regularization is enabled. Commented suggestions preserve earlier, broader search options. Trial pruning is currently inactive.

### 3. Inspect the study and train a selected configuration

```python
import optuna

study = optuna.load_study(
    study_name="formless-v2-bigsearch-5-noprune",
    storage="sqlite:///optuna_study.db",
)
print("Best weighted RMSE:", study.best_value)
print("Parameters:", study.best_trial.params)
print("Measurements:", study.best_trial.user_attrs)
```

Each completed trial records `tested_rmse` and `inference_time`. Prediction time measures the complete `model.predict(X_test)` call; it is an exploratory batch timing measurement, not a standardized serving benchmark. The search records trial results but does not save trial models. Copy a selected configuration into `train.py` to train and save it.

## Evaluation scope and next improvements

This repository captures an experimental workflow. No benchmark report, trained checkpoint, or verified accuracy improvement is included. The code supports measurement, but the committed files alone do not establish predictive performance.

Several changes would make a stronger evaluation and reproducibility package:

- **Reserve an untouched final test period.** The search repeatedly uses the later 5% partition as its objective, so that partition serves as a search holdout.
- **Separate sequence boundaries carefully.** Overlapping windows are constructed before Keras splits validation data, so neighboring training and validation targets can overlap. The scaler also sees the validation portion of the earlier partition. Split and fit preprocessing within each evaluation fold.
- **Audit causal feature construction.** The first day's derived open is backfilled from that day's final close, and timestamp gaps need explicit treatment.
- **Add baselines and repeat runs.** Compare against a last-value forecast, report error by horizon, and measure variation across seeds or walk-forward folds.
- **Package reproducible runs.** Pin dependencies, record data provenance and hardware, export study results, and store the chosen configuration alongside each checkpoint.
- **Unify training and inference.** The legacy inference script references removed wavelet helpers, a missing two-year dataset, and different artifact names. The miner requires external `template` and `vali_config` modules plus Bittensor; it also uses a different feature set and a 57-observation lookback. Both need alignment with the current 13-feature model before use.

Sequence windows are materialized in memory, and large lookbacks and batch sizes can require substantial RAM and accelerator memory. Resource use should be measured alongside model quality when expanding the search.

## Attribution

`miner.py` retains the original MIT-license header and copyright notices for Yuma Rao and Taoshi Inc. The repository does not currently include a standalone license file.
