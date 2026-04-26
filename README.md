# Pool-Based Active Learning Benchmark

This repository provides a practical CLI for running pool-based active learning
experiments on tabular classification datasets. It also keeps the original
reproduction workflow for the benchmark paper:

> An Expanded Benchmark that Rediscovers and Affirms the Edge of Uncertainty
> Sampling for Active Learning in Tabular Datasets, TMLR 2025.

The default entrypoint is now `src/main.py`. Use `src/legacy_main.py` when you
need to reproduce the original benchmark scripts exactly.

## Installation

Create an environment and install the requirements matching your Python version:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With `uv`, the Python 3.13 setup is:

```shell
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements-py313.txt
```

For Python 3.12 or 3.13:

```shell
pip install -r requirements-py312.txt
# or
pip install -r requirements-py313.txt
```

On macOS, install CMake first and use the macOS requirements:

```shell
brew install cmake
pip install -r requirements-macos.txt
```

Some legacy strategies depend on local forks or older builds of libact, ALiPy,
and scikit-activeml. The new `main.py` can run sklearn-native experiments
without cloning those legacy folders. For exact reproduction, follow the legacy
setup from the paper scripts and run `legacy_main.py`.

## Quick Start

Run a sklearn built-in tabular dataset:

```shell
cd src
python main.py --dataset sklearn:breast_cancer --strategy margin --model RandomForest
```

Run a LIBSVM-format dataset:

```shell
cd src
python main.py --dataset libsvm:/absolute/path/to/data.txt --strategy entropy --model LR
```

List available datasets and strategies:

```shell
python main.py --list-datasets
python main.py --list-strategies
```

Results are written to `results/runs/<run-name>/` with:

- `aubc.csv`
- `learn_curve.csv`
- `f1_score_curve.csv`
- `class_dist.csv`
- `detail.csv`
- `metadata.json`

## CLI Options

Common options:

```shell
python main.py \
  --dataset sklearn:wine \
  --strategy entropy \
  --model RandomForest \
  --seed 0 \
  --n-trials 5 \
  --init-labels 20 \
  --budget 100 \
  --batch-size 1 \
  --test-size 0.4 \
  --scale
```

Supported model names:

- `LR`
- `RBFSVM`
- `RandomForest`
- `XGBoost`

Dataset specs:

- `sklearn:breast_cancer`
- `sklearn:digits`
- `sklearn:iris`
- `sklearn:wine`
- `libsvm:/path/to/file`

Labels are automatically encoded to integer classes. The initial labeled pool
contains at least one example from every class, so `--init-labels` must be at
least the number of classes.

## Query Strategies

The CLI exposes stable strategy names grouped by active learning taxonomy.
Some strategies are implemented directly in the new sklearn-native runner;
others are listed with their legacy implementation so users can discover them
and run them through `legacy_main.py`.

| Category | CLI name | Strategy | Framework / note |
|---|---|---|---|
| Baseline | `random` | Random sampling | sklearn-native |
| Model uncertainty | `least_confidence` | Least Confidence | sklearn-native |
| Model uncertainty | `margin` | Margin Sampling | sklearn-native |
| Model uncertainty | `entropy` | Entropy Sampling | sklearn-native |
| Model uncertainty | `qbc` | Query-by-Committee | sklearn-native bagging committee |
| Expected model changing / improvement | `expected_average_precision` | Expected Average Precision | sklearn-native approximation; legacy `skal_us_eap` |
| Expected model changing / improvement | `eer` | Expected Error Reduction | sklearn-native approximation; legacy `eer` |
| Expected model changing / improvement | `mc_eer` | Monte Carlo EER | sklearn-native approximation; legacy `skal_eer` |
| Expected model changing / improvement | `variance_reduction` | Variance Reduction | libact legacy `vr` |
| Representative Exploiting | `coreset` | CoreSet | sklearn-native farthest-first |
| Representative Exploiting | `kcenter` | K-Center Greedy | sklearn-native farthest-first |
| Representative Exploiting | `graph_density` | Graph Density | legacy Google AL |
| Representative Exploiting | `hierarchical` | Hierarchical Sampling | legacy Google AL |
| Hybrid criteria | `quire` | QUIRE | libact legacy |
| Hybrid criteria | `dwus` | DWUS | libact legacy |
| Hybrid criteria | `infodiv` | Information-Diversity | legacy Google AL |
| Hybrid criteria | `mcm` | Margin Cluster Mean | legacy Google AL |
| Hybrid criteria | `bmdr` | BMDR | ALiPy legacy |
| Hybrid criteria | `spal` | SPAL | ALiPy legacy |
| Bayesian methods | `bald` | BALD | scikit-activeml legacy |
| Bayesian methods | `greedy_bald` | GreedyBALD | scikit-activeml legacy |
| Bayesian methods | `batch_bald` | BatchBALD | scikit-activeml integration target |
| Bayesian methods | `epistemic_uncertainty` | Epistemic Uncertainty Sampling | sklearn-native committee approximation |
| Redesigned Learning Framework | `albl` | ALBL | libact bandit-learning legacy |
| Redesigned Learning Framework | `lal` | LAL | ALiPy legacy |
| Oracle / answer-peeking diagnostic | `bso` | BSO / beam-search optimization | not a fair deployable AL method |

Strategies marked as legacy are discoverable from `main.py --list-strategies`,
but the new sklearn-native runner will ask you to use `legacy_main.py` for the
original implementation.

## Experiment Recipes

Compare uncertainty strategies on the same dataset:

```shell
cd src
for strategy in least_confidence margin entropy qbc; do
  python main.py --dataset sklearn:breast_cancer --strategy "$strategy" --model RandomForest --budget 100
done
```

Use a small budget for a smoke test:

```shell
python main.py --dataset sklearn:iris --strategy random --model LR --init-labels 6 --budget 5
```

Use scaling with an SVM:

```shell
python main.py --dataset sklearn:wine --strategy margin --model RBFSVM --scale
```

Name an output folder explicitly:

```shell
python main.py --dataset sklearn:digits --strategy coreset --model RandomForest --run-name digits-coreset-rf
```

## Reproduction Workflow

The original reproduction-oriented implementation is preserved in
`src/legacy_main.py`. Existing shell scripts under `src/` now call that entrypoint.

Examples:

```shell
cd src
bash run-reproduce-google.sh
bash run-reproduce-libact.sh
bash run-reproduce-alipy.sh
bash run-reproduce-bso.sh
```

The legacy workflow expects the original dataset layout and, for some methods,
local folders such as `libact-dev`, `alipy-dev`, `scikit-activeml-dev`, and
`active-learning`.

## Citation

If you use this benchmark, please cite:

```bibtex
@article{lu2025an,
  title={An Expanded Benchmark that Rediscovers and Affirms the Edge of Uncertainty Sampling for Active Learning in Tabular Datasets},
  author={Po-Yi Lu and Yi-Jie Cheng and Chun-Liang Li and Hsuan-Tien Lin},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=855yo1Ubt2},
  note={Reproducibility Certification}
}
```

```bibtex
@inproceedings{zhan2021comparative,
  title={A Comparative Survey: Benchmarking for Pool-based Active Learning.},
  author={Zhan, Xueying and Liu, Huan and Li, Qing and Chan, Antoni B},
  booktitle={IJCAI},
  pages={4679--4686},
  year={2021}
}
```

## Contact

Please open an issue or contact Po-Yi (Poy) Lu at
`ariapoy@gmail.com` / `d09944015@csie.ntu.edu.tw`.
