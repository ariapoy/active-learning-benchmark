# Pool-Based Active Learning Benchmark

This is an implementation of our paper: "[Re-benchmarking Pool-Based Active Learning for Binary Classification](https://arxiv.org/abs/2306.08954)"

We reproduce and re-benchmark the previous work: [#SV74 A Comparative Survey: Benchmarking for Pool-based Active Learning](https://ijcai-21.org/program-survey/)

*Update on 04/17*, we notice that *Zhan et al.* released the source code: <https://github.com/SineZHAN/ComparativeSurveyIJCAI2021PoolBasedAL>

*Update on 07/23*, we merge our paper to the benchmark."[A More Robust Baseline for Active Learning by Injecting Randomness to Uncertainty Sampling](https://icml.cc/virtual/2023/27400)". Please change the branch to **robust-baseline**.

- `git checkout robust-baseline`

## Quick start

**Call for Contribution and Future Work**

We call for the community to further provide more experimental results to this benchmark.
We provide below suggested future work:

1. Models for tabular datasets
    - Random Forest
    - Gradient Boosting Decision Trees
2. Tasks and domains
    - multi-class classifications
    - regression problems
    - image classifications
    - object detection
    - natural language processing
4. Evaluation metrics
    - Deficiency score, Data Utilization Rate, Start Quality, and Average End Quality

### How to start?

1. Provide new datasets: we support LIBSVM dataset format.
    - Update `data/get_data.sh` to download more tabular data.
2. Provide new query strategy: we support libact, Google, and ALiPy modules.
    - Update `src/config.py` to import new query strategies.
3. Provide new experimental settings: we provide common settings as arguments.
    - Update `src/main.py` to adjust the settings such as the size of a test set, size of an initial labeled pool, query-oriented model, task-oriented model, etc.

```shell
cd data; bash get_data.sh  # download datasets
cd ..;
cd src; python main.py  # run experiments, you will see two CSV files. *-aubc.csv* and *-detail.csv*
python main.py -h  # call helper functions
```

## Requriements

- Ubuntu >= 20.04.3 LTS (focal)
- Python >= 3.8, for [ntucllab/libact](https://github.com/ntucllab/libact)

## Installation

Note. We only verify the installation steps on Ubuntu. Please raise the issue if you have any problem.

When you use Python in [3.8, 3.9].

0. (optional) `apt install vim git python3 python3-venv build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev -y`
1. `git clone https://github.com/ariapoy/active-learning-benchmark.git act-benchmark; cd act-benchmark`
2. `python3 -m venv act-env; source act-env/bin/activate`
3. `pip install -r requirements.txt`
4. `git clone https://github.com/ariapoy/active-learning.git`
5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cp -r alipy-dev/alipy alipy-dev/alipy_dev`
6. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; python setup.py build; python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`
7. `git clone https://github.com/ariapoy/scikit-activeml.git scikit-activeml-dev; cp -r scikit-activeml-dev/skactiveml scikit-activeml-dev/skactiveml_dev`
8. `cd data; bash get_data_zhan21.sh; cd ..`
9. `cd src; python main.py -h`

**Warning!** If you use Python == 3.13

3\. `pip install -r requirements-py313.txt`

**Warning!** If you use Python == 3.12

3\. `pip install -r requirements-py312.txt`

**Warning!** If you use Python >= 3.11

6\. `git clone https://github.com/ariapoy/libact.git libact-dev; cp -r libact-dev/libact libact-dev/libact_dev`

You CANNOT obtain the results of Hinted Support Vector Machine (HintSVM) and Variability Reduction (VR) for the benchmark.

**Warning!** If you use Python == 3.10

5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cd alipy-dev; git checkout py3.10; cd .. ; cp -r alipy-dev/alipy alipy-dev/alipy_dev`

**Warning!** If your env cannot support liblapack

6. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; LIBACT_BUILD_VARIANCE_REDUCTION=0 python setup.py build; LIBACT_BUILD_VARIANCE_REDUCTION=0 python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`

You CANNOT obtain the results of Variability Reduction (VR) for the benchmark.

**Warning!** If your OS is macOS.

0\. `brew install cmake`

3\. `pip install -r requirements-macos.txt`

## Step-by-Step tutorial

Bellowing examples demonstrate how to use the benchmark for quick use, evaluating existing AL query strategies on your own datasets, and adding new AL query strategies for evaluating.

### Running AL experiments

This is an example of running compatible uncertainty sampling (US-C) on *Haberman* dataset based on RBF kernel SVM.

`python main.py --tool google --qs_name margin-zhan --hs_name google-zhan --gs_name zhan --seed 0 --n_trials 1 --data_set haberman;`

### Evaluation (WIP)

TODO. Split evaluation part from *main.py*

### Customize datasets (WIP)

### Customize query strategies (WIP)

## Reproduce all experiments for [Re-benchmarking Pool-Based Active Learning for Binary Classification](https://arxiv.org/abs/2306.08954)

### List of current settings

1. Settings of initial pools
    - Size of test set (`--tst_size=0.4`): $40\%$
    - Size of initial labeled pool (`--init_lbl_size=20`): $20$.
    - Construction of initial labeled pool (`--exp_name="RS"`): random split training set (not test set) into labeled and unlabeled pools.
    - Data preprocessing (`--exp_names="scale"`): apply `scaler = StandardScaler()` to dataset.
2. List of query strategies, their corresponding query-oriented model, and task-oriented model.
	- task-oriented model $\mathcal{G}$: SVM(RBF)

| QS       | query-oriented model                                                      |
|----------|---------------------------------------------------------------------------|
| US-C     | SVM(RBF)                                                                  |
| US-NC    | LR(C=0.1)                                                                 |
| QBC      | LR(C=1); SVM(Linear, probability=True); SVM(RBF, probability=True); LDA   |
| VR       | LR(C=1)                                                                   |
| EER      | SVM(RBF, probability=True)                                                |
| Core-Set | N/A                                                                       |
| Graph    | N/A                                                                       |
| Hier     | N/A                                                                       |
| HintSVM  | SVM(RBF)                                                                  |
| QUIRE    | SVM(RBF)                                                                  |
| DWUS     | SVM(RBF)                                                                  |
| InfoDiv  | SVM(RBF)                                                                  |
| MCM      | SVM(RBF)                                                                  |
| BMDR     | SVM(RBF)                                                                  |
| SPAL     | SVM(RBF)                                                                  |
| ALBL     | # Combination of QSs with same query-oriented model: US-C; US-NC; HintSVM |
| LAL      | SVM(RBF)                                                                  |

### Steps of reproducing

1. Reproduce all results in Zhan et al. (Warning! It will take you a very long time!)

```shell
cd src;
bash run-reproduce-google.sh  # run all google datasets
bash run-reproduce-libact.sh  # run all libact datasets
bash run-reproduce-libact.sh  # run all libact datasets
bash run-reproduce-bso.sh  # run all bso datasets
bash run-reproduce-infeasible.sh  # run all infeasible time datasets, only for time test
```
**Note**
- `N_JOBS`: number of workers. Users can accelerate according to their number of CPUs.
  **WARNING!** Some methods could be slower because of insufficient resources.

2. Reproduce all figures and tables in this work.

```shell
cd results; gdown 1qzezDD_fe43ctNBHC4H5W0w6skJcBlxB -O aubc.zip;
unzip aubc.zip;
gdown 1xKUT3CHHOwYY0yFxak1XKf3vWiAXQFSQ -O detail.zip;
unzip detail.zip;
python analysis.py;  # choice 1
# open and run analysis.ipynb  # choice 2
```

## Citing
If you use our code in your research or applications, please consider citing our and previous papers.

```
@misc{lu2023rebenchmarking,
      title={Re-Benchmarking Pool-Based Active Learning for Binary Classification}, 
      author={Po-Yi Lu and Chun-Liang Li and Hsuan-Tien Lin},
      year={2023},
      eprint={2306.08954},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@inproceedings{zhan2021comparative,
  title={A Comparative Survey: Benchmarking for Pool-based Active Learning.},
  author={Zhan, Xueying and Liu, Huan and Li, Qing and Chan, Antoni B},
  booktitle={IJCAI},
  pages={4679--4686},
  year={2021}
}
```

## Contact
If you have any further questions or want to discuss Active Learning with me, please leave issues or contact Po-Yi (Poy) Lu <ariapoy@gmail.com>/<d09944015@csie.ntu.edu.tw>.
