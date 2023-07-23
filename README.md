# Pool-Based Active Learning Benchmark

*Update on 07/23*, we merge our paper to the benchmark."[A More Robust Baseline for Active Learning by Injecting Randomness to Uncertainty Sampling](https://icml.cc/virtual/2023/27400)". Please change the branch to **robust-baseline**.

- `git checkout robust-baseline`

## Requriements

- Ubuntu >= 20.04.3 LTS (focal)
- Python >= 3.8, for [ntucllab/libact](https://github.com/ntucllab/libact)

## Installation

0. (optional) `apt install vim git python3 python3-venv build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev -y`
1. `git clone https://github.com/ariapoy/active-learning-benchmark.git act-benchmark; cd act-benchmark`
2. `python3 -m venv act-env; source act-env/bin/activate`
3. `pip install -r requirements.txt`
4. `git clone https://github.com/ariapoy/active-learning.git`
5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cp -r alipy-dev/alipy alipy-dev/alipy_dev`
6. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; python setup.py build; python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`
7. `cd data; bash get_data_zhan21.sh; cd ..`
8. `cd src; python main.py -h`

**Warning!** If you use Python == 3.11

3. `pip install -r requirements-3.11.txt`

**Warning!** If you use Python >= 3.10

5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cd alipy-dev; git checkout py3.10; cd .. ; cp -r alipy-dev/alipy alipy-dev/alipy_dev`

**Warning!** If your env cannot support liblapack

6. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; LIBACT_BUILD_VARIANCE_REDUCTION=0 python setup.py build; LIBACT_BUILD_VARIANCE_REDUCTION=0 python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`

You CANNOT obtain the results of Variability Reduction (VR) for the benchmark.

## Quick Start

1. Run an experiment by yourself.

```shell
cd src; python main.py -h  # see help function
```

2. Reproduce all results in the paper. (Warning! It will take you a very long time!)

```shell
cd src;
bash run-benchmark.sh  # run epsilon-uncertainty sampling with different epsilons for the benchmark
bash run-epsSchedule.sh  # run linear scheduling for the epsilon-uncertainty sampling
bash run-biasvar.sh  # run to get history of query index for bias and variance analysis
```
**Note**
- `N_JOBS`: number of workers. Users can accelerate according to their number of CPUs.
  **WARNING!** Some methods could be slower because of insufficient resources.

3. Reproduce all figures and tables in this work.

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
If you have any further questions or want to discuss Active Learning with me, please contact Po-Yi (Poy) Lu <ariapoy@gmail.com>/<d09944015@csie.ntu.edu.tw>.
