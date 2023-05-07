# Active Learning Benchmark

Related work: [#SV74 A Comparative Survey: Benchmarking for Pool-based Active Learning](https://ijcai-21.org/program-survey/)
(*Update on 04/17* We found Zhan et al. released their source code: <https://github.com/SineZHAN/ComparativeSurveyIJCAI2021PoolBasedAL>)

## Requriements

- Ubuntu >= 20.04.3 LTS (focal)
- Python >= 3.8, for [ntucllab/libact](https://github.com/ntucllab/libact)

## Installation

0. (optional) `apt install vim git python3 python3-venv build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev -y`
1. `git clone https://github.com/ariapoy/active-learning-benchmark.git mlrc2021; cd mlrc2021`
2. `python3 -m venv mlrc21-env; source mlrc21-env/bin/activate`
3. `pip install -r requirements.txt`
4. `git clone https://github.com/ariapoy/active-learning.git`
5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cp -r alipy-dev/alipy alipy-dev/alipy_dev`
6. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; python setup.py build; python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`
7. `cd data; bash get_data_zhan21.sh; cd ..`
8. `cd src; python main.py -h`

**Warning!** If you use Python == 3.10

5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cd alipy-dev; git checkout py3.10; cd .. ; cp -r alipy-dev/alipy alipy-dev/alipy_dev`

**Warning!** If your env cannot support liblapack

6. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; LIBACT_BUILD_VARIANCE_REDUCTION=0 python setup.py build; LIBACT_BUILD_VARIANCE_REDUCTION=0 python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`

You CANNOT obtain the results of Variability Reduction (VR) for the benchmark.

## Quick Start

1. Run an experiment by yourself.

```shell
cd src; python main.py -h  # see help function
```

2. Reproduce all results in Zhan et al. (Warning! It will spend you very long time!)

```shell
cd src;
bash run-reproduce.sh  # run all small datasets
bash run-reproduce-large.sh  # run all large datasets
bash run-reproduce-infeasible.sh  # run all infeasible time datasets, only for time test
```
**Note**
- `N_JOBS`: number of workers. User can accelerate according to their number of CPUs.
  **WARNING!** Some methods could be slower because of insufficient resource.

3. Reproduce all figures and tables in this work.

```shell
cd results; gdown 1qzezDD_fe43ctNBHC4H5W0w6skJcBlxB -O aubc.zip;
unzip aubc.zip;
gdown 1xKUT3CHHOwYY0yFxak1XKf3vWiAXQFSQ -O detail.zip;
unzip detail.zip;
# open and run analysis.ipynb
```
