# Active Learning Benchmark

Related work: [#SV74 A Comparative Survey: Benchmarking for Pool-based Active Learning](https://ijcai-21.org/program-survey/)

## Requriements

- Ubuntu >= 20.04.3 LTS (focal)
- Python >= 3.8, for [ntucllab/libact](https://github.com/ntucllab/libact)

## Installation

0. (optional) `apt install vim git python3 python3-venv -y`
1. `git clone https://github.com/ariapoy/active-learning-benchmark.git mlrc2021; cd mlrc2021`
2. `python3 -m venv mlrc21-env; source mlrc21-env/bin/activate`
3. `pip install -r requirements.txt`
4. `git clone https://github.com/ariapoy/active-learning.git`
5. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cp -r alipy-dev/alipy alipy-dev/alipy_dev`
6. `apt install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev -y`
7. `git clone https://github.com/ariapoy/libact.git libact-dev; cd libact-dev; python setup.py build; python setup.py install; cd ..; cp -r libact-dev/libact libact-dev/libact_dev`
8. `cd data; bash get_data_zhan21.sh; cd ..`.
9. `cd src; python main.py -h`.

**Warning!** If you use Python == 3.10

1. `git clone https://github.com/ariapoy/ALiPy.git alipy-dev; cd alipy-dev; git checkout py3.10; cd .. ; cp -r alipy-dev/alipy alipy-dev/alipy_dev`

## Quick Start

To run several experiments of the query strategy on the dataset.

1. `bash run.sh`, sequentially run multiple datasets-experiments with specific query strategy, and save results at the end.
2. `bash run-once.sh` parallel run multiple seeds-experiments with specific query strategy, datasets, and save results at each iteration.

After you finish it, you can mv results and *collect.py*, *verify.py* to specific directory.

1. `mkdir ${QS-NAME}; mv *.csv *.txt collect.py verify.py ${QS-NAME}; cd ${QS-NAME}`
2. `python collect.py`, collect all results from different SERVER.
3. `python verify.py`, calculate statistics(mean) of AUBC.

## Run on AWS EC2

0. `ssh -i ${KEY-NAME}.pem ubuntu@${IP ADDRESS}`
1. `sudo apt update; sudo apt install python3.8-venv -y`
2. `sudo apt-get install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev -y`
3. Follow above **Installation**.

After you install the enviornments, you can measure computation resources and running time by:

1. `time bash run-time.sh`


