import functools
import numpy as np
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool

import argparse
import copy
import os
import time
import logging
from tqdm import tqdm

# For libact
import sys
try:
    sys.path.append("../libact-dev/")
    from libact_dev.base.dataset import Dataset as libact_Dataset
    from libact_dev.models import SklearnProbaAdapter as libact_skProbaAdapter
    from libact_dev.models import SklearnContiAdapter as libact_skContiAdapter
    from libact_dev.labelers import IdealLabeler
    from libact_dev.query_strategies import KCenterGreedy as libactKCG
    from libact_dev.query_strategies import UncertaintySampling as libact_US
    from libact_dev.query_strategies import ActiveLearningByLearning as libact_ALBL
    from libact_dev.query_strategies import HintSVM as libact_HSVM
    from libact_dev.query_strategies import DWUS as libact_DWUS
    from libact_dev.query_strategies import RandomSampling as libact_RS
    from libact_dev.query_strategies import QUIRE as libact_QUIRE
    from libact_dev.query_strategies import QueryByCommittee as libact_QBC
    from libact_dev.query_strategies import VarianceReduction as libact_VR
except Exception as e:
    pass

# For Google Active Learning Playground
sys.path.append("../active-learning/")
from sampling_methods.constants import get_wrapper_AL_mapping
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import AL_MAPPING
get_wrapper_AL_mapping()

# # For ALiPy debugging
from alipy.data_manipulate import split
sys.path.append("../alipy-dev/")
from alipy_dev.query_strategy import QueryInstanceRandom
from alipy_dev.query_strategy import QueryInstanceLAL
from alipy_dev.query_strategy import QueryExpectedErrorReduction, QueryInstanceBMDR, QueryInstanceSPAL
from alipy_dev.query_strategy import QueryInstanceRandom, QueryInstanceUncertainty, QueryInstanceQBC
from alipy_dev.experiment import State
from alipy_dev.utils.multi_thread import aceThreading

# For scikit-activeml
sys.path.append("../scikit-activeml-dev/")
from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling, BatchBALD, EpistemicUncertaintySampling, QueryByCommittee, Quire, MonteCarloEER, GreedyBALD, CoreSet
from skactiveml.pool import RandomSampling as skal_RS
