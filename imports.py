# -*- coding: utf-8 -*-
"""
__author__ = "Mahamdi Mohammed"
__copyright__ = "Copyright 2020, PFE"
__license__ = "MIT"
__version__ = "0.2"
__email__ = "fm_mahamdi@esi.dz"
__status__ = "Production"
"""

from sklearn.tree import export_graphviz
from IPython.display import display
import IPython
import graphviz
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from utils import display_all, str_to_cat, proc_df,split_vals,print_score,apply_cats,draw_tree
import numpy as np
import pandas as pd
import dill, os
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.svm import SVC



from class_report import *