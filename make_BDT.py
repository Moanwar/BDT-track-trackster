import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import mplhep as hep
import vector as vec
import matplotlib
from tqdm import tqdm
import math
import os
import pickle
from functions import distWrap_numba,flatten_numba
from make_dataset import make_dataset
from plotting import save_histos_to_file
from train import train_and_validate_model
from bayesian_search import bayesian_search


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import sys
#sys.path.append('/eos/home-t/tipaulet/.local/lib/python3.9/site-packages')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import shap


import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
plt.style.use(hep.style.CMS)

import argparse

parser = argparse.ArgumentParser(description="BDT regression for trackster association analysis")
parser.add_argument("--do_all", action="store_true", help="Do all the work: load data, make plots, do bayesian search and train the model")
parser.add_argument("--load_data", action="store_true", help="(Re-)process .root files and save the dataset to a pickle file (BDT1_data.pkl)")
parser.add_argument("--make_plots", action="store_true", help="Make and save histograms of input features (saved in ./featureplots/)")
parser.add_argument("--bayesiansearch", action="store_true", help="Perform Bayesian hyperparameter search using training data")
parser.add_argument("--train_validate", action="store_true", help="Train final XGBoost model and make plots of residuals and SHAP values")
parser.add_argument("-j", "--n_jobs",type=int,default=-1, help="Number of jobs to use for bayesian search (default: -1, same as CPU cores)")
parser.add_argument("--n_iterations", type=int, default=50, help="Number of iterations for Bayesian search (default: 50)")
parser.add_argument("--folderlist", nargs="+", default=["./histos/"], help="List of input folders with step3 ROOT files (default: ['./histos/'])")

args = parser.parse_args()
load_data = args.load_data
make_plots = args.make_plots
bayesiansearch = args.bayesiansearch
train_validate = args.train_validate
n_cores = args.n_jobs
n_iterations = args.n_iterations
folderlist = args.folderlist
if args.do_all:
    load_data = True
    make_plots = True
    bayesiansearch = True
    train_validate = True

CUT=0.8





                                         
                    


    

if load_data:
    make_dataset(folderlist)

with open('BDT1_data.pkl', 'rb') as file:
    dict_info=pickle.load(file)

if make_plots:
    save_histos_to_file(dict_info)

print(f"Number of data points {len(dict_info['FirstR'])}")

#converting dictionaty to dataframe in order to use it with xgboost
df = pd.DataFrame(dict_info)
df_features = df[['eta_trk','phi_trk','pt_trk',"etaErr","phiErr", "etaphiCov","nTrks_0.005","nTrks_0.010","nTrks_0.030","nTrks_0.050","En_0.005","En_0.010","En_0.030","En_0.050", "FirstE", "FirstR"]]
df_label = df[['R', 'contamination']]
df_weights=df[["simEnergy"]]

#split in training and test sets, weight given by the simEnergy
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(df_features, df_label, df_weights, test_size=0.2, random_state=32)

#convert to DMatrix format for xgboost
train_dmatrix = xgb.DMatrix(X_train, label=y_train, weight=w_train)
test_dmatrix = xgb.DMatrix(X_test, label=y_test, weight=w_test)


if bayesiansearch:
    bayesian_search(X_train, y_train, w_train,X_test,y_test,w_test,n_iterations,n_cores)
    
with open('BDT1_best_params.pkl', 'rb') as file:
    best_params=pickle.load(file)

if train_validate:
    train_and_validate_model(X_train, X_test, y_train, y_test, w_train, w_test, train_dmatrix, test_dmatrix, best_params)
    



