import xgboost as xgb
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.metrics import mean_squared_error

def train_and_validate_model(X_train, X_test, y_train, y_test, w_train, w_test,train_dmatrix,test_dmatrix,params_in):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    params = {
    'objective': 'reg:absoluteerror',
    }
    params.update(params_in)
    evals_result = {}
    xg_reg = xgb.train(params, train_dmatrix,evals=[(train_dmatrix, 'train'), (test_dmatrix, 'test')], evals_result=evals_result,  verbose_eval=False, num_boost_round=params["n_estimators"])

    #plot train vs test mean absulute error as a function of step
    plt.figure(figsize=(10, 5))
    plt.plot(evals_result['train']['mae'], label='Train ')
    plt.plot(evals_result['test']['mae'], label='Test ')
    plt.xlabel('Boosting Round')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.savefig("plots/Loss_vs_boosting_round_finalModel.png")
    plt.show()
    
    #predict on test data
    y_pred = xg_reg.predict(test_dmatrix)

    #calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    # Create an explainer using your trained model
    explainer = shap.TreeExplainer(xg_reg)

    #compute SHAP values for test data
    shap_values = explainer.shap_values(X_test)
    print(np.array(shap_values).shape)
    shap_values_r = shap_values[..., 0]  # SHAP values for the first output 'r'
    shap_values_puc = shap_values[..., 1]  # SHAP values for the second output 'PUc'
    
    #converts SHAP values for 'r' to a DataFrame
    shap_values_r_df = pd.DataFrame(shap_values_r, columns=X_test.columns)
    print(shap_values_r_df.describe())
    
    #converts SHAP values for 'PUc' to a DataFrame
    shap_values_puc_df = pd.DataFrame(shap_values_puc, columns=X_test.columns)
    print(shap_values_puc_df.describe())

    #plots SHAP summary for 'r'
    shap.summary_plot(shap_values_r, X_test, plot_size=(10,6)) # , max_display=X_test.shape[1] is for the max number of variables to be displayed (default is 20)
    plt.savefig("plots/Shap_BDT1_r.png",dpi=300)

    #plots SHAP summary for 'PU_c'
    shap.summary_plot(shap_values_puc, X_test, plot_size=(10,6)) # , max_display=X_test.shape[1] is for the max number of variables to be displayed (default is 20)
    plt.savefig("plots/Shap_BDT1_PU_c.png",dpi=300)


    #END SHAP

    #from now on: residual plotting

    y_pred = xg_reg.predict(xgb.DMatrix(X_test))
    y_pred_train = xg_reg.predict(xgb.DMatrix(X_train))

    residuals_train=np.array(y_train)-np.array(y_pred_train)


    #convert y_test to a numpy array (if it's a DataFrame)
    y_test_array = np.array(y_test)

    #compute residuals
    residuals = y_test_array - y_pred



    #convert residuals to a DataFrame for easy handling
    residuals_df = pd.DataFrame(residuals, columns=['r_residuals', 'PUc_residuals'])
    residuals_train_df=pd.DataFrame(residuals_train, columns=['r_residuals', 'PUc_residuals'])

    #calculate mean and standard deviation for both r_residuals and PUc_residuals
    r_mean = residuals_df['r_residuals'].mean()
    r_std = residuals_df['r_residuals'].std()


    PUc_mean = residuals_df['PUc_residuals'].mean()
    PUc_std = residuals_df['PUc_residuals'].std()

    # plotting residuals
    fig, axs = plt.subplots(1, 2, figsize=(18,8))
    rng=0.5
    bins = np.linspace(-rng, rng, 100)  # create bins
    residuals_df['r_residuals'].hist(bins=bins, ax=axs[0], density=True, alpha=0.8, label='Residuals test set')
    residuals_train_df['r_residuals'].hist(bins=bins, ax=axs[0], density=True, alpha=1, label='Residuals train set',histtype='step')

    axs[0].set_title(f'r_residuals')
    axs[0].legend()
    axs[0].text(0.1, 20, f'Test mean: {r_mean:.5f}\nStd dev: {r_std:.5f}')
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('density')


    min_value = min(residuals_df['PUc_residuals'].min(), residuals_train_df['PUc_residuals'].min())
    max_value = max(residuals_df['PUc_residuals'].max(), residuals_train_df['PUc_residuals'].max())
    rng=0.5
    bins = np.linspace(-rng, rng, 100)  # create bins
    residuals_df['PUc_residuals'].hist(bins=bins, ax=axs[1], density=True, alpha=0.8, label='Residuals test set',histtype='bar')
    residuals_train_df['PUc_residuals'].hist(bins=bins, ax=axs[1], density=True, alpha=0.8, label='Residuals train set',histtype='step')

    #legend
    axs[1].legend()

    axs[1].set_title(f'PUc_residuals')
    axs[1].text(0.05, 20, f'Test mean: {PUc_mean:.5f}\nStd dev: {PUc_std:.5f}')
    axs[1].set_xlabel('Residuals')
    axs[1].set_ylabel('density')
    plt.tight_layout()
    #plt.show()
    plt.savefig("plots/Residuals.png",dpi=300)