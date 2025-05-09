import xgboost as xgb
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.metrics import mean_squared_error

#y_train_transformed = np.log1p(y_train)
#y_pred = np.expm1(model.predict(X_test))  # inverse transform
#'max_depth': 6
#'eta': 0.05
#'n_estimators': 600

def train_and_validate_model(X_train, X_test, y_train, y_test, w_train, w_test,train_dmatrix,test_dmatrix,params_in):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    params = {
    'objective': 'reg:absoluteerror',
    }
    params.update(params_in)
    print(" params ", params)
    evals_result = {}
    xg_reg = xgb.train(params, train_dmatrix,evals=[(train_dmatrix, 'train'), (test_dmatrix, 'test')], evals_result=evals_result,  verbose_eval=False, num_boost_round=params["n_estimators"])

    #plot train vs test mean absulute error as a function of step
    plt.figure(figsize=(10, 5))
    plt.plot(evals_result['train']['mae'], label='Train ')
    plt.plot(evals_result['test']['mae'], label='Test ')
    plt.xlabel('Boosting Round')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Loss_vs_boosting_round_finalModel.png")
    plt.show()
    
    #predict on test data
    y_pred = xg_reg.predict(test_dmatrix)

    #xg_reg.save_model('model.bin')
    xg_reg.save_model("BDT1_BestModel.json")
        
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
    #shap_values_rpull = shap_values[..., 2] 
    #shap_values_mtd = shap_values[..., 3]
    
    #converts SHAP values for 'r' to a DataFrame
    shap_values_r_df = pd.DataFrame(shap_values_r, columns=X_test.columns)
    print(shap_values_r_df.describe())
    
    #converts SHAP values for 'PUc' to a DataFrame
    shap_values_puc_df = pd.DataFrame(shap_values_puc, columns=X_test.columns)
    print(shap_values_puc_df.describe())

    #shap_values_rpull_df = pd.DataFrame(shap_values_rpull, columns=X_test.columns)
    #print(shap_values_rpull_df.describe())
    
    #shap_values_mtd_df = pd.DataFrame(shap_values_mtd, columns=X_test.columns)
    #print(shap_values_mtd_df.describe())

    shap.summary_plot(shap_values_r, X_test, plot_size=(10,6), show=False)
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Shap_BDT1_r.png", dpi=300, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values_puc, X_test, plot_size=(10,6), show=False)
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Shap_BDT1_PU_c.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    #shap.summary_plot(shap_values_rpull, X_test, plot_size=(10,6), show=False)
    #plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Shap_BDT1_rpull_c.png", dpi=300, bbox_inches='tight')
    #plt.close()
    
    #shap.summary_plot(shap_values_mtd, X_test, plot_size=(10,6), show=False)
    #plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Shap_BDT1_mtd_c.png", dpi=300, bbox_inches='tight')
    #plt.close()
    #END SHAP

    #from now on: residual plotting

    # Predict
    y_pred = xg_reg.predict(xgb.DMatrix(X_test))
    y_pred_train = xg_reg.predict(xgb.DMatrix(X_train))

    # Compute residuals
    residuals = np.array(y_test) - y_pred
    residuals_train = np.array(y_train) - y_pred_train

    # Convert to DataFrame
    residuals_df = pd.DataFrame(residuals, columns=['r_residuals', 'PUc_residuals'])#, 'Rpull_residuals', 'mtd_residuals'])
    residuals_train_df = pd.DataFrame(residuals_train, columns=['r_residuals', 'PUc_residuals'])#, 'Rpull_residuals', 'mtd_residuals'])

    # Plotting setup
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    bins = np.linspace(-0.5, 0.5, 100)

    residual_keys = ['r_residuals', 'PUc_residuals']#, 'Rpull_residuals', 'mtd_residuals']
    positions = [(0, 0), (0, 1)]#, (1, 0), (1, 1)]
    
    for key, (i, j) in zip(residual_keys, positions):
        ax = axs[i][j]
        test_vals = residuals_df[key]
        train_vals = residuals_train_df[key]
        mean = test_vals.mean()
        std = test_vals.std()
        
        test_vals.plot.hist(bins=bins, ax=ax, density=True, alpha=0.8, label='Test set')
        train_vals.plot.hist(bins=bins, ax=ax, density=True, alpha=1.0, label='Train set', histtype='step')
        ax.axvline(mean, color='k', linestyle='--', label='Test mean')
        
        ax.set_title(f"{key}")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')
        
        # Adjust text location to top-left corner
        ymax = ax.get_ylim()[1]
        xmin = ax.get_xlim()[0]
        ax.text(xmin + 0.02, ymax * 0.85, f"Mean: {mean:.4f}\nStd: {std:.4f}", verticalalignment='top')

    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Residuals_All.png", dpi=300)
    plt.close()
    '''
    # Define targets and their respective indices in the prediction output
    targets = {
        'R': 0,
        'contamination': 1,
        #'Rpull': 2,
        #'MTDvalue': 3
    }
    # Axis limits for each plot (adjust if needed)
    xlims = {
        'R': (0, 0.5),
        'contamination': (0, 0.5),
        #'Rpull': (0, 0.5),
        #'MTDvalue': (0, 0.5)
    }
    
    for name, idx in targets.items():
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test[name], y_pred[:, idx], alpha=0.4, label=name)
        plt.plot([0, 0.5], [0, 0.5], 'r--', label='Ideal')
        
        plt.xlabel(f'True {name}')
        plt.ylabel(f'Predicted {name}')
        plt.title(f'Predicted vs True {name}')
        #plt.xlim(xlims[name])
        #plt.ylim(xlims[name])
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Predicted_{name}.png", dpi=300)
        plt.close()
    '''
    # Define targets and their respective indices in the prediction output
    targets = {
        'R': 0,
        'contamination': 1,
        # 'Rpull': 2,
        # 'MTDvalue': 3
    }
    
    # Axis limits for each plot (adjust if needed)
    xlims = {
        'R': (0, 0.5),
        'contamination': (0, 0.5),
        # 'Rpull': (0, 0.5),
        # 'MTDvalue': (0, 0.5)
    }
    
    for name, idx in targets.items():
        plt.figure(figsize=(6, 6))
        
        # Apply inverse log (exp) for R only
        if name == 'R':
            true_vals = np.exp(y_test[name])
            pred_vals = np.exp(y_pred[:, idx])
        else:
            true_vals = y_test[name]
            pred_vals = y_pred[:, idx]
            
        plt.scatter(true_vals, pred_vals, alpha=0.4, label=name)
        plt.plot([0, 0.5], [0, 0.5], 'r--', label='Ideal')
        
        plt.xlabel(f'True {name}')
        plt.ylabel(f'Predicted {name}')
        plt.title(f'Predicted vs True {name}')
        plt.xlim(xlims[name])
        plt.ylim(xlims[name])
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Predicted_{name}.png", dpi=300)
        plt.close()
