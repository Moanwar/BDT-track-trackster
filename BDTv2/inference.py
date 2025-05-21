import xgboost as xgb
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#y_train_transformed = np.log1p(y_train)
#y_pred = np.expm1(model.predict(X_test))  # inverse transform
#'max_depth': 6
#'eta': 0.05
#'n_estimators': 600

def run_inference(X_train, X_test, y_train, y_test, w_train, w_test,train_dmatrix,test_dmatrix,params_in,contamination_train,contamination_test):

    if not os.path.exists("plots"):
        os.makedirs("plots")

    print(" params ", params_in)

    best_model = XGBRegressor()
    best_model.load_model("BDT1_Bayes.json")

    # Predict
    y_pred = best_model.predict(X_test)  
    #plot train vs test mean absulute error as a function of step    
    #calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Create an explainer using your trained model
    explainer = shap.TreeExplainer(best_model)
    #compute SHAP values for test data
    X_test_sample = X_test.sample(n=100, random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    print(np.array(shap_values).shape)
    #shap_values_r = shap_values[..., 0]  # SHAP values for the first output 'r'
    #shap_values_puc = shap_values[..., 1]  # SHAP values for the second output 'PUc'
    #converts SHAP values for 'r' to a DataFrame
    shap_values_r_df = pd.DataFrame(shap_values, columns=X_test.columns)
    print(shap_values_r_df.describe())
    #converts SHAP values for 'PUc' to a DataFrame
    #shap_values_puc_df = pd.DataFrame(shap_values_puc, columns=X_test.columns)
    #print(shap_values_puc_df.describe())
    shap.summary_plot(shap_values, X_test_sample, plot_size=(10,6), show=False)
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Shap_BDT1_r.png", dpi=300, bbox_inches='tight')
    plt.close()
    #shap.summary_plot(shap_values_puc, X_test, plot_size=(10,6), show=False)
    #plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Shap_BDT1_PU_c.png", dpi=300, bbox_inches='tight')
    #plt.close()    
    #from now on: residual plotting

    # Predict
    print("step 0 is running ")
    y_pred = best_model.predict(X_test)
    print("step 1 is running ")
    y_pred_train = best_model.predict(X_train)

    print("y_test shape:", np.shape(y_test))
    print("y_pred shape:", np.shape(y_pred))

    y_train_flat = y_train.values.flatten()
    y_test_flat = y_test.values.flatten()
    residuals = y_test_flat - y_pred
    residuals_train = y_train_flat - y_pred_train
    residuals_df = pd.DataFrame({'residual': residuals})
    residuals_train_df = pd.DataFrame({'residual': residuals_train})


    # Compute residuals
    print("step 2 is running ")
    print("step 3 is running ")
    # Convert to DataFrame
    print("step 4 is running ")
    print("step 5 is running ")
    #print(" residuals_train_df ", residuals_train_df)
    # Choose the residual key to plot
    residual_key = 'r_residuals'
    # Extract values
    print("residuals_train type:", type(residuals_train))
    print("residuals_train shape:", np.shape(residuals_train))
    print(" residuals_train_df.head() ",residuals_train_df.head())
    print(" residuals_train_df.dtypes ",residuals_train_df.dtypes)
    print(" residuals_train_df.shape ",residuals_train_df.shape)
    print(" residuals_train_df.memory_usage = ",residuals_train_df.memory_usage(deep=True))

    print("step 6 is running ")
    test_vals = residuals_df.values.flatten()
    print("step 7 is running ")
    train_vals = residuals_train_df.values.flatten()
    # Compute mean and std for test set
    print("step 8 is running ")
    mean = test_vals.mean()
    print("step 9 is running ")
    std = test_vals.std()
    # Plot
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-0.5, 0.5, 100)
    # Test residuals
    print("step 10 is running ")
    plt.hist(test_vals, bins=bins, density=True, alpha=0.8, label='Test set')
    # Train residuals
    print("step 11 is running ")
    plt.hist(train_vals, bins=bins, density=True, alpha=1.0, label='Train set', histtype='step')    
    # Vertical line at mean
    print("step 12 is running ")
    plt.axvline(mean, color='k', linestyle='--', label='Test mean')
    # Labels and text
    print("step 13 is running ")
    plt.title(f"Residuals: {residual_key}")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    # Add mean and std as text on the plot
    ymax = plt.gca().get_ylim()[1]
    xmin = plt.gca().get_xlim()[0]
    plt.text(xmin + 0.02, ymax * 0.85, f"Mean: {mean:.4f}\nStd: {std:.4f}", verticalalignment='top')
    # Save and close
    plt.tight_layout()
    print("step 14 is running ")
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Residuals_All.png", dpi=300)
    print("step 15 is running ")
    plt.close()
    #pt_trk
    
    true_vals = np.expm1(y_test.values).ravel()
    pred_vals = np.expm1(y_pred)
    #true_vals = y_test.values + 0.3
    #pred_vals = y_pred + 0.3
    residuals = pred_vals - true_vals

    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals, residuals, alpha=0.4, color='purple')
    plt.axhline(y=0, color='k', linestyle='--', lw=1, label='Zero Residual')
    plt.xlabel('True R')
    plt.ylabel('Residual (Predicted - True)')
    plt.title('Residuals vs True R')
    plt.xlim(0, 0.2)
    plt.ylim(-0.3,0.3)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Residuals_vs_TrueR.png", dpi=300)
    plt.close()


    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Predicted R - True R (Residuals)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.xlim(-0.5,0.5)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/ResidualHistogram_R.png", dpi=300)
    plt.close()

    #true_r = np.expm1(y_test['R'])
    #pred_r = np.expm1(y_pred[:, targets['R']])
    # Avoid division by zero by adding a small epsilon if necessary
    epsilon = 1e-6
    response = pred_vals / (true_vals + epsilon)

    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals, response, alpha=0.4, color='brown')
    plt.axhline(y=1, color='k', linestyle='--', lw=1, label='Ideal Response')
    plt.xlabel('True R')
    plt.ylabel('Predicted R / True R (Response)')
    plt.title('Response Plot vs True R')
    plt.xlim(0,0.5)
    plt.grid(True)
    # plt.ylim(0.5, 1.5) # Adjust y-limits as needed
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Response_vs_TrueR.png", dpi=300)
    plt.close()

    # Extract pt_trk from X_test
    pt_trk_vals = X_test["pt_trk"].values

    # Plot pt_trk vs true_vals
    plt.figure(figsize=(8, 6))
    plt.scatter(pt_trk_vals, true_vals, alpha=0.4, color='blue', label='True R')
    plt.xlabel('pt_trk')
    plt.ylabel('True R')
    plt.xlim(0,120)
    plt.ylim(0,0.5)
    plt.title('True R vs pt_trk')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/TrueR_vs_pt_trk.png", dpi=300)
    plt.close()
    
    # Plot pt_trk vs pred_vals
    plt.figure(figsize=(8, 6))
    plt.scatter(pt_trk_vals, pred_vals, alpha=0.4, color='green', label='Predicted R')
    plt.xlabel('pt_trk')
    plt.ylabel('Predicted R')
    plt.xlim(0,120)
    plt.title('Predicted R vs pt_trk')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/PredR_vs_pt_trk.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.4, color='red', label='Predicted R')
    plt.xlabel('True R')
    plt.ylabel('Predicted R')
    plt.title('True vs Predicted R')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/True_vs_PredR.png", dpi=300)
    plt.close()

    # Correlation plot: contamination vs predicted R
    plt.figure(figsize=(8,6))
    plt.scatter(pred_vals, contamination_test, alpha=0.3, color='teal')
    plt.xlabel('Predicted R')
    plt.ylabel('Contamination')
    plt.title('Contamination vs Predicted R')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Contamination_vs_PredR.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(pt_trk_vals, contamination_test, alpha=0.3, color='darkorange')
    plt.xlabel('pt_trk')
    plt.ylabel('Contamination')
    plt.title('Contamination vs pt_trk')
    plt.xlim(0,120)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Contamination_vs_pt_trk.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(true_vals, contamination_test, alpha=0.3, color='seagreen')
    plt.xlabel('True R')
    plt.ylabel('Contamination')
    plt.title('Contamination vs True R')
    plt.xlim(0,0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots/Contamination_vs_TrueR.png", dpi=300)
    plt.close()

    
