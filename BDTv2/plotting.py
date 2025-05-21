    

import mplhep as hep
import os 
import matplotlib.pyplot as plt

def save_histos_to_file(data_dict):
    #creates the "featureplots" directory if it doesn't exist
    if not os.path.exists("/eos/user/m/moanwar/www/linking/bdt_plots/plots"):
        os.makedirs("/eos/user/m/moanwar/www/linking/bdt_plots/plots")
    if not os.path.exists("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots"):
        os.makedirs("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots")

    for key, values in data_dict.items():
        if not values:  # If the list is empty, skip
            print(f"Empty list for '{key}', no histogram done.")
            continue
        print(" plotting is : ", key)
        #plotting single features 
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=100, alpha=1.0, log=True)
        plt.title(f"Feature '{key}'")
        plt.xlabel(key)
        plt.ylabel("Counts")
        plt.grid(True)

        #save the plot as a separate file
        plt.savefig(os.path.join("/eos/user/m/moanwar/www/linking/bdt_plots/plots/featureplots", f"{key}_histogram.png"))
        plt.close() 
