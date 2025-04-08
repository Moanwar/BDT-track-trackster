    

import mplhep as hep
import os 
import matplotlib.pyplot as plt

def save_histos_to_file(data_dict):
    #creates the "featureplots" directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists("plots/featureplots"):
        os.makedirs("plots/featureplots")

    for key, values in data_dict.items():
        if not values:  # If the list is empty, skip
            print(f"Empty list for '{key}', no histogram done.")
            continue

        #plotting single features 
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=100, alpha=1.0, log=True)
        plt.title(f"Feature '{key}'")
        plt.xlabel(key)
        plt.ylabel("Counts")
        plt.grid(True)

        #save the plot as a separate file
        plt.savefig(os.path.join("plots/featureplots", f"{key}_histogram.png"))
        plt.close() 
