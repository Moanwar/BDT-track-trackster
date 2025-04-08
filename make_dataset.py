import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from functions import distWrap_numba,flatten_numba
import math

CUT=0.8

def make_dataset(folderlist=["./histos/"]):
    dict_info={"R" : [],
               "eta_trk" : [],
               "phi_trk" : [],
               "pt_trk" : [],
               "deltaR_trk" : [],
               "etaErr" : [],
               "phiErr" : [],
               "etaphiCov" : [],
               "contamination" : [],
               "simEnergy" : [],
               "simPt" : [],
               "nTrks_0.005" : [],
               "nTrks_0.010" : [],
               "nTrks_0.030" : [],
               "nTrks_0.050" : [],
               "En_0.005" : [],
               "En_0.010" : [],
               "En_0.030" : [],
               "En_0.050" : [],
               "FirstR" : [],
               "FirstE" : []
               
              }
    
    print("Searching for .root files in these folders: ", folderlist)
    eventcounter=0 
    for directory in folderlist:
        print(f"Processing directory: {directory}")
            
        #check if folder exists
        if not os.path.isdir(directory):
            print(f"Directory {directory} does not exist. Skipping.")
            continue
        
        #for loop in files in directory
        for filename in tqdm(os.listdir(directory)):
            if ".sys" in filename: continue
            if filename.endswith(".root"):
                file_path = os.path.join(directory, filename)
            file=uproot.open(file_path)
            print(f"Processing file: {file_path}")
            try: #avoids some faulty .root files with no information contained in them
                alltracksters = file['ticlDumper/ticlTracksterLinks']
                allsimtrackstersCP = file['ticlDumper/simtrackstersCP']
                alltracks = file["ticlDumper/tracks"]
                allassociations = file['ticlDumper/associations']
                tsCP=allsimtrackstersCP.arrays(["trackIdx","regressed_energy","regressed_pt"])
                tracksters = alltracksters.arrays(["barycenter_phi","barycenter_eta","raw_energy"])
                associations = allassociations.arrays(['ticlTracksterLinks_recoToSim_CP_sharedE',"ticlTracksterLinks_recoToSim_CP"])
                trks=alltracks.arrays(["track_hgcal_eta","track_hgcal_phi","track_pt","track_id", 'track_hgcal_etaErr','track_hgcal_phiErr','track_hgcal_etaphiCov'])
            except: 
                continue
            for ev in range(len(tsCP)):
                eventcounter+=1
                assEv=associations[ev]
                for i in range(len(tsCP[ev]["trackIdx"])):
                    if len(tsCP[ev]["trackIdx"])>2: print("Warning, more than 2 sim CPs are present in one event:",len(tsCP[ev]["trackIdx"]))
                    if tsCP[ev]["trackIdx"][i]==-1: continue
                    try:
                        trackIdx=np.where(trks[ev]["track_id"]==tsCP[ev]["trackIdx"][i])[0][0]
                    except ValueError:
                        continue
                    simEnergy=tsCP[ev]["regressed_energy"][i]
                    simPt=tsCP[ev]["regressed_pt"][i]
                    
                    refEta=trks[ev]["track_hgcal_eta"][trackIdx]
                    refPhi=trks[ev]["track_hgcal_phi"][trackIdx]
                    tsEv = tracksters[ev]
                    otherTsEta = tsEv["barycenter_eta"]
                    otherTsPhi = tsEv["barycenter_phi"]
                    distance = distWrap_numba(refEta, refPhi, otherTsEta, otherTsPhi)
                    idx_sort = np.array(distance).argsort()
                    distance_sorted = distance[idx_sort]
                    tsEnergy_sorted = tsEv.raw_energy[idx_sort]
                    sharedEnergy_sorted = assEv["ticlTracksterLinks_recoToSim_CP_sharedE"][idx_sort]
                    assocIdxs_sorted = assEv["ticlTracksterLinks_recoToSim_CP"][idx_sort]
                    correctTrackMask=(assocIdxs_sorted==i)
                    totalSharedE=np.sum(sharedEnergy_sorted[correctTrackMask])
                    distScan=np.linspace(0,0.7,100)
                    listE=[]
                    listContamination=[]
                    radius=-1
                    frac= np.cumsum(flatten_numba(sharedEnergy_sorted[(correctTrackMask)]))/totalSharedE
                    #contamination= 1.-np.cumsum(sharedEnergy_sorted[correctTrackMaskrackMask])
                    firstAboveThrIdx = np.argmax(frac>CUT)
                    radius=distance_sorted[firstAboveThrIdx]
                    
                    try:
                        contamination=1. - np.sum(sharedEnergy_sorted[(distance_sorted<radius) & (correctTrackMask) ])/np.sum(tsEnergy_sorted[distance_sorted<radius])
                    except:
                        contamination=0
                    if math.isnan(contamination):
                        contamination=0
                    
                    if radius!=-1:
                        dict_info["R"].append(radius)
                        dict_info["eta_trk"].append(abs(trks[ev]["track_hgcal_eta"][trackIdx]))
                        dict_info["phi_trk"].append(trks[ev]["track_hgcal_phi"][trackIdx])
                        dict_info["pt_trk"].append(trks[ev]["track_pt"][trackIdx])
                        dict_info["etaErr"].append(trks[ev]["track_hgcal_etaErr"][trackIdx]*1.5)
                        dict_info["phiErr"].append(trks[ev]["track_hgcal_phiErr"][trackIdx]*1.5)
                        dict_info["deltaR_trk"].append(math.sqrt((trks[ev]["track_hgcal_phiErr"][trackIdx]*1.5)**2+ (trks[ev]["track_hgcal_etaErr"][trackIdx]*1.5)**2))
                        dict_info["etaphiCov"].append(trks[ev]["track_hgcal_etaphiCov"][trackIdx]*1.5*1.5)
                        dict_info["contamination"].append(contamination)
                        dict_info["simEnergy"].append(simEnergy)
                        dict_info["simPt"].append(simPt)
                        
                        dict_info["nTrks_0.005"].append(len(tsEnergy_sorted[distance_sorted<0.005]))
                        dict_info["nTrks_0.010"].append(len(tsEnergy_sorted[distance_sorted<0.01]))
                        dict_info["nTrks_0.030"].append(len(tsEnergy_sorted[distance_sorted<0.03]))
                        dict_info["nTrks_0.050"].append(len(tsEnergy_sorted[distance_sorted<0.05]))
                        
                        dict_info["En_0.005"].append(np.sum(tsEnergy_sorted[distance_sorted<0.005]))
                        dict_info["En_0.010"].append(np.sum(tsEnergy_sorted[distance_sorted<0.01]))
                        dict_info["En_0.030"].append(np.sum(tsEnergy_sorted[distance_sorted<0.03]))
                        dict_info["En_0.050"].append(np.sum(tsEnergy_sorted[distance_sorted<0.05]))
                        
                        dict_info["FirstR"].append(distance_sorted[0])
                        FirstE=tsEnergy_sorted[distance_sorted == distance_sorted[0]]
                        if len(FirstE) > 1 : raise Exception("2 first tracksters")
                        dict_info["FirstE"].append(tsEnergy_sorted[distance_sorted == distance_sorted[0]][0])
                        
            
    
    with open('BDT1_data.pkl', 'wb') as handle:
            pickle.dump(dict_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Job done on {eventcounter} events, result saved in BDT1_data.pkl " )
