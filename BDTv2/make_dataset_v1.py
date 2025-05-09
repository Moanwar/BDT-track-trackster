import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from functions import distWrap_numba,flatten_numba,distpull_numba,mtdValue_numba
import math

CUT=0.8

def make_dataset(folderlist=["./histos/"]):
    dict_info={"R" : [],
               "eta_trk" : [],
               "phi_trk" : [],
               "pt_trk" : [],
               "en_trk" : [],
               "deltaR_trk" : [],
               "deltaR" : [],
               "MTD_value"  : [],
               "DeltaR_pull": [],
               "en_trks": [],
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
                tracksters = alltracksters.arrays(["barycenter_phi","barycenter_eta","raw_energy","time",
                                                   "timeError","regressed_energy","barycenter_etaError",
                                                   "barycenter_phiError", 'barycenter_x', 'barycenter_y', 'barycenter_z'])
                associations = allassociations.arrays(['ticlTracksterLinks_recoToSim_CP_sharedE',"ticlTracksterLinks_recoToSim_CP"])
                trks=alltracks.arrays(["track_hgcal_eta","track_hgcal_phi","track_pt","track_id", 'track_hgcal_etaErr',
                                       'track_hgcal_phiErr','track_hgcal_etaphiCov', 'track_time_err','track_time_mtd',
                                       'track_time_mtd_err','track_p','track_pos_mtd','track_beta', 'track_pos_mtd.theVector.theX',
                                       'track_pos_mtd.theVector.theY','track_pos_mtd.theVector.theZ','track_quality',
                                       'track_missing_outer_hits'])

            except: 
                continue
            for ev in range(len(tsCP)):
                eventcounter+=1
                assEv=associations[ev]
                for i in range(len(tsCP[ev]["trackIdx"])):
                    if len(tsCP[ev]["trackIdx"])>2: print("Warning, more than 2 sim CPs are present in one event:",len(tsCP[ev]["trackIdx"]))
                    if tsCP[ev]["trackIdx"][i]==-1: continue
                    matches = np.where(trks[ev]["track_id"] == tsCP[ev]["trackIdx"][i])[0]
                    if len(matches) == 0:
                        continue
                    trackIdx = matches[0]
                    if trks[ev]["track_missing_outer_hits"][trackIdx] > 4 :
                        continue
                    simEnergy=tsCP[ev]["regressed_energy"][i]
                    simPt=tsCP[ev]["regressed_pt"][i]
                    refEta=trks[ev]["track_hgcal_eta"][trackIdx]
                    refPhi=trks[ev]["track_hgcal_phi"][trackIdx]
                    refEtaErr=trks[ev]["track_hgcal_etaErr"][trackIdx]
                    refPhiErr=trks[ev]["track_hgcal_phiErr"][trackIdx]
                    refx=trks[ev]["track_pos_mtd.theVector.theX"][trackIdx]
                    refy=trks[ev]["track_pos_mtd.theVector.theY"][trackIdx]
                    refz=trks[ev]["track_pos_mtd.theVector.theZ"][trackIdx]
                    refBeta=trks[ev]["track_beta"][trackIdx]
                    reftime=trks[ev]["track_time_mtd"][trackIdx]
                    reftimeErr=trks[ev]["track_time_mtd_err"][trackIdx]
                    refEnergy=trks[ev]["track_p"][trackIdx]
                    if (
                            refBeta == 0.0 and
                            refx == 0.0 and
                            refy == 0.0 and
                            refz == 0.0 and
                            reftime == 0.0 and
                            reftimeErr < 0.0 
                    ):
                        continue 
                    tsEv=tracksters[ev]
                    otherTsEta=tsEv["barycenter_eta"]
                    otherTsPhi=tsEv["barycenter_phi"]
                    otherTsEtaErr=tsEv["barycenter_etaError"]
                    otherTsPhiErr=tsEv["barycenter_phiError"]
                    otherTsX=tsEv["barycenter_x"]
                    otherTsY=tsEv["barycenter_y"]
                    otherTsZ=tsEv["barycenter_z"]
                    otherTsTime=tsEv["time"]
                    otherTsTimeErr=tsEv["timeError"]
                    otherTsRegEnergy=tsEv["regressed_energy"]

                    distance = distWrap_numba(refEta, refPhi, otherTsEta, otherTsPhi)
                    distancePull = distpull_numba(refEta, refEtaErr ,refPhi, refPhiErr, otherTsEta, otherTsEtaErr, otherTsPhi, otherTsPhiErr)
                    MTDvalue = mtdValue_numba(refx,refy,refz,refBeta,reftime,reftimeErr,otherTsX,otherTsY,otherTsZ,otherTsTime,otherTsTimeErr)

                    idx_sort = np.array(distance).argsort()
                    distance_sorted = distance[idx_sort]
                    distancePull_sorted = distancePull[idx_sort]
                    MTDvalue_sorted = MTDvalue[idx_sort]
                    tsRegEnergy_sorted = tsEv.regressed_energy[idx_sort]
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
                    if frac.size == 0:
                        radius = -1
                    else:
                        firstAboveThrIdx = np.argmax(frac>CUT)
                        radius=distance_sorted[firstAboveThrIdx]
                    denom = np.sum(tsEnergy_sorted[distance_sorted < radius])
                    if radius!=-1:
                        thr_distance_sorted= distance_sorted[distance_sorted > radius]
                        thr_distancePull_sorted= distancePull_sorted[distance_sorted > radius]
                        thr_tsRegEnergy_sorted= tsRegEnergy_sorted[distance_sorted > radius]
                        thr_MTDvalue_sorted= MTDvalue_sorted[distance_sorted > radius]
                    if denom > 0:
                        contamination = 1. - np.sum(sharedEnergy_sorted[(distance_sorted < radius) & (correctTrackMask)]) / denom
                    else:
                        contamination = -1
                    if radius != -1 and len(thr_MTDvalue_sorted) > 0:
                        print(" distance_sorted ", thr_distance_sorted)
                        print(" distancePull_sorted ", thr_distancePull_sorted)
                        print(" MTDvalue_sorted ", thr_MTDvalue_sorted)
                        print(" tsRegEnergy_sorted ", thr_tsRegEnergy_sorted)

                    '''
                    if radius!=-1:
                        dict_info["R"].append(radius)
                        dict_info["eta_trk"].append(abs(trks[ev]["track_hgcal_eta"][trackIdx]))
                        dict_info["phi_trk"].append(trks[ev]["track_hgcal_phi"][trackIdx])
                        dict_info["pt_trk"].append(trks[ev]["track_pt"][trackIdx])
                        dict_info["en_trk"].append(trks[ev]["track_p"][trackIdx])
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
                        
                        dict_info["En_0.005"].append(np.sum(tsRegEnergy_sorted[distance_sorted<0.005]))
                        dict_info["En_0.010"].append(np.sum(tsRegEnergy_sorted[distance_sorted<0.01]))
                        dict_info["En_0.030"].append(np.sum(tsRegEnergy_sorted[distance_sorted<0.03]))
                        dict_info["En_0.050"].append(np.sum(tsRegEnergy_sorted[distance_sorted<0.05]))
                        
                        dict_info["FirstR"].append(distance_sorted[0])
                        FirstE=tsEnergy_sorted[distance_sorted == distance_sorted[0]]
                        if len(FirstE) > 1 : raise Exception("2 first tracksters")
                        dict_info["FirstE"].append(tsEnergy_sorted[distance_sorted == distance_sorted[0]][0])
                        
            
    
    with open('BDT1_data.pkl', 'wb') as handle:
            pickle.dump(dict_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Job done on {eventcounter} events, result saved in BDT1_data.pkl " )
                    '''
