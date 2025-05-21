import awkward as ak
print(ak.__version__)

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
    dict_info={"logR" : [],
               "R" : [],
               "logP1R" : [],
               "contamination" : [],
               "MTDvalue" : [],
               "Rpull" : [],
               "eta_trk" : [],
               "phi_trk" : [],
               "pt_trk" : [],
               "etaErr" : [],
               "phiErr" : [],
               "deltaR_trk" : [],
               "etaphiCov" : [],
               "en_trk" : [],
               "time_mtd_trk" : [],
               "timeErr_mtd_trk" : [],
               "x_mtd_trk" : [],
               "y_mtd_trk" : [],
               "z_mtd_trk" : [],
               "beta_trk" : [],
               "nhits_trk": [],
               "outer_hits_trk" : [],
               "inner_hits_trk":[],
               "time_trk": [],
               "time_quality_trk": [],
               "hgcal_pt_trk":[],
               "hgcal_xyCov_trk": [],
               "hgcal_yErr_trk": [],
               "hgcal_xErr_trk": [],
               "hgcal_x_trk": [],
               "hgcal_y_trk": [],
               "hgcal_z_trk": [],
               "simEnergy" : [],
               "all_scores": [],
               "all_frac": [],
               }
    data_points = []    
    print("Searching for .root files in these folders: ", folderlist)
    eventcounter=0 
    max_score_reco2sim = 0.5
    min_shared_reco2sim = 0.8
    for directory in folderlist:
        print(f"Processing directory: {directory}")
            
        #check if folder exists
        if not os.path.isdir(directory):
            print(f"Directory {directory} does not exist. Skipping.")
            continue
        
        #for loop in files in directory
        for filename in tqdm(os.listdir(directory)):
            if ".sys" in filename : continue
            if filename.endswith(".root"):
                file_path = os.path.join(directory, filename)
            file=uproot.open(file_path)
            print(f"Processing file: {file_path}")
            try: #avoids some faulty .root files with no information contained in them
                alltracksters = file['ticlDumper/ticlTracksterLinks']
                allsimtrackstersCP = file['ticlDumper/simtrackstersCP']
                alltracks = file["ticlDumper/tracks"]
                allassociations = file['ticlDumper/associations']
                simts_SC = file['ticlDumper/simtrackstersSC']
                
                simTracksters = simts_SC.arrays(['raw_em_energy','raw_energy', 'regressed_energy', 'pdgID', 'NTracksters','NClusters'])
                tsCP=allsimtrackstersCP.arrays(["trackIdx","regressed_energy","regressed_pt"])
                tracksters = alltracksters.arrays(["barycenter_phi","barycenter_eta","raw_energy","time",
                                                   "timeError","regressed_energy",
                                                   #comment line below for ther runs 
                                                   "barycenter_etaError","barycenter_phiError",
                                                   'barycenter_x', 'barycenter_y', 'barycenter_z'])
                associations = allassociations.arrays(['ticlTracksterLinks_recoToSim_CP_sharedE',"ticlTracksterLinks_recoToSim_CP",
                                                       "ticlTracksterLinks_recoToSim_CP_score"])
            
                trks=alltracks.arrays(["track_hgcal_eta","track_hgcal_phi","track_pt","track_id",
                                       #comment line below for ther runs
                                       'track_hgcal_etaErr','track_hgcal_phiErr','track_hgcal_etaphiCov', 'track_p',
                                       'track_time_err','track_time_mtd','track_time_mtd_err',
                                       'track_pos_mtd','track_beta', 'track_pos_mtd.theVector.theX',
                                       'track_pos_mtd.theVector.theY','track_pos_mtd.theVector.theZ','track_quality',
                                       'track_missing_outer_hits','track_nhits', 'track_time_quality', 'track_time',
                                       'track_missing_inner_hits', 'track_hgcal_pt',
                                       #comment line below for ther runs
                                       'track_hgcal_xyCov', 'track_hgcal_yErr','track_hgcal_xErr',
                                       'track_hgcal_z', 'track_hgcal_y', 'track_hgcal_x'])

            except: 
                continue
            for ev in range(len(tsCP)):
                eventcounter+=1
                assEv=associations[ev]
                for i in range(len(tsCP[ev]["trackIdx"])):
                    #print(" simTracksters[ev].NTracksters ", simTracksters[ev].NTracksters)
                    if simTracksters[ev].NTracksters != 2 : continue
                    if tsCP[ev]["trackIdx"][i]==-1: continue
                    matches = np.where(trks[ev]["track_id"] == tsCP[ev]["trackIdx"][i])[0]
                    if len(matches) == 0:
                        continue
                    trackIdx = matches[0]
                    if trks[ev]["track_missing_outer_hits"][trackIdx] > 4 :
                        continue
                    #print(" simTracksters[ev].NTracksters ", simTracksters[ev].NTracksters)
                    simEnergy=tsCP[ev]["regressed_energy"][i]
                    simPt=tsCP[ev]["regressed_pt"][i]
                    refEta=trks[ev]["track_hgcal_eta"][trackIdx]
                    refPhi=trks[ev]["track_hgcal_phi"][trackIdx]
                    #comment line below for ther runs
                    refEtaErr=trks[ev]["track_hgcal_etaErr"][trackIdx]
                    refPhiErr=trks[ev]["track_hgcal_phiErr"][trackIdx]
                    #uncomment line below for ther runs                                                                                     
                    #refEtaErr=trks[ev]["track_hgcal_eta"][trackIdx]
                    #refPhiErr=trks[ev]["track_hgcal_phi"][trackIdx]
                    refx=trks[ev]["track_pos_mtd.theVector.theX"][trackIdx]
                    refy=trks[ev]["track_pos_mtd.theVector.theY"][trackIdx]
                    refz=trks[ev]["track_pos_mtd.theVector.theZ"][trackIdx]
                    refBeta=trks[ev]["track_beta"][trackIdx]
                    reftime=trks[ev]["track_time_mtd"][trackIdx]
                    reftimeErr=trks[ev]["track_time_mtd_err"][trackIdx]
                    #comment line below for ther runs
                    refEnergy=trks[ev]["track_p"][trackIdx]
                    #uncomment line below for ther runs
                    #refEnergy=trks[ev]["track_pt"][trackIdx]
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
                    #comment line below for ther runs
                    otherTsEtaErr=tsEv["barycenter_etaError"]
                    otherTsPhiErr=tsEv["barycenter_phiError"]
                    #uncomment line below for ther runs
                    #otherTsEtaErr=tsEv["barycenter_eta"]
                    #otherTsPhiErr=tsEv["barycenter_phi"]
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
                    assocIdxs_score_sorted = assEv["ticlTracksterLinks_recoToSim_CP_score"][idx_sort]
                    #correctTrackMask=(assocIdxs_sorted==i)
                    correctTrackMask = (assocIdxs_sorted == i) #& (assocIdxs_score_sorted < max_score_reco2sim) & ( (sharedEnergy_sorted/tsEnergy_sorted) > min_shared_reco2sim)
                    totalSharedE=np.sum(sharedEnergy_sorted[correctTrackMask])
                    scores = assocIdxs_score_sorted[correctTrackMask]
                    sharedEnergy_frac = sharedEnergy_sorted[correctTrackMask] / tsEnergy_sorted
                    distScan=np.linspace(0,0.7,100)
                    listE=[]
                    listContamination=[]
                    best_idx = None
                    #frac= np.cumsum(flatten_numba(sharedEnergy_sorted[(correctTrackMask)]))/totalSharedE
                    #contamination= 1.-np.cumsum(sharedEnergy_sorted[correctTrackMaskrackMask])
                    # Precompute cumulative sums
                    # Cumulative sums
                    '''
                    frac= np.cumsum(flatten_numba(sharedEnergy_sorted[(correctTrackMask)]))/totalSharedE
                    #contamination= 1.-np.cumsum(sharedEnergy_sorted[correctTrackMaskrackMask])
                    if frac.size == 0:
                        radius = -1
                        mtdvalue = -1
                        Rpull = -1
                    else:
                        firstAboveThrIdx = np.argmax(frac>=CUT)
                        scores_within_radius = scores[:firstAboveThrIdx + 1]
                        sharedEnergy_within_frac_radius = sharedEnergy_frac[:firstAboveThrIdx + 1]
                        #full_radius_points=distance_sorted[:firstAboveThrIdx + 1]
                        radius=distance_sorted[firstAboveThrIdx]
                        mtdvalue=MTDvalue_sorted[firstAboveThrIdx]
                        Rpull=distancePull_sorted[firstAboveThrIdx]
                        flat_radii = np.array([x for sublist in distance_sorted[:firstAboveThrIdx + 1] for x in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])
                        flat_scores = np.array([x for sublist in scores_within_radius[:firstAboveThrIdx + 1] for x in sublist])
                        #print(" flat_radii ", flat_radii)
                        #print(" flat_scores ", flat_scores)
                        flat_Rpull = np.array(distancePull_sorted[:firstAboveThrIdx + 1])
                        flat_mtdvalue = np.array(MTDvalue_sorted[:firstAboveThrIdx + 1])
                        flat_sharedEnergy = np.array([x for sublist in sharedEnergy_frac[:firstAboveThrIdx + 1] for x in sublist])
                    denom = np.sum(tsEnergy_sorted[distance_sorted < radius])
                    if denom > 0:
                        contamination = 1. - np.sum(sharedEnergy_sorted[(distance_sorted < radius) & (correctTrackMask)]) / denom
                    else:
                        contamination = -1
                    if radius!=-1 and mtdvalue!= -1 and Rpull!= -1 and contamination!= -1 :
                        flat_scores = np.array([x for sublist in scores_within_radius if len(sublist) > 0 for x in sublist])
                        flat_frac = np.array([x for sublist in sharedEnergy_within_frac_radius if len(sublist) > 0 for x in sublist])
                        entry = [flat_frac , flat_scores, trks[ev]["track_pt"][trackIdx], radius, contamination]
                        data_points.append(entry)
                        '''
                    track_pt = trks[ev]["track_pt"][trackIdx]

                    frac = np.cumsum(flatten_numba(sharedEnergy_sorted[correctTrackMask])) / totalSharedE

                    # Edge case: No valid track
                    if frac.size == 0:
                        radius = -1
                        mtdvalue = -1
                        Rpull = -1
                        contamination = -1
                    else:
                        found = False
                        for idx in np.where(frac >= CUT)[0]:
                            r = distance_sorted[idx]
                            in_radius_mask = distance_sorted < r

                            denom = np.sum(tsEnergy_sorted[in_radius_mask])
                            if denom > 0:
                                cont = 1. - np.sum(sharedEnergy_sorted[in_radius_mask & correctTrackMask]) / denom
                            else:
                                cont = -1

                            # Apply contamination cut only if track_pt < 40
                            if (track_pt < 40 and 0 <= cont < 0.2) or (track_pt >= 40):
                                radius = r
                                mtdvalue = MTDvalue_sorted[idx]
                                Rpull = distancePull_sorted[idx]
                                contamination = cont

                                # Now flatten arrays only within this radius
                                scores_within_radius = scores[:idx + 1]
                                sharedEnergy_within_frac_radius = sharedEnergy_frac[:idx + 1]
                                
                                flat_radii = np.array([x for sublist in distance_sorted[:idx + 1] for x in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])
                                flat_scores = np.array([x for sublist in scores_within_radius if len(sublist) > 0 for x in sublist])
                                flat_Rpull = np.array(distancePull_sorted[:idx + 1])
                                flat_mtdvalue = np.array(MTDvalue_sorted[:idx + 1])
                                flat_sharedEnergy = np.array([x for sublist in sharedEnergy_frac[:idx + 1] if len(sublist) > 0 for x in sublist])

                                found = True
                                break
                            
                        if not found:
                            radius = -1
                            mtdvalue = -1
                            Rpull = -1
                            contamination = -1

                        # Store result only if valid
                    if radius != -1 and mtdvalue != -1 and Rpull != -1 and contamination != -1:
                        entry = [flat_sharedEnergy, flat_scores, track_pt, radius, contamination]
                        data_points.append(entry)

                        #print(" entry ", entry)
                        #for scores in flat_scores:
                        dict_info["all_scores"].append(flat_scores)
                        #for frac in flat_frac:
                        #print(" flat_scores ", flat_scores)
                        dict_info["all_frac"].append(flat_sharedEnergy)
                        #print(" flat_sharedEnergy ", flat_sharedEnergy)
                        dict_info["logP1R"].append(np.log1p(radius))
                        #print(" radius ", radius)
                        dict_info["R"].append(radius)
                        dict_info["logR"].append(np.log(radius))
                        dict_info["contamination"].append(contamination)
                        dict_info["MTDvalue"].append(mtdvalue)
                        dict_info["Rpull"].append(Rpull)
                        dict_info["simEnergy"].append(simEnergy)
                        dict_info["eta_trk"].append(abs(trks[ev]["track_hgcal_eta"][trackIdx]))
                        dict_info["phi_trk"].append(trks[ev]["track_hgcal_phi"][trackIdx])
                        dict_info["pt_trk"].append(trks[ev]["track_pt"][trackIdx])
                        #comment line below for ther runs
                        dict_info["en_trk"].append(trks[ev]["track_p"][trackIdx])
                        #uncomment line below for ther runs
                        #dict_info["en_trk"].append(trks[ev]["track_pt"][trackIdx])
                        dict_info["time_mtd_trk"].append(trks[ev]["track_time_mtd"][trackIdx])
                        dict_info["timeErr_mtd_trk"].append(trks[ev]["track_time_mtd_err"][trackIdx])
                        dict_info["x_mtd_trk"].append(trks[ev]["track_pos_mtd.theVector.theX"][trackIdx])
                        dict_info["y_mtd_trk"].append(trks[ev]["track_pos_mtd.theVector.theY"][trackIdx])
                        dict_info["z_mtd_trk"].append(trks[ev]["track_pos_mtd.theVector.theZ"][trackIdx])
                        #comment line below for ther runs
                        dict_info["etaErr"].append(trks[ev]["track_hgcal_etaErr"][trackIdx]*1.5)
                        dict_info["phiErr"].append(trks[ev]["track_hgcal_phiErr"][trackIdx]*1.5)
                        #uncomment line below for ther runs
                        #dict_info["etaErr"].append(trks[ev]["track_hgcal_eta"][trackIdx]*1.5)
                        #dict_info["phiErr"].append(trks[ev]["track_hgcal_phi"][trackIdx]*1.5)
                        #comment line below for ther runs
                        dict_info["deltaR_trk"].append(math.sqrt((trks[ev]["track_hgcal_phiErr"][trackIdx]*1.5)**2+ (trks[ev]["track_hgcal_etaErr"][trackIdx]*1.5)**2))
                        #uncomment line below for ther runs
                        #dict_info["deltaR_trk"].append(math.sqrt((trks[ev]["track_hgcal_phi"][trackIdx]*1.5)**2+ (trks[ev]["track_hgcal_eta"][trackIdx]*1.5)**2))
                        dict_info["etaphiCov"].append(trks[ev]["track_hgcal_etaphiCov"][trackIdx]*1.5*1.5)
                        #dict_info["etaphiCov"].append(trks[ev]["track_hgcal_eta"][trackIdx]*1.5*1.5)

                        dict_info["beta_trk"].append(trks[ev]["track_beta"][trackIdx])
                        dict_info["nhits_trk"].append(trks[ev]["track_nhits"][trackIdx])
                        dict_info["outer_hits_trk"].append(trks[ev]["track_missing_outer_hits"][trackIdx])
                        dict_info["inner_hits_trk"].append(trks[ev]["track_missing_inner_hits"][trackIdx])
                        dict_info["time_trk"].append(trks[ev]["track_time"][trackIdx])
                        dict_info["time_quality_trk"].append(trks[ev]["track_time_quality"][trackIdx])
                        dict_info["hgcal_pt_trk"].append(trks[ev]["track_hgcal_pt"][trackIdx])
                        #comment line below for ther runs
                        dict_info["hgcal_xyCov_trk"].append(trks[ev]["track_hgcal_xyCov"][trackIdx])
                        dict_info["hgcal_yErr_trk"].append(trks[ev]["track_hgcal_yErr"][trackIdx])
                        dict_info["hgcal_xErr_trk"].append(trks[ev]["track_hgcal_xErr"][trackIdx])
                        #uncomment line below for ther runs
                        #dict_info["hgcal_xyCov_trk"].append(trks[ev]["track_hgcal_x"][trackIdx])
                        #dict_info["hgcal_yErr_trk"].append(trks[ev]["track_hgcal_y"][trackIdx])
                        #dict_info["hgcal_xErr_trk"].append(trks[ev]["track_hgcal_x"][trackIdx])

                        dict_info["hgcal_x_trk"].append(trks[ev]["track_hgcal_x"][trackIdx])
                        dict_info["hgcal_y_trk"].append(trks[ev]["track_hgcal_y"][trackIdx])
                        dict_info["hgcal_z_trk"].append(trks[ev]["track_hgcal_z"][trackIdx])
    
    with open('BDT1_data.pkl', 'wb') as handle:
            pickle.dump(dict_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Job done on {eventcounter} events, result saved in BDT1_data.pkl " )
                    
    with open("track_score_data.pkl", "wb") as f:
        pickle.dump(data_points, f)
