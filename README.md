** BDT to compute linking radius and PU contamination in order to link tracksters in HGCAL with tracks reconstructed in the CMS tracker **
Look at updated documentation with formatting, in drive, [here](https://docs.google.com/document/d/1qcI5ugK9bQdd3UMzZesJZUQN1z4YZ2bKjOm6277GlgE/edit?usp=sharing)
# Documentation for the training of the BDT for track-trackster linking in HGCAL

### Objective: optimizing the linking performance of tracks with tracksters in HGCAL..


Given a reconstructed track in the endcap region, the aim of this BDT is 
To compute the optimal radius around which tracksters can be linked
To give an estimate of the PU contamination inside the linking radius


In order to do this, the given code uses as input data which is SinglePion samples with PU = 200 . But in principle other samples can be used. 
More specifically, the input are .root files with trees that are dumped by the TICLDumper.cc plugin, with some minor modifications in order to add the track position uncertainty to the dumper (thanks Mark for the help) .
The commit of the code is the following: [commit](https://github.com/cms-sw/cmssw/compare/master...Tizianop6:cmssw:trackPosUncertainty_TICLDumper). 


## How to generate the samples: 

Workflow used to generate the config files. 
> runTheMatrix.py -w upgrade -l 29688.203 -j 0

Some modifications:

Modify  step1 parameter in order to set the pT range (also eta if needed) of the generated sample samples. Used range: 5-100 Gev . Eta left as default 1.7-2.7
Starting from the command  cmsDriver.py which is written inside the steps 2 ,3 ,4 in order to add pileup add the following at the end of the command --pileup AVE_200_BX_25ns --pileup_input das:/NAME OF THE DATASET
For the PU dataset look at the MinBias samples for a given CMSSW distribution inside DAS.

For instance, these are the commands that I’ve used:


> cmsDriver.py step2 -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:@relvalRun4 --conditions auto:phase2_realistic_T33 --datatier GEN-SIM-DIGI-RAW -n 10 --eventcontent FEVTDEBUGHLT --geometry ExtendedRun4D110 --era Phase2C17I13M9 --procModifiers ticl_v5 --no_exec --filein file:step1.root --fileout file:step2.root --pileup AVE_200_BX_25ns --pileup_input das:/RelValMinBias_14TeV/CMSSW_15_0_0_pre3-141X_mcRun4_realistic_v3_STD_MinBias_Run4D110_GenSim-v1/GEN-SIM

> cmsDriver.py step3 -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --conditions auto:phase2_realistic_T33 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM --geometry ExtendedRun4D110 --era Phase2C17I13M9 --procModifiers ticl_v5 --no_exec --filein file:step2.root --fileout file:step3.root --pileup AVE_200_BX_25ns --pileup_input das:/RelValMinBias_14TeV/CMSSW_15_0_0_pre3-141X_mcRun4_realistic_v3_STD_MinBias_Run4D110_GenSim-v1/GEN-SIM

> cmsDriver.py step4 -s HARVESTING:@phase2Validation+@phase2+@miniAODValidation+@miniAODDQM --conditions auto:phase2_realistic_T33 --mc --geometry ExtendedRun4D110 --scenario pp --filetype DQM --era Phase2C17I13M9 --procModifiers ticl_v5 -n 100 --no_exec --filein file:step3_inDQM.root --fileout file:step4.root --pileup AVE_200_BX_25ns --pileup_input das:/RelValMinBias_14TeV/CMSSW_15_0_0_pre3-141X_mcRun4_realistic_v3_STD_MinBias_Run4D110_GenSim-v1/GEN-SIM



In order to activate the TICLDumper: add the following lines at the end of the step3
> from RecoHGCal.TICL.customiseTICLFromReco import customiseTICLForDumper
> process = customiseTICLForDumper(process)




## How to run the dataset extraction and BDT training

The github repository contains the code to carry out the BDT training, the script that needs to be run is make_BDT.py: one can use python3 make_BDT.py –help the see all the possible arguments that can be passed to the script. 
In order to carry out the training one must
Load the samples (by default in the “./histos/” folder) 
Analyze the samples to construct the dataset
If needed: plot the input feature distribution
Carry out hyperparameter tuining
Train the model and assess performance



> BDT regression for trackster association analysis
>
>optional arguments:
>  -h, --help        	show this help message and exit
>  --do_all          	Do all the work: load data, make plots, do bayesian search and train the model
>  --load_data       	(Re-)process .root files and save the dataset to a pickle file (BDT1_data.pkl)
>  --make_plots      	Make and save histograms of input features (saved in ./featureplots/)
>  --bayesiansearch  	Perform Bayesian hyperparameter search using training data
>  --train_validate  	Train final XGBoost model and make plots of residuals and SHAP values
>  -j N_JOBS, --n_jobs N_JOBS
>                    	Number of jobs to use for bayesian search (default: -1, same as CPU cores)
>  --n_iterations N_ITERATIONS
>                    	Number of iterations for Bayesian search (default: 50)
>  --folderlist FOLDERLIST [FOLDERLIST ...]
>                    	List of input folders with step3 ROOT files (default: ['./histos/'])

For instance, assuming I have two directories with the samples and I want to run all the pipeline on 12 threads, a usage of the script can be the following:

python3 –do_all –folderlist ./histos/ ./histos_2/ -j 12 



## Brief outline of the code
The main ideas of the code are the following:
<ol><li>Load the trees coming the TICLDumper, in a given directory</li>
<li>Consider the CaloParticles in the event (considered only CP from signal vertex, no PU CPs) </li>
<li>Find the track associated to the CP</li>
<li>Then look at all the simTrackstersCP which are associated to that track</li>
<li>Take all the simTrackstersCP and order them in ascending order in distance from the track position (in eta-phi) at the HGCAL entrance. The distance considered is the Delta R distance in eta-phi. </li>
<li>Look at the associator ticlTracksterLinks_recoToSim_CP_sharedE , and sum  the shared energy values between the ticlTracksterLinks and the CaloParticle. Define: R as the DeltaR value in which 80% of the total ticlTracksterLinks_recoToSim_CP_sharedE for that CaloParticle lies inside. </li>
<li>The energy must come from tracksterLinks which share a non zero amount of energy with the CP, not counting contamination. </li>
<li>Example to be clear: if I have a CaloParticle of 100 GeV, the radius R is the radius in which I reconstruct not necessarily 80 GeV from tracksters inside the radius, but it could be 87 GeV, 80 coming from the considered CP and 7 from contamination.  A more detailed decision whether to associate the single trackster inside the radius or not will be carried out using another technique. </li>
<li>The PU (or in general tracksters coming from other CPs) contamination is defined as 1-(reconstructed trackster energy shared with the CP inside the radius)/(total reconstructed trackster energy inside the radius) </li>
<li>Other features, which are more straightforward to compute: </li>
<ul><li>Track eta, phi, pT uncertainty on eta, phi and covariance of the two, pT uncertainty</li>
<li>Energy and DeltaR distance of the closest trackster to the track</li>
<li>Number of tracksters and sum of their energy inside DeltaR radiuses =0.005, 0.010, 0.030, 0.050</li>
<li>Other features can be added eventually (like, DeltaR error of track as combination of error on phi and eta is already there, need to check correlation between phi and eta and see whether to include it or not) </li>
</ul>
<li>Training of the network: </li>
<ul><li>Splitting of the dataset in train and test set (80/20%)</li>
<li>The idea is to use a Boosted Decision Tree model whose hyperparameters are tuned using a Bayesian search</li>
<li>The loss function is taken to be the Mean Absolute Error, reasonable for the regression task, and also because the Mean Squared Error is prone to the presence of outliers, leading to the fact that by changing the dataset splitting seed the loss of training and test set changes considerably</li>
<li>The number of iterations of the bayesian search can be tuned with command line argument “–n_iters NUMBER_ITERATIONS”</li>
<li>Hyperparameter tuning carried out on the train set, with cross validation (5-fold, but can be modified)</li>
<li>After the tuning, the tuned model is trained on the train dataset and its performance is assessed on the test dataset</li>
<li>The plots displaying the performance on the test dataset are reported in the “plots” folder they show the loss values of train and test dataset as a function of </li>boosting round, the SHAP values of the two output of the BDT and the residuals (predicted-truevalue) of PU_contamination and linking R</li>
</ul>
</ol>
# Dependencies

Python modules: 

> scikit-optimize (used scikit_optimize-0.10.2)
> shap
> XGboost 2.1.3 (some previous versions do not support dual output with MAE loss function)

Can do a requirements.txt file with this content:

> xgboost==2.1.3
> scikit-learn==1.3.2

And then install with the following command

> pip install -r requirements.txt


