import os
import h5py
import numpy as np
import pandas as pd
import math
import sys
import itertools
#import setGPU

minVal = int(sys.argv[1])
maxVal = int(sys.argv[2])

#f = h5py.File("data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth.z",'r')
f = h5py.File("/eos/project/d/dshep/hls-fml/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth.z", "r")
d = f['t_allpar_new'][()]
features = d.dtype.names

# Convert to dataframe
features_df = pd.DataFrame(d,columns=features)

jet_features = []
for feature in features:
    if feature.find("j_") != -1: jet_features.append(feature)

evt = features_df.iloc[0]
print(evt[['j_pt','j1_px','j_g', 'j_q', 'j_w', 'j_z', 'j_t', 'j_undef']])

featuresConst = ['j1_px', 'j1_py', 'j1_pz', 'j1_e', 'j1_erel', 'j1_pt', 'j1_ptrel', 'j1_eta', 'j1_etarel', 'j1_etarot', 'j1_phi', 'j1_phirel', 'j1_phirot', 'j1_deltaR', 'j1_costheta', 'j1_costhetarel']

nX = 224
nY = 224
nConstituents = 188
#nConstituents = 50
jetR = 1.0
# datasets to store
jets = np.array([])
jetImage = np.array([])
jetImageRotated = np.array([])
# charged hadrons
thisJetImage = np.zeros([1,nX,nY])
thisJetImageRotated = np.zeros([1,nX,nY])
jetPt = -9999.
#jet4MOM = rt.TLorentzVector()
iConstituents = 0
iConstituentsMax = 0
firstEvt = True
#for i in range(d.shape[0]):
for i in range(minVal,maxVal):
    if i >= d.shape[0]: continue
    evt = features_df.iloc[i]
    #print("NEW PARTICLE")
    if evt['j_pt'] != jetPt and jetPt != -9999.:
        #print("ENTERED")
        # new jet
        jetPt = evt['j_pt']
        # save the jet image collected so far
        jets = np.concatenate((jets, myJetArray), axis=0) if jets.size else myJetArray
        jetImage = np.concatenate((jetImage,thisJetImage), axis=0) if jetImage.size else thisJetImage
        jetImageRotated = np.concatenate((jetImageRotated,thisJetImageRotated), axis=0) if jetImageRotated.size else thisJetImageRotated
        #
        # re-initialize the jet images  
        thisJetImage = np.zeros([1,nX,nY])
        thisJetImageRotated = np.zeros([1,nX,nY])
        iConstituentsMax = max(iConstituents, iConstituentsMax)
        iConstituents = 0
    #min pT cut at 50 MeV
    if firstEvt:
        jetPt = evt['j_pt']
        firstEvt = False
    myJetArray = np.array(evt[jet_features])
    iEta = int(evt['j1_etarel']/jetR*nX + nX/2.)
    iPhi = int(evt['j1_phirel']/jetR*nY + nY/2.)
    iEta = int(min(iEta, nX-1))
    iEta = int(max(iEta, 0.))
    iPhi = int(min(iPhi, nY-1))
    iPhi = int(max(iPhi, 0.))
    if evt['j1_pt'] < 0.05: continue 
    thisJetImage[0,iEta,iPhi] += evt['j1_pt']
    ##### now rotated
    iEta = int(evt['j1_etarot']/jetR*nX + nX/2.)
    iPhi = int(evt['j1_phirot']/jetR*nY + nY/2.)
    iEta = int(min(iEta, nX-1))
    iEta = int(max(iEta, 0.))
    iPhi = int(min(iPhi, nY-1))
    iPhi = int(max(iPhi, 0.))
    if evt['j1_pt'] < 0.05: continue
    thisJetImageRotated[0,iEta,iPhi] += evt['j1_pt']
    iConstituents +=1
jets = jets.reshape((jetImage.shape[0],len(jet_features)))
#
fOUT = h5py.File("/eos/project/d/dshep/hls-fml/jetImage_1evt_%i_%i.h5" %(minVal, maxVal), "w")
fOUT.create_dataset("jetImage", data=jetImage, compression='gzip')
fOUT.create_dataset("jetImageRotated", data=jetImageRotated, compression='gzip')
fOUT.create_dataset("jets", data=jets, compression='gzip')
dt = h5py.special_dtype(vlen=str)
dset = fOUT.create_dataset('jetFeatureNames', (len(jet_features),), dtype=dt)
for i in range(len(jet_features)): dset[i] = jet_features[i]
fOUT.close()

