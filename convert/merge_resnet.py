import glob
import sys
import h5py
import numpy as np
import os

keys = ["jetFeatureNames", "jetImage", "jetImageRotated", "jets"]

jetImage = np.array([])
jetImageRotated = np.array([])
jets = np.array([])

# we only need one of these
jetFeatureNames = np.array([])

fileIn = open(sys.argv[1], "r")
fileList = fileIn.readlines()

f = h5py.File(fileList[0][:-1], "r")
jetFeatureNames = f.get("jetFeatureNames")
f.close

for fileIN in fileList:
    fileIN = fileIN[:-1]
    print(fileIN)
    if not os.path.isfile(fileIN):
        continue
    f = h5py.File(fileIN, "r")
    print(list(f.keys()))
    x = np.array(f.get("jetImage"))
    jetImage = np.concatenate([jetImage, x], axis=0) if jetImage.size else x
    x = np.array(f.get("jetImageRotated"))
    jetImageRotated = (
        np.concatenate([jetImageRotated, x], axis=0) if jetImageRotated.size else x
    )
    x = np.array(f.get("jets"))
    jets = np.concatenate([jets, x], axis=0) if jets.size else x

    print(jetImage.shape)
    f.close()

fOUT = h5py.File(sys.argv[2], "w")
fOUT.create_dataset("jetImage", data=jetImage, compression="gzip")
fOUT.create_dataset("jetImageRotated", data=jetImageRotated, compression="gzip")
fOUT.create_dataset("jets", data=jets, compression="gzip")
fOUT.create_dataset("jetFeatureNames", data=jetFeatureNames, compression="gzip")
fOUT.close()
