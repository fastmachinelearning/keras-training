import h5py
import numpy as np
import pandas as pd

# import setGPU

f = h5py.File(
    "/eos/project/d/dshep/hls-fml/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth.z",
    "r",
)

d = f["t_allpar_new"][()]
features = d.dtype.names
# Convert to dataframe
features_df = pd.DataFrame(d, columns=features)
jetPt = -9999
boundaries = np.array([])

for i in range(d.shape[0]):
    evt = features_df.iloc[i]
    if evt["j_pt"] != jetPt:
        boundaries = (
            np.concatenate([boundaries, np.array([i])], axis=0)
            if boundaries.size
            else np.array([i])
        )
        jetPt = evt["j_pt"]

iJet = 0

while iJet < len(boundaries) - 1:
    next_iJet = min(len(boundaries) - 1, iJet + 1)
    minVal = boundaries[iJet]
    maxVal = boundaries[next_iJet]
    print(minVal, maxVal)
    iJet = next_iJet
