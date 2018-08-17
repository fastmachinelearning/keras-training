from __future__ import print_function
from root_numpy import root2array, tree2array
from root_numpy import testdata
#from rootpy.root2hdf5 import root2hdf5
import h5py
import ROOT as rt
from sklearn.utils import shuffle
from optparse import OptionParser

if __name__ == "__main__":
    
    parser = OptionParser()
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-s','--seed'   ,action='store',type='int',dest='seed'   ,default=None, help='seed')
    (options,args) = parser.parse_args()
    
    for fileName in args:
        print('converting %s -> %s'%(fileName, fileName.replace('.root','.z')))
        # Convert a TTree in a ROOT file into a NumPy structured array
        arr = root2array(fileName, options.tree)
        # Shuffle array
        arr = shuffle(arr, random_state=options.seed)
        # open HDF5 file and write dataset
        h5File = h5py.File(fileName.replace('.root','.z'),'w')        
        h5File.create_dataset(options.tree, data=arr,  compression='lzf')
        h5File.close()
        del h5File
        #root2hdf5(fileName, fileName.replace('.root','.h5'))
