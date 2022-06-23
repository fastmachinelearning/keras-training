from __future__ import print_function
from root_numpy import root2array, tree2array
from root_numpy import testdata
import h5py
import ROOT as rt
from sklearn.utils import shuffle
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tree",
        action="store",
        type=str,
        dest="tree",
        default="t_allpar_new",
        help="tree name",
    )
    parser.add_argument(
        "-s",
        "--seed",
        action="store",
        type=int,
        dest="seed",
        default=None,
        help="seed",
    )
    parser.add_argument('fileNames', metavar='N', type=str, nargs='+',
                    help='file name to convert')
    args = parser.parse_args()

    for fileName in args.fileNames:
        print("converting %s -> %s" % (fileName, fileName.replace(".root", ".z")))
        # Convert a TTree in a ROOT file into a NumPy structured array
        arr = root2array(fileName, args.tree)
        # Shuffle array
        arr = shuffle(arr, random_state=args.seed)
        # open HDF5 file and write dataset
        h5File = h5py.File(fileName.replace(".root", ".z"), "w")
        h5File.create_dataset(args.tree, data=arr, compression="lzf")
        h5File.close()
