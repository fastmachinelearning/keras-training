import uproot
import pandas
import numpy as np
import pandas as pd
import h5py
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
    parser.add_argument(
        "--split", action="store", type=int, dest="split", default=10, help="split"
    )
    parser.add_argument('fileNames', metavar='N', type=str, nargs='+',
                    help='file name to convert')
    args = parser.parse_args()

    np.random.seed(args.seed)
    for fileName in args.fileNames:
        upfile = uproot.open(fileName)
        tree = upfile[args.tree]

        df = tree.arrays(library='pd')
        df = df.sample(frac=1)
        dfArrays = [
            g.sort_values("j1_pt", axis=0, ascending=False)
            for _, g in df.groupby(["j_index"], sort=False, as_index=False)
        ]

        a = np.array(range(len(dfArrays)))
        a = np.random.permutation(a)
        for isplit, indices in enumerate(np.array_split(a, args.split)):
            df_split = pd.concat([dfArrays[i] for i in indices], ignore_index=True)
            arr = np.array(df_split.to_records())
            h5File = h5py.File(fileName.replace(".root", "_%i.z" % isplit), "w")
            h5File.create_dataset(args.tree, data=arr, compression="lzf")
            h5File.close()
