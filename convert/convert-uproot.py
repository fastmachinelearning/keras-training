import uproot
import pandas
import numpy as np
import pandas as pd
import h5py

if __name__ == "__main__":
    
    parser = OptionParser()
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-s','--seed'   ,action='store',type='int',dest='seed'   ,default=None, help='seed')
    parser.add_option('--split', action='store',type='int',dest='split'   ,default=10, help='split')
    (options,args) = parser.parse_args()

    np.random.seed(options.seed)
    for fileName in args:
        upfile = uproot.open(fileName)
        tree = upfile[options.tree]

        df = tree.pandas.df()
        df = df.sample(frac=1)
        #df = pd.concat([g.sort_values('j1_pt',axis=0,ascending=False) for _, g in df.groupby(['j_index'], sort=False, as_index=False)], ignore_index=True)
        dfArrays = [g.sort_values('j1_pt',axis=0,ascending=False) for _, g in df.groupby(['j_index'], sort=False, as_index=False)]
        
        a = np.array(range(len(dfArrays)))
        a = np.random.permutation(a)
        for isplit, indices in enumerate(np.array_split(a, options.split)):
            df_split = pd.concat([dfArrays[i] for i in indices],ignore_index=True)
            arr = np.array(df_split.to_records())
            print(arr.shape)   
            h5File = h5py.File(infile.replace('.root','_%i.z'%isplit),'w')
            h5File.create_dataset(optoins.tree, data=arr,  compression='lzf')
            h5File.close()

