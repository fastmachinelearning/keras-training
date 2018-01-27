import ROOT as rt
from array import array
import random
import os
from optparse import OptionParser
import sys

def getTree(myTree, oldTree, listOfBranches):
    
    rando = random.randint(1,999999)
    # first structure
    stringMyStruct1 = "struct MyStruct1{"
    for branch in listOfBranches:
        if oldTree.GetBranchStatus(branch.GetName()):
            stringMyStruct1 += "float %s;"%(branch.GetName())
    stringMyStruct1 += "};"
    tempMacro = open("tempMacro_%d.C"%rando,"w")
    tempMacro.write(stringMyStruct1+"};}")
    tempMacro.close()
    rt.gROOT.ProcessLine(stringMyStruct1)
    from ROOT import MyStruct1

    # fills the bins list and associate the bins to the corresponding variables in the structure
    s1 = MyStruct1()
    for branch in listOfBranches:
        if oldTree.GetBranchStatus(branch.GetName()):
            print branch.GetName(), branch.GetClassName()
            myTree.Branch(branch.GetName(), rt.AddressOf(s1,branch.GetName()),'%s/F'%branch.GetName())
    os.system("rm tempMacro_%d.C"%rando)
    return s1


def addLeaves(tree,fileName):
    events = tree.GetEntries()
    leaves = ["j_g/I","j_q/I","j_w/I","j_z/I","j_t/I","j_undef/I"]
    leafValues = [array("I", [0]),array("I", [0]),array("I", [0]),array("I", [0]),array("I", [0]),array("I", [0])]
    newfile = rt.TFile.Open(fileName.replace('.root','_truth.root'),'RECREATE')

    particleBranches = [branch.GetName() for branch in tree.GetListOfBranches() if 'j1_' in branch.GetName()]

    tree.SetBranchStatus("*",1)
    # remove these branches
    tree.SetBranchStatus("njets",0)
    tree.SetBranchStatus("j_passCut",0)
    tree.SetBranchStatus("j_tau21_b1",0)
    tree.SetBranchStatus("j_tau21_b2",0)
    tree.SetBranchStatus("j_tau21_b1_mmdt",0)
    tree.SetBranchStatus("j_tau21_b2_mmdt",0)
    
    newtree = rt.TTree(tree.GetName()+'_new',tree.GetName()+'_new')
    s1 = getTree(newtree,tree,tree.GetListOfBranches())

    for j in range(len(leaves)):
        newBranch = newtree.Branch( leaves[j].split('/')[0] , leafValues[j], leaves[j])
    for i in range(events):
        tree.GetEntry(i)
        isNaN = False
        if rt.TMath.IsNaN(tree.j_n2_b1_mmdt[0]) or rt.TMath.IsNaN(tree.j_n2_b1[0]) or rt.TMath.IsNaN(tree.j_tau32_b2_mmdt[0]) or rt.TMath.IsNaN(tree.j_tau32_b2[0]):
            isNaN = True
        if isNaN: continue
                
        for j in range(len(leaves)):
            leafValues[j][0] = 0
        if 'gg' in fileName:
            leafValues[0][0] = 1
        elif 'qq' in fileName:
            leafValues[1][0] = 1
        elif 'WW' in fileName:
            leafValues[2][0] = 1
        elif 'ZZ' in fileName:
            leafValues[3][0] = 1
        elif 'tt' in fileName:
            leafValues[4][0] = 1
        else:
            leafValues[5][0] = 1

        for branch in tree.GetListOfBranches():
            if branch.GetName() in particleBranches: continue
            if tree.GetBranchStatus(branch.GetName()):
                obj = getattr(tree, branch.GetName())
                if hasattr(obj, "__getitem__"):
                    setattr(s1, branch.GetName(), obj[0] )
                else:
                    setattr(s1, branch.GetName(), obj )

        if len(particleBranches)>0:
            nParticles = len(getattr(tree, particleBranches[0]))
            particleObj = [getattr(tree, branchName) for branchName in particleBranches]
            for i_particle in range(0, nParticles):
                for branchName, obj in zip(particleBranches,particleObj):
                    setattr(s1, branchName, obj[i_particle])
                newtree.Fill()
        else:
            newtree.Fill()
    
        if i % 3000 == 0:
            print "%s of %s: %s" % (i,events,leafValues)
    newtree.Write()
    print "Saved tree with %s events . . ." % ( newtree.GetEntries() )
    newfile.Close()
    del newfile
    #end of AddLeaves()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar', help='tree name')
    (options,args) = parser.parse_args()
    
    for fileName in args:
        tFile = rt.TFile.Open(fileName)
        tree = tFile.Get(options.tree)
        addLeaves(tree, fileName)
        tFile.Close()
        del tFile
