import ROOT as rt
from array import array
import random
import os
from optparse import OptionParser
import sys
import numpy as np

def getTree(myTree, oldTree, listOfBranches, additionalBranches = []):
    
    rando = random.randint(1,999999)
    # first structure
    stringMyStruct1 = "struct MyStruct1{"
    for branch in listOfBranches:
        if oldTree.GetBranchStatus(branch.GetName()):
            stringMyStruct1 += "float %s;"%(branch.GetName())
    for branchName in additionalBranches:
        stringMyStruct1 += "float %s;"%(branchName)
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
    for branchName in additionalBranches:
        print branchName
        myTree.Branch(branchName, rt.AddressOf(s1,branchName),'%s/F'%branchName)
    os.system("rm tempMacro_%d.C"%rando)
    return s1


def addLeaves(tree,fileName):
    events = tree.GetEntries()
    leaves = ["j_g/I","j_q/I","j_w/I","j_z/I","j_t/I","j_undef/I"]
    leafValues = [array("I", [0]),array("I", [0]),array("I", [0]),array("I", [0]),array("I", [0]),array("I", [0])]
    newfile = rt.TFile.Open(fileName.replace('.root','_truth.root'),'RECREATE')

    particleBranches = [branch.GetName() for branch in tree.GetListOfBranches() if 'j1_' in branch.GetName()]
    additionalBranches = []
    if len(particleBranches)>0:
        additionalBranches = ['j1_erel','j1_pt','j1_ptrel','j1_eta','j1_etarel','j1_phi','j1_phirel','j1_deltaR']

    tree.SetBranchStatus("*",1)
    # remove these branches
    tree.SetBranchStatus("njets",0)
    tree.SetBranchStatus("j_passCut",0)
    tree.SetBranchStatus("j_tau21_b1",0)
    tree.SetBranchStatus("j_tau21_b2",0)
    tree.SetBranchStatus("j_tau21_b1_mmdt",0)
    tree.SetBranchStatus("j_tau21_b2_mmdt",0)
    
    newtree = rt.TTree(tree.GetName()+'_new',tree.GetName()+'_new')
    s1 = getTree(newtree,tree,tree.GetListOfBranches(), additionalBranches=additionalBranches)

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
            particlePt, particleEta, particlePhi, particleE = [], [], [], []
            jet = rt.TLorentzVector()
            for i_particle in range(0, nParticles):
                particle = rt.TLorentzVector(tree.j1_px[i_particle], tree.j1_py[i_particle], tree.j1_pz[i_particle], tree.j1_e[i_particle])
                jet += particle
                particlePt.append( particle.Pt() )
                particleEta.append( particle.Eta() )
                particlePhi.append( particle.Phi() )
                particleE.append( particle.E() )
            for i_particle in np.argsort([-pt for pt in particlePt]):
                particle = rt.TLorentzVector(tree.j1_px[i_particle], tree.j1_py[i_particle], tree.j1_pz[i_particle], tree.j1_e[i_particle])
                #if particlePt[i_particle] < 1: continue
                for branchName, obj in zip(particleBranches,particleObj):
                    setattr(s1, branchName, obj[i_particle])
                setattr(s1, 'j1_erel', float(particleE[i_particle]/jet.E()))
                setattr(s1, 'j1_pt', float(particlePt[i_particle]))
                setattr(s1, 'j1_ptrel', float(particlePt[i_particle]/tree.j_pt[0]))
                setattr(s1, 'j1_eta', float(particleEta[i_particle]))
                setattr(s1, 'j1_etarel', float(particleEta[i_particle]-tree.j_eta[0]))
                setattr(s1, 'j1_phi', float(particlePhi[i_particle]))
                setattr(s1, 'j1_phirel', float(particlePhi[i_particle]-jet.Phi()))
                setattr(s1, 'j1_deltaR', float(jet.DeltaR(particle)))
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
