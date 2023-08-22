# -*- coding: utf-8 -*-
"""
2_Generate_Datasets.py iterates over the sigma profile database and generates
force field atom types for each entry. Entries that are not described by the
force field (i.e., missing atom type parameters) are discarded. It also splits
the final database into training and testing sets (90/10) using stratified
sampling. 
The force fields available are:
    El - Simply assignes atomic numbers to each atom
    MMFF - As implemented in https://doi.org/10.1186/s13321-014-0037-3 using
           RDKit
    GAFF - Using antechamber and the atom type list in
           https://ambermd.org/antechamber/gaff.html#atomtype
           Requires the Antechamber folder to be built using
           Aux_Generate_AC_Input_Files.py and Aux_Run_AC.py

The stratified splitting is done considering the atom types in each molecule.
Molecules with unique atom types (atom types that only occur in nine or less
molecules) are forced into the training set. All other cases follow the
iterative stratification for multi-label data method developed by Sechidis et 
al. and Szymański & Kajdanowicz.
 
This script generates the following files (** replaced by force field name):
    . **_2_Generate_Datasets.log - Log file containing entries removed and info
      about the final, curated datasets.
    . **_spDatabase_Train.csv - Training split of the curated dataset
    . **_spDatabase_Test.csv - Testing split of the curated dataset
    . **_Spektral_Training.pkl - Training split of the curated datset as a
      Spektral graph dataset object.
    . **_Spektral_Testing.pkl - Testing split of the curated datset as a
      Spektral graph dataset object.

Sections:
    . Configuration
    . GAFF Atom Type Numerical Assignment
    . Imports
    . Auxiliary Functions
        . getAtomTypes()
        . generateGraph()
        . graphDataset()
    . Main Script: Part 1
    . Main Script: Part 2
    . Main Script: Part 3
    . Main Script: Part 4

Last edit: 2023-07-27
Author: Dinis Abranches
"""

# =============================================================================
# Configuration
# =============================================================================

# Path to Databases folder (should contain spDatabase.cs)
databasesFolder=r'C:\Users\dinis\Desktop\GSP\Main\Databases'
# Force Field used for atom typing
ffType='GAFF' # One of: "El" | "MMFF"| "GAFF"

# =============================================================================
# GAFF Atom Type Numerical Assignment
# =============================================================================

# https://ambermd.org/antechamber/gaff.html#atomtype
listGAFF=['c','c1','c2','c3','ca','n','n1','n2','n3','n4','na','nh','no','f',
          'cl','br','i','o','oh','os','s2','sh','ss','s4','s6','hc','ha','hn',
          'ho','hs','hp','p2','p3','p4','p5','h1','h2','h3','h4','h5','n','nb',
          'nc','nd','sx','sy','cc','cd','ce','cf','cp','cq','cu','cv','cx',
          'cy','pb','pc','pd','pe','pf','px','py',
          # Included in Antechamber
          'cg','ch','ne','s','nf']

# =============================================================================
# Imports
# =============================================================================

# General
import os
import pickle

# Specific
import pandas
import numpy
import spektral
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers as rdFF
from skmultilearn.model_selection.iterative_stratification\
    import iterative_train_test_split as stratSplit

# =============================================================================
# Auxiliary Functions
# =============================================================================

def getAtomTypes(SMILES,index,ffType):
    """
    getAtomTypes() obtaines the atom type for the molecule with SMILES "SMILES"
    and spDB index "index". The atom type is determined by ffType.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.
    index : int
        VT-2005 index of the molecule.
    ffType : str
        Force Field type. One of: "El", "MMFF", "GAFF"
            
    Returns
    -------
    atomTypes : list of ints
        List containing the numeric atom types of the atoms in the molecule.
    error : boolean
        Whether an error occured.

    """
    # Define outputs
    atomTypes=[]
    error=0
    # Check ffType
    if ffType=='El':
        # Obtain RDKit mol object
        molecule=Chem.MolFromSmiles(SMILES)
        # Make hydrogens explicit
        molecule=AllChem.AddHs(molecule)
        # Retrieve node-level features
        for atom in molecule.GetAtoms():
            # Get atomic number
            atomType=atom.GetAtomicNum()
            # Append
            atomTypes.append(atomType)
    elif ffType=='MMFF':
        # Obtain RDKit mol object
        molecule=Chem.MolFromSmiles(SMILES)
        # Make hydrogens explicit
        molecule=AllChem.AddHs(molecule)
        # Generate initial 3D structure of the molecule
        AllChem.EmbedMolecule(molecule)
        # Initialize MMFF props object
        prop=rdFF.MMFFGetMoleculeProperties(molecule)
        # Check if MMFF exists for entry
        if prop is None:
            error=1
        else:
            # Retrieve node-level features
            for atom in molecule.GetAtoms():
                # Get MMFF atom type
                atomType=prop.GetMMFFAtomType(atom.GetIdx())
                # Append
                atomTypes.append(atomType)
    elif ffType=='GAFF':
        # Convert index to YYYY format
        index=str(index).zfill(4)
        # Get .ac file path
        acPath=os.path.join(databasesFolder,'Antechamber',index+'.ac')
        # Check if file exists
        if os.path.isfile(acPath):
            # Open .ac file
            with open(acPath) as file:
                # Skip first two lines
                file.readline()
                file.readline()
                # Loop over atoms
                while True:
                    # Read atom line
                    aux=file.readline()
                    # Check if atom section ended
                    if aux.split()[0]!='ATOM': break
                    # Get GAFF atom type
                    atomTypeSTR=aux.split()[-1]
                    # Check atom type exists in GAFF list
                    if atomTypeSTR in listGAFF:
                        # Convert to numerical type
                        atomType=listGAFF.index(atomTypeSTR)
                        # Append
                        atomTypes.append(atomType)
                    else:
                        error=1
                        break
        else:
            error=1
    # Output
    return atomTypes,error

def generateGraph(SMILES,index,ffType,labels,listFF):
    """
    generateGraph() generates a spektral graph object of the molecule described
    by the SMILES string "SMILES". This is done using the adjacency matrix
    generated by RDKit.
    The one-hot vector of the FF atom type is included as a node-level feature.
    "labels" are added as the "label" parameter of the graph object.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.
    int : int
        VT-2005 index of the molecule.
    ffType : str
        Force Field type. One of: "El", "MMFF", "GAFF"
    labels : numpy array (L,)
        Label vector associated to the molecule. For sigma profiles, L=51.
    listFF : list of ints
        List containing all atom types in the database being used. This is used
        to one-hot encode the atom types in the molecule.

    Returns
    -------
    graph : spektral.data.graph.Graph object
        Spektral graph object representing the molecule "SMILES".

    """
    # Generate rdkit molecule object
    molecule=Chem.MolFromSmiles(SMILES)
    # Make hydrogens explicit
    molecule=AllChem.AddHs(molecule)
    # Generate initial 3D structure of the molecule
    AllChem.EmbedMolecule(molecule)
    # Minimizie initial guess with MMFF
    AllChem.MMFFOptimizeMolecule(molecule)
    # Get adjacency matrix
    adjacencyMatrix=Chem.GetAdjacencyMatrix(molecule).astype('float32')
    # Get node features (atom types)
    nodeFt,__=getAtomTypes(SMILES,index,ffType)
    # Generate one-hot encoded atom type matrix for molecule
    aux=numpy.zeros((len(nodeFt),len(listFF)))
    # Iterate over nodeFt
    for i in range(len(nodeFt)):
        # Map force field atom type to index in uniqueAtomTypes
        oneHotIndex=listFF.index(nodeFt[i])
        # One-hot encode atom i
        oneHot=spektral.utils.one_hot(oneHotIndex,len(listFF))
        # Add entry to aux
        aux[i,:]=oneHot
    # Convert aux to numpy array
    nodeFt=numpy.array(aux)
    # Generate Spektral graph object
    graph=spektral.data.graph.Graph(x=nodeFt,a=adjacencyMatrix,e=None,y=labels,
                                    VT2005Index=index)
    # Output
    return graph

class graphDataset(spektral.data.Dataset):
    """
    spektral.data.Dataset object containing the spektral graph objects of the
    molecules in any given dataset.
    """
    def __init__(self,spDataset,ffType,listFF,verbose=1):
        """
        Initialize object.

        Parameters
        ----------
        spDataset : pandas DataFrame
            Dataframe containing the sigma profile dataset of interest, which
            is converted to a spektral graph dataset.
        ffType : str
            Force Field type. One of: "El", "MMFF", "GAFF"
        listFF : list of ints
            List containing all atom types in the database being used. This is
            used to one-hot encode the atom types in the molecule.
        verbose : int, optional
            Whether to use (1) or not (0) tqdm.
            The default is 1.

        Returns
        -------
        None.

        """
        self.spDataset=spDataset
        self.ffType=ffType
        self.listFF=listFF
        self.verbose=verbose
        super().__init__()

    def read(self):
        """
        read() is called alongside __init__(). It iterates over
        "spDataset" and, for each entry, generates the corresponding
        graph object.

        Returns
        -------
        dataset : list of spektral.data.graph.Graph objects
            List containing the spektral graph objects of all molecules in
            "spDataset".

        """
        # Initialize container for Spektral dataset
        dataset=[]
        # Iterate over dataset provided
        if self.verbose:
            iterator=tqdm(self.spDataset.itertuples(),
                          'Generating Graph Dataset: ')
        else:
            iterator=self.spDataset.itertuples()
        for entry in iterator:
            # Get index
            index=entry[1]
            # Get SMILES string
            SMILES=entry[2]
            # Get labels
            labels=numpy.array(entry[3:])
            # Generate Spektral graph
            graph=generateGraph(SMILES,index,self.ffType,labels,self.listFF)
            # Append to dataset
            dataset.append(graph)
        # Output
        return dataset

# =============================================================================
# Main Script: Part 1
# =============================================================================
"""
Objectives:
    . Remove molecules from the dataset where at least one atom is not
      described by the force field requested.
    . Obtain all force field atom types for each molecule.
"""
# Define spDatabase path
spDB_Path=os.path.join(databasesFolder,'spDatabase.csv')
# Load spDatabase
spDB=pandas.read_csv(spDB_Path)
# Define path to log file
logFilePath=os.path.join(databasesFolder,ffType+'_spDatabase_Generation.log')
# Generate empty spDatabase
columnString=['VT-2005 Index','SMILES']
sigmaAxis=numpy.linspace(-0.025,0.025,51)
for n in range(51): columnString.append(f'{sigmaAxis[n]:.3f}')
new_spDB=pandas.DataFrame([],columns=columnString)
# Initialize atom type list
atomTypesList=[]
# Open log file
with open(logFilePath,'w') as logFile:
# Iterate over spDB
    for n in tqdm(range(len(spDB)), 'Molecule: '):
        # Get VT-2005 Index
        index=spDB.iloc[n,0]
        # Get SMILES string of molecule
        SMILES=spDB.iloc[n,1]
        # Get sigma profile
        sp=spDB.iloc[n,2:]
        # Get atom types
        atomTypes,error=getAtomTypes(SMILES,index,ffType)
        # Check errors
        if error:
            logFile.write('Failure for VT-2005 index: '+str(index))
            logFile.write('\n   '+ffType+' not fully available for SMILES: '
                          +SMILES+'\n\n')
            continue
        # Append full list of atom types in current molecule to atomTypesList
        atomTypesList.append(atomTypes)
        # Add entry to spDatabase
        new_spDB.loc[len(new_spDB.index)]=[index,SMILES]\
            +sp.to_list()
    # Write to log file
    logFile.write('Finished checking spDatabase for usage with '
                  +ffType
                  +' atom types.')

# =============================================================================
# Main Script: Part 2
# =============================================================================
"""
Objectives:
    . Obtain total number of different atom types in the new sp dataset.
    . Obtain atom type counter in each individual molecule, as well as in the
      full new sp dataset.
"""
# Get set of unique force field atom types
uniqueAtomTypes=list(set([item for sub in atomTypesList for item in sub]))
# Get total number of atoms in new sp dataset
totalAtoms=len([item for sub in atomTypesList for item in sub])
# Total size of new sp dataset
N=len(new_spDB)
# Count total number of different atom types
F=len(uniqueAtomTypes)
# Generate one-hot encoded atom type counter matrix (N,F) for the dataset
atomTypeMatrix=numpy.zeros((N,F))
# Iterate over molecules in new sp dataset
for n in range(N):
    # Retrive list of atom types in molecule n
    atomTypes=atomTypesList[n]
    # Generate one-hot encoded atom type matrix for molecule
    aux=numpy.zeros((len(atomTypes),F))
    # Iterate over atomTypes
    for i in range(len(atomTypes)):
        # Map force field atom type to index in uniqueAtomTypes
        oneHotIndex=uniqueAtomTypes.index(atomTypes[i])
        # One-hot encode atom i
        oneHot=spektral.utils.one_hot(oneHotIndex,F)
        # Add entry to aux
        aux[i,:]=oneHot
    # Sum over aux
    oneHotCounter=aux.sum(axis=0)
    # Add oneHotCounter to atomTypeMatrix
    atomTypeMatrix[n,:]=oneHotCounter
# Convert atomTypeMatrix to int
atomTypeMatrix=atomTypeMatrix.astype(int)
# Sum rows of atomTypeMatrix
atomTypeCounter=atomTypeMatrix.sum(axis=0).tolist()
# Print info to log file
with open(logFilePath,'a') as logFile:
    logFile.write('\n   Dataset size: '+str(N))
    logFile.write('\n   Total number of atoms in dataset: '+str(totalAtoms))
    logFile.write('\n   Total number of '+ffType+' atom types in dataset: '
                  +str(F))
    logFile.write('\n   '+ffType+' atom types in dataset: '
                  +str(uniqueAtomTypes))
    logFile.write('\n   '+ffType+' atom type counter in dataset: '
                  +str(atomTypeCounter))

# =============================================================================
# Main Script: Part 3
# =============================================================================
"""
Objectives:
    . Split the dataset into training and testing using stratified sampling
    based on the atom type counter per molecule.
"""
# Intialize new_spDB_Train
new_spDB_Train=pandas.DataFrame([],columns=columnString)
# Retrieve atom types that only occur in nine or less molecules
singleAtomTypes=[i for i,x in enumerate(atomTypeCounter) if x<2]
# Retrieve molecules that contain singleAtomTypes
indexList=[]
for singleAtomType in singleAtomTypes:
    # Retrieve index of molecule
    cond1=atomTypeMatrix[:,singleAtomType]>0
    # Append to indexList
    for index in numpy.argwhere(cond1)[:,0]:
        indexList.append(index)
# Remove duplicate entries
indexList=list(set(indexList))
indexList.sort()
# Add these molecules to training set
for index in indexList:
    new_spDB_Train=pandas.concat([new_spDB_Train,
                                  new_spDB.iloc[index,:].to_frame().T],
                                 ignore_index=True)
# Remove these molecules from new_spDB and atomTypeMatrix
new_spDB=new_spDB.drop(indexList)
atomTypeMatrix=numpy.delete(atomTypeMatrix,indexList,axis=0)
# Perform stratified splitting
testFrac=0.1/(1-len(new_spDB_Train)/N)
split=stratSplit(new_spDB.to_numpy(),atomTypeMatrix,testFrac)
new_spDB_Train=pandas.concat((new_spDB_Train,
                              pandas.DataFrame(split[0],columns=columnString)),
                             axis=0)
new_spDB_Test=pandas.DataFrame(split[2],columns=columnString)
new_spDB_Train=new_spDB_Train.sort_values('VT-2005 Index')
new_spDB_Test=new_spDB_Test.sort_values('VT-2005 Index')
# Save files
new_spDB_Train.to_csv(os.path.join(databasesFolder,
                                   ffType+'_spDatabase_Train.csv'),
                      index=False)
new_spDB_Test.to_csv(os.path.join(databasesFolder,
                                  ffType+'_spDatabase_Test.csv'),
                     index=False)

# =============================================================================
# Main Script: Part 4
# =============================================================================
"""
Objectives:
    . Generate spektral graph datasets for the training and testing sigma
    profile datasets.
    . Save final files.
"""
# Generate training and testing spektral datasets
spektralTraining=graphDataset(new_spDB_Train,ffType,uniqueAtomTypes)
spektralTesting=graphDataset(new_spDB_Test,ffType,uniqueAtomTypes)
# Save variables
with open(os.path.join(databasesFolder,
                       ffType+'_Spektral_Training.pkl'),'wb') as file:
    pickle.dump(spektralTraining,file,pickle.HIGHEST_PROTOCOL)
with open(os.path.join(databasesFolder,
                       ffType+'_Spektral_Testing.pkl'),'wb') as file:
    pickle.dump(spektralTesting,file,pickle.HIGHEST_PROTOCOL)
