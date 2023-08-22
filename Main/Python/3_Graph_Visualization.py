# -*- coding: utf-8 -*-
"""
3_Graph_Visualization is a small Python script to visualize the graphs used in
this work, as well as the labels (sigma profiles). For a given identifier, the
script plots:
    1. Molecule skeleton (RDKit)
    2. Graph representation of the molecule, with  atom types as node-level 
       features (networkx)
    3. Sigma profile of the molecule

Sections:
    . Configuration
    . GAFF Atom Type Numerical Assignment
    . Imports
    . Auxiliary Functions
        . getAtomTypes()
        . generateGraph_NetworkX()
        . plotGraph_NetworkX()
    . Main Script

Last edit: 2023-07-27
Author: Dinis Abranches
"""

# =============================================================================
# Configuration
# =============================================================================

# Path to the "Databases" folder
databasesFolder=r'C:\Users\dinis\Desktop\GSP\Main\Databases'
# Force Field used for atom typing
ffType='MMFF' # One of: "El" | "MMFF"| "GAFF"
# Identifier of desired molecule: index (int) or SMILES (string)
identifier=405

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

# Specific
import pandas
import numpy
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers as rdFF
from rdkit.Chem import Draw
import networkx
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"]=600

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

def generateGraph_NetworkX(SMILES,index):
    """
    generateGraph_NetworkX() generates a NetworkX graph object of the molecule
    described by the SMILES string "SMILES". The nodes and edges are added
    considering the atom indexes in the molecule. Each node has one node-level
    feature (MMFF atom type, not one-hot encoded for visualization purposes).

    Parameters
    ----------
    SMILES : string
        SMILES string of the molecule of interest.
    index : int
        VT-2005 index of the molecule.

    Returns
    -------
    graph : networkx.classes.graph.Graph object
        NetworkX graph object representing the molecule "SMILES".

    """
    # Generate rdkit molecule object
    molecule=Chem.MolFromSmiles(SMILES)
    # Make hydrogens explicit
    molecule=AllChem.AddHs(molecule)
    # Get atom types
    atomTypes,error=getAtomTypes(SMILES,index,ffType)
    # Check error
    if error: raise ValueError(ffType
                               +' Force Field not available for desired index')
    
    # Initialize empty NetworkX graph
    graph=networkx.Graph()
    # Add nodes by looping over atoms in molecule
    for n,atom in enumerate(molecule.GetAtoms()):
        attribute={'feature':atomTypes[n]}
        # Add node to graph
        graph.add_nodes_from([(atom.GetIdx(),attribute)])
    # Add edges by looping over bonds in molecule
    for bond in molecule.GetBonds():
        # Add edge to graph
        graph.add_edges_from([(bond.GetBeginAtomIdx(),
                               bond.GetEndAtomIdx())])
    # Output
    return graph

def plotGraph_NetworkX(graph):
    """
    plotGraph_NetworkX() plots the NetworkX graph "graph". The graph is plotted
    using the node features as labels, and using node and edge features as
    color scales.

    Parameters
    ----------
    graph : networkx.classes.graph.Graph object
        NetworkX graph object representing the molecule "SMILES".

    Returns
    -------
    None.

    """
    # Define layout (kamada_kawai_layout is an excellent option for molecules)
    layout=networkx.kamada_kawai_layout(graph)
    # Plot graph
    labels=networkx.get_node_attributes(graph,'feature')
    nodeFeatures=list(labels.values())
    uniqueFeatures=set(nodeFeatures)
    colorScheme=numpy.linspace(0,1000,num=len(uniqueFeatures)).round()
    colorDict=dict(zip(uniqueFeatures,colorScheme))
    replacer=colorDict.get
    colorList=[replacer(n,n) for n in nodeFeatures]
    networkx.draw_networkx(graph,pos=layout,
                           node_color=colorList,
                           labels=labels,
                           font_color='w')
    plt.axis('off')
    plt.show()
    # Output
    return None

# =============================================================================
# Main Script
# =============================================================================

# Load entire database
spDB=pandas.read_csv(os.path.join(databasesFolder,'spDatabase.csv'))
# Try to find "identifier" in "Index" column
target=spDB[spDB['VT-2005 Index']==identifier]
# If target is empty, try to find it in "SMILES" column
if target.empty:
    target=spDB[spDB['SMILES']==identifier]
# If target is still empty, raise exception
if target.empty:
    raise ValueError('Could not find identifier.')
# Unpack target
index=target.iloc[0,0]
SMILES=target.iloc[0,1]
sp=target.iloc[0,2:].to_list()
# Draw RDKit Molecule
molecule=Chem.MolFromSmiles(SMILES)
molecule=AllChem.AddHs(molecule)
fig=plt.figure()
axes=plt.axes([0.6, 0.47, 0.38, 0.38],frameon=True)
axes.imshow(Draw.MolToImage(molecule))
axes.axis('off')
plt.show()
plt.clf()
# Generate networkx graph object
graph=generateGraph_NetworkX(SMILES,index)
# Plot graph
plotGraph_NetworkX(graph)
# Plot sigma profile
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 13
plt.rcParams["savefig.pad_inches"] = 0.02
plt.plot()
sigmaAxis=numpy.linspace(-0.025,0.025,51)
plt.plot(sigmaAxis,sp,'--k')
plt.plot(sigmaAxis,sp,'*k')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
plt.show()
# Print information
print('Molecule Index: '+str(target.iloc[0,0]))
print('Molecule SMILES: '+target.iloc[0,1])
