# -*- coding: utf-8 -*-
"""
1_VT2005_To_SMILES.py obtains the SMILES string of the compounds present in the
original VT-2005 sigma profile database.

The original VT-2005 database contains:
    . VT-2005_Sigma_Profile_Database_Index_v2.xls
        Excel file with the index, name, CAS number, and additional information
        about each compound in the database.
    . VT-2005_Sigma_Profiles_v2
        Folder with txt files. Each txt file contains the sigma profile of a
        given compound.
    . VT-2005_GO_OUTMOL_Files_v2
        Folder with Materials Studio Energy Calculation *.outmol output files.

For each entry in the original VT-2005 sigma profile database, the following
steps are performed:
    1. Obtain the sigma profile from the txt file in
       "VT-2005_Sigma_Profiles_v2".
    2. Generate the SMILES string using "xyz2mol" and the OUTMOL file in
       "VT-2005_GO_OUTMOL_Files_v2".
If step 2 fails, the entry is removed from the dataset. This is documented in
the log file.

This script generates the following files:
    . spDatabase.csv - Sigma profile database
    . 1_VT2005_To_SMILES.log - Log file containing entries removed and info
      about the final dataset.

Structure of spDatabase.csv:
    . Column 0 - VT-2005 molecule index
    . Columm 1 - Molecule SMILES string
    . Column 2:52 - Sigma profile values (vector of size 51)

Sections:
    . Configuration
    . Imports
    . Auxiliary Functions
        . goToLastLine()
        . readOutmol()
    . Main Script

Last edit: 2023-07-27
Author: Dinis Abranches
"""

# =============================================================================
# Configuration
# =============================================================================

# Path to the original VT-2005 database
originalPath=\
    r'C:\Users\dinis\Desktop\GSP\Main\Databases\Original_VT_2005'
# Path to the folder where the new curated dataset is to be saved
newPath=r'C:\Users\dinis\Desktop\GSP\Main\Databases'
# List of Exceptions
exceptions={'0383':'[HH]',
            '0386':'[O-][N+](=O)[N+]([O-])=O',
            '1069':'O[PH]O'}

# =============================================================================
# Imports
# =============================================================================

# General
import os

# Specific
import numpy
import pandas
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import xyz2mol

# =============================================================================
# Auxiliary Functions
# =============================================================================

def goToLastLine(file,fragmentList):
    """
    goToLastLine() places the pointer of "file" at the last line containing all
    fragments in fragmentList. Line fragments are obtained with line.split().

    Parameters
    ----------
    file : _io.TextIOWrapper object
        File of interest.
    fragmentList : list of strings
        List containing fragments (splits) of the line of interest.
        
    Returns
    -------
    found : boolean
        Wether the line was found.

    """
    # Rewind file
    file.seek(0)
    # Define found
    found=False
    # Initiate line counter
    lineNum=0
    # Read file line by line
    for line in file:
        # Split line
        lineSplit=line.split()
        # Check if lineSplit contains fragmentList
        isContained=all(fragment in lineSplit for fragment in fragmentList)
        if isContained:
            found=True
            # Update line number of last occurrence
            lastOccurrenceLine=lineNum
        # Update line number
        lineNum+=1
    # Rewind file
    file.seek(0)
    # Go to last line, if found
    for __ in range(lastOccurrenceLine+1):
        file.readline()
    # Output
    return found

def readOutmol(outmolPath):
    """
    readOutmol() reads an OUTMOL file and returns the xyz information of the
    last geometry cycle optimization.

    Parameters
    ----------
    outmolPath : string
        Path to the outmol file of interest.

    Raises
    ------
    ValueError
        An exception is raised if the outmol file does not contain geometry
        information.

    Returns
    -------
    atomList : list of ints
        List where each entry is the element of an atom. Converted from a list
        of strings using [xyz2mol.int_atom(atom) for atom in atomList]
    coordsList : list of lists of floats
        List where each entry is a list containing the xyz coordinates of an
        atom.

    """
    # Initialize atom and coordinates lists
    atomList=[]
    coordsList=[]
    # Open outmol file
    with open(outmolPath,'r') as file:
        # Find last "Input Coordinates (Angstroms)" section
        found=goToLastLine(file,['Input','Coordinates','(Angstroms)'])
        # Raise exception if line not found
        if not found: raise ValueError('Could not read the following outmol '
                                       +'file:\n'+outmolPath)
        # Skip two lines
        for __ in range(2): file.readline()
        # Read XYZ lines
        for line in file:
            # Split line
            lineSplit=line.split()
            # Check if line is an xyz line
            if len(lineSplit)!=5: break
            # Append atoms and coords
            atomList.append(lineSplit[1])
            coordsList.append([float(lineSplit[2]),
                               float(lineSplit[3]),
                               float(lineSplit[4])])
    # Convert atomList to xyz2mol standards
    atomList=[xyz2mol.int_atom(atom) for atom in atomList]
    # Output
    return atomList,coordsList

# =============================================================================
# Main Script
# =============================================================================

# Define path to the log file
logFilePath=os.path.join(newPath,'1_VT2005_To_SMILES.log')
# Define path to the outmol folder
outmolFolder=os.path.join(originalPath,'VT-2005_GO_OUTMOL_Files_v2')
# Define path to the txt folder   
txtFolder=os.path.join(originalPath,'VT-2005_Sigma_Profiles_v2')
# Define path to the VT-2005 index file
indexPath=os.path.join(originalPath,
                       'VT-2005_Sigma_Profile_Database_Index_v2.xls')
# Read VT-2005 Index File
indexFile=pandas.read_excel(indexPath)
# Generate empty spDatabase
columnString=['VT-2005 Index','SMILES']
sigmaAxis=numpy.linspace(-0.025,0.025,51)
for n in range(51): 
    columnString.append(f'{sigmaAxis[n]:.3f}')
spDatabase=pandas.DataFrame([],columns=columnString)
# Open log file
with open(logFilePath,'w') as logFile:
# Iterate over indexFile
    for n in tqdm(range(len(indexFile)), 'Molecule: '):
        # Get vt-2005 index of compound
        index=indexFile.iloc[n,0]
        # Convert index to YYYY format
        index=str(index).zfill(4)
        # Check if exception
        if index in list(exceptions.keys()):
            SMILES=exceptions[index]
            logFile.write('Manual override for VT-2005 index: '+index)
            logFile.write('\n   SMILES string used: '+SMILES+'\n\n')
        else:
            # Get txt path
            txtPath=os.path.join(txtFolder,'VT2005-'+index+'-PROF.txt')  
            # Read sigma profile
            sp=pandas.read_csv(txtPath,dtype=float,header=None,
                               delim_whitespace=True)
            # Get outmol path
            outmolPath=os.path.join(outmolFolder,'VT2005-'+index+'-GO.outmol')
            # Read outmolPath
            atomList,coordsList=readOutmol(outmolPath)
            # Convert to mol
            molList=xyz2mol.xyz2mol(atomList,coordsList)
            # Check size of molList
            if len(molList)>1:
                logFile.write('Failure for VT-2005 index: '+index)
                logFile.write('\n   More than one possibility for file: '
                              +os.path.basename(outmolPath)+'\n\n')
                continue
            elif not molList:
                logFile.write('Failure for VT-2005 index: '+index)
                logFile.write('\n   Could not convert outmol file to SMILES: '
                              +os.path.basename(outmolPath)+'\n\n')
                continue
            # Convert to SMILES
            aux=Chem.MolToSmiles(molList[0])
            molecule=Chem.MolFromSmiles(aux)
            SMILES=Chem.MolToSmiles(molecule)
        # Add entry to spDatabase
        spDatabase.loc[len(spDatabase.index)]=[index,SMILES]\
            +sp.iloc[:,1].to_list()
    # Print atom information to log
    logFile.write('Finished converting database.')
    logFile.write('\n   Dataset size: '+str(len(spDatabase)))
# Write dataset to file
spDatabase.to_csv(os.path.join(newPath,'spDatabase.csv'),index=False)