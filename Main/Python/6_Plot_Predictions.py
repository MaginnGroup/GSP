# -*- coding: utf-8 -*-
"""
6_Plot_Predictions.py plots GCN predictions for assorted compounds.

Sections:
    . Imports
    . Configuration
    . Main Functions
        . graphDataset()
        . splitTrainVal()
        . GCN_Model_SP()
        . composite()
        . generateGCN()
    . Pred Vs. Exp
    . Assorted Compounds
    
Last edit: 2023-07-27
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import pickle
import copy

# Specific
import numpy
import pandas
import spektral
import tensorflow
from spektral import transforms
from tensorflow.keras import backend
from matplotlib import pyplot as plt
from sklearn import metrics
from skmultilearn.model_selection.iterative_stratification\
    import iterative_train_test_split as stratSplit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# Configuration
# =============================================================================

# Path to Model Folder
modelFolder=r'C:\Users\dinis\Desktop\GSP\Main\Models'
# Path to the "Databases" folder
databasesFolder=r'C:\Users\dinis\Desktop\GSP\Main\Databases'
# Force Field used for atom typing
ffType='MMFF' # One of: "El" | "MMFF"| "GAFF"
# Hyperparameters
hp={'conv1_channels': 300,
    'conv2_channels': 300,
    'conv3_channels': 231,
    'L2 coeff.': 1e-10,
    'alpha': 0.0005000455726511215,
    'batchSize': 16}

# =============================================================================
# Main Functions
# =============================================================================

class graphDataset(spektral.data.Dataset):
    """
    spektral.data.Dataset object containing the spektral graph objects of the
    molecules in any given dataset.
    """

def splitTrainVal(graphSet_Train_):
    """
    splitTrainVal() splits graphSet_Train_ into training and validation sets
    (80/20) using stratified sampling based on the atom type counter per
    molecule.
    
    The stratified splitting is done considering the atom types in each
    molecule. Molecules with unique atom types (atom types that only occur in
    four or less molecules) are forced into the training set. All other cases
    follow the iterative stratification for multi-label data method developed
    by Sechidis et al. and Szymański & Kajdanowicz.

    Parameters
    ----------
    graphSet_Train_ : Spektral dataset object
        Original training dataset.

    Returns
    -------
    graphSet_Train : Spektral dataset object
        Spektral training dataset.
    graphSet_Val : Spektral dataset object
        Spektral validation dataset.

    """
    # Get original set size
    N=graphSet_Train_.n_graphs
    # Get full mask of original set
    fullMask=list(range(len(graphSet_Train_)))
    # Define atomTypeMatrix
    atomTypeMatrix=numpy.zeros((N,
                                graphSet_Train_.n_node_features))
    # Iterate over graphSet_Train_
    for n,graph in enumerate(graphSet_Train_):
        # Extract feature set matrix
        x=graph.x
        # Get atom type count
        xCount=x.sum(axis=0)
        # Add to atomTypeMatrix
        atomTypeMatrix[n,:]=xCount
    # Sum rows of atomTypeMatrix
    atomTypeCounter=atomTypeMatrix.sum(axis=0).tolist()
    # Retrieve atom types that only occur in four or less molecules
    singleAtomTypes=[i for i,x in enumerate(atomTypeCounter) if x<5]
    # Retrieve molecules that contain singleAtomTypes
    trainMask=[]
    for singleAtomType in singleAtomTypes:
        # Retrieve index of molecule
        cond1=atomTypeMatrix[:,singleAtomType]>0
        # Append to trainMask
        for index in numpy.argwhere(cond1)[:,0]:
            trainMask.append(index)
    # Remove duplicate entries
    trainMask=list(set(trainMask))
    trainMask.sort()
    # Remove trainMask from available mask and from atomTypeMatrix
    availableMask=[index for index in fullMask if index not in trainMask]
    atomTypeMatrix=numpy.delete(atomTypeMatrix,trainMask,axis=0)
    # Perform stratified splitting
    valFrac=0.2/(1-len(trainMask)/N)
    split=stratSplit(numpy.array(availableMask).reshape(-1,1),
                     atomTypeMatrix,valFrac)
    # Get trainMask and valMask
    trainMask=trainMask+split[0][:,0].tolist()
    trainMask.sort()
    valMask=split[2][:,0].tolist()
    valMask.sort()
    # Get train and val sets
    graphSet_Train=copy.deepcopy(graphSet_Train_[trainMask])
    graphSet_Val=copy.deepcopy(graphSet_Train_[valMask])
    # Ouput
    return graphSet_Train,graphSet_Val

class GCN_Model_SP(tensorflow.keras.models.Model):
    """
    tensorflow.keras.models.Model object containing the architecture of the
    graph neural network model for sigma profile regression.
    """
    def __init__(self,architecture):
        """
        __init__() constrcuts the architecture of the model.

        Parameters
        ----------
        architecture : dict
            See configuration section.

        Returns
        -------
        None.

        """
        super().__init__()
        # Unpack architecture
        conv1_channels=architecture.get('conv1_channels')
        conv2_channels=architecture.get('conv2_channels')
        conv3_channels=architecture.get('conv2_channels')
        reg=tensorflow.keras.regularizers.L2(architecture.get('L2 coeff.'))
        ki='he_uniform'
        # Define userLayers list
        self.userLayers=[]
        # First conv layer
        if conv1_channels>0:
            conv1Layer=spektral.layers.GCNConv(conv1_channels,
                                               activation='relu',
                                               kernel_initializer=ki,
                                               kernel_regularizer=reg,
                                               use_bias=False)
            self.userLayers.append(conv1Layer)
        # Second conv layer
        if conv2_channels>0:
            conv2Layer=spektral.layers.GCNConv(conv2_channels,
                                               activation='relu',
                                               kernel_initializer=ki,
                                               kernel_regularizer=reg,
                                               use_bias=False)
            self.userLayers.append(conv2Layer)
        # Third conv layer
        if conv3_channels>0:
            conv3Layer=spektral.layers.GCNConv(conv3_channels,
                                               activation='relu',
                                               kernel_initializer=ki,
                                               kernel_regularizer=reg,
                                               use_bias=False)
            self.userLayers.append(conv3Layer)
        # Dense layer (X*W)
        dense=tensorflow.keras.layers.Dense(51,
                                            activation='relu',
                                            kernel_initializer=ki,
                                            kernel_regularizer=reg,
                                            use_bias=False)
        self.userLayers.append(dense)
        # Pooling layer
        poolLayer=spektral.layers.GlobalSumPool()
        self.userLayers.append(poolLayer)
    def call(self, inputs):
        """
        call() propagates "inputs" through GCN model.

        Parameters
        ----------
        inputs : tuple
            Tuple containing the adjacency tensor (batch,N,N) and the feature
            tensor (batch,N,F).

        Returns
        -------
        x : tf.Tensor
            Tensor containg the predicted sigma profiles (batch,1,51).

        """
        # Extract node feature vector (x) and adjacency matrix (a) from inputs
        X,A=inputs
        # Conv layers:
        for n in range(len(self.userLayers)-2):
            X=self.userLayers[n]([X,A])
        # Dense layer (X*W)
        X=self.userLayers[-2](X)
        # Pooling layer
        X=self.userLayers[-1](X)
        # Output
        return X

class composite(tensorflow.keras.losses.Loss):
    """
    Composite loss function (MAE+MSLE).
    """
    def __init__(self):
        # Define standard tf MAE
        self.getMAE=tensorflow.keras.losses.MeanAbsoluteError()
        super().__init__()
    def call(self,y_true,y_pred):
        # Get MAE
        MAE=self.getMAE(y_true,y_pred)
        # Get MSLE
        buff=0.1
        y_pred=tensorflow.convert_to_tensor(y_pred)+buff
        y_true=tensorflow.cast(y_true,y_pred.dtype)+buff
        LE=tensorflow.math.log(y_true)-tensorflow.math.log(y_pred)
        SLE=tensorflow.math.square(LE)
        MSLE=backend.mean(SLE,axis=-1)
        # Get loss
        loss=MAE+MSLE
        return loss

def generateGCN(architecture,graphSet_Train,graphSet_Val,verbose=1):
    """
    modelFit() fits the GCN model using the training and validation datasets
    provided.

    Parameters
    ----------
    architecture : dict
        See configuration section.
    graphSet_Train : Spektral dataset object
        Spektral training dataset.
    graphSet_Val : Spektral dataset object
        Spektral validation dataset.
    verbose : int
        Verbose parameter passed to model.fit().

    Returns
    -------
    model : tf.keras object
        tf.keras model object of the fitted GCN.
    hist : TYPE
        DESCRIPTION.

    """
    # Unpack architecture
    alpha=architecture.get('alpha')
    batchSize=architecture.get('batchSize')
    # Build model
    model=GCN_Model_SP(architecture)
    # Generate optimizer
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=alpha)
    # Compile model
    model.compile(optimizer=optimizer,loss=composite())
    # Create loaders
    loaderTrain=spektral.data.BatchLoader(graphSet_Train,batch_size=batchSize,
                                          shuffle=True)
    loaderVal=spektral.data.BatchLoader(graphSet_Val,shuffle=False)
    # Define early stopping
    earlyStop=tensorflow.keras.callbacks.EarlyStopping(patience=1000,
                                                       mode='min',
                                                     restore_best_weights=True)
    # Fit the model
    hist=model.fit(loaderTrain.load(),
                   steps_per_epoch=loaderTrain.steps_per_epoch,
                   epochs=1, # Arbitrarily large number
                   validation_data=loaderVal.load(),
                   validation_steps=loaderVal.steps_per_epoch,
                   verbose=verbose,
                   callbacks=earlyStop)
    # Output
    return model,hist

# =============================================================================
# Pred Vs. Exp
# =============================================================================

# Plot Configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['font.size'] = 14
plt.rcParams["savefig.pad_inches"] = 0.02

with open(os.path.join(modelFolder,ffType+'_labelTensor_Train.pkl'),'rb') as f:
    labelTensor_Train=pickle.load(f)
with open(os.path.join(modelFolder,ffType+'_Y_Train.pkl'),'rb') as f:
    Y_Train=pickle.load(f)
with open(os.path.join(modelFolder,ffType+'_labelTensor_Val.pkl'),'rb') as f:
    labelTensor_Val=pickle.load(f)
with open(os.path.join(modelFolder,ffType+'_Y_Val.pkl'),'rb') as f:
    Y_Val=pickle.load(f)
with open(os.path.join(modelFolder,ffType+'_labelTensor_Test.pkl'),'rb') as f:
    labelTensor_Test=pickle.load(f)
with open(os.path.join(modelFolder,ffType+'_Y_Test.pkl'),'rb') as f:
    Y_Test=pickle.load(f)

v=[0,150]
plt.plot(v,v,'--',linewidth=2,c='silver')
plt.plot(labelTensor_Train.flatten(),
         Y_Train.flatten(),'.k',markersize=5,label='Training Data')
plt.plot(labelTensor_Val.flatten(),
         Y_Val.flatten(),'.r',markersize=5,label='Validation Data')
plt.plot(labelTensor_Test.flatten(),
         Y_Test.flatten(),'.b',markersize=5,label='Testing Data')
plt.xlabel(r'Exp. Sigma Profile $\rm/Å^{2}$')
plt.ylabel(r'Pred. Sigma Profile $\rm/Å^{2}$')
plt.xlim(v)
plt.ylim(v)
# Get metrics
R2_Train=metrics.r2_score(labelTensor_Train.flatten(),
                          Y_Train.flatten())
R2_Val=metrics.r2_score(labelTensor_Val.flatten(),
                        Y_Val.flatten())
R2_Test=metrics.r2_score(labelTensor_Test.flatten(),
                         Y_Test.flatten())
MAE_Train=metrics.mean_absolute_error(labelTensor_Train.flatten(),
                                      Y_Train.flatten())
MAE_Val=metrics.mean_absolute_error(labelTensor_Val.flatten(),
                                    Y_Val.flatten())
MAE_Test=metrics.mean_absolute_error(labelTensor_Test.flatten(),
                                     Y_Test.flatten())
# Print metrics in plot
plt.text(0.03,0.93,'MAE='+'{:.2f}'.format(MAE_Train),
         horizontalalignment='left',transform=plt.gca().transAxes,c='k')
plt.text(0.03,0.84,'MAE='+'{:.2f}'.format(MAE_Val),
         horizontalalignment='left',transform=plt.gca().transAxes,c='r')
plt.text(0.03,0.75,'MAE='+'{:.2f}'.format(MAE_Test),
         horizontalalignment='left',transform=plt.gca().transAxes,c='b')
plt.text(0.97,0.21,'$R^2=$'+'{:.2f}'.format(R2_Train),
         horizontalalignment='right',transform=plt.gca().transAxes,c='k')
plt.text(0.97,0.12,'$R^2=$'+'{:.2f}'.format(R2_Val),
         horizontalalignment='right',transform=plt.gca().transAxes,c='r')
plt.text(0.97,0.03,'$R^2=$'+'{:.2f}'.format(R2_Test),
         horizontalalignment='right',transform=plt.gca().transAxes,c='b')
if ffType=='El': title='A.N.'
else: title=ffType
plt.text(0.97,0.5,title,
         horizontalalignment='right',transform=plt.gca().transAxes,c='k')
plt.show()

# =============================================================================
# Assorted Compounds
# =============================================================================

# Load training and testing sets
with open(os.path.join(databasesFolder,
                       ffType+'_Spektral_Training.pkl'),'rb') as f:
    graphSet_Train=pickle.load(f)
with open(os.path.join(databasesFolder,
                       ffType+'_Spektral_Testing.pkl'),'rb') as f:
    graphSet_Test=pickle.load(f)
# Load GCN model weights
with open(os.path.join(modelFolder,ffType+'_GCN.pkl'),'rb') as f:
      weights=pickle.load(f)
# Recover GCN model
model,__=generateGCN(hp,graphSet_Train,graphSet_Test,verbose=1)
model.set_weights(weights)
# Apply filters to adjacency matrices
graphSet_Train.apply(transforms.GCNFilter())
graphSet_Test.apply(transforms.GCNFilter())
# Predict Sigma Profiles
loaderTrain=spektral.data.BatchLoader(graphSet_Train,shuffle=False)
Y_Train=model.predict(loaderTrain.load(),
                           steps=loaderTrain.steps_per_epoch)
loaderTest=spektral.data.BatchLoader(graphSet_Test,shuffle=False)
Y_Test=model.predict(loaderTest.load(),
                     steps=loaderTest.steps_per_epoch)
# Extract labels as tensors
nTrain=graphSet_Train.n_graphs
labelTensor_Train=numpy.zeros((nTrain,51))
for n in range(nTrain):
    labelTensor_Train[n,:]=graphSet_Train[n].y
nTest=graphSet_Test.n_graphs
labelTensor_Test=numpy.zeros((nTest,51))
for n in range(nTest): 
    labelTensor_Test[n,:]=graphSet_Test[n].y

# Plot Configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.size'] = 15
plt.rcParams["savefig.pad_inches"] = 0.02


# --- Plot best- and worst-performing cases
# Concatenate train, val, and test
Y_True=numpy.concatenate((labelTensor_Train,labelTensor_Test))
Y_Pred=numpy.concatenate((Y_Train,Y_Test))
# Find best and worst performing cases
lossContainer=[]
lossFunction=composite()
for n in range(Y_True.shape[0]):
    # Measure mean squared logarithmic error between sigma profiles
    loss=lossFunction(Y_True[n].reshape(-1,),Y_Pred[n].reshape(-1,))
    lossContainer.append(loss)
lossContainer=numpy.array(lossContainer)
bestIndex1=lossContainer.argmin()
lossContainer[bestIndex1]=lossContainer.mean()
bestIndex2=lossContainer.argmin()
worstIndex1=lossContainer.argmax()
lossContainer[worstIndex1]=lossContainer.mean()
worstIndex2=lossContainer.argmax()

# Find smiles string and VT-2005 index of each case
spDB=pandas.concat((graphSet_Train.spDataset,
                    graphSet_Test.spDataset))
x=(spDB.iloc[:,2:]-Y_True[bestIndex1].reshape(-1,)).abs().sum(axis=1)
x=pandas.to_numeric(x).argmin()
bestVTIndex1=spDB.iloc[x,0]
bestSMILES1=spDB.iloc[x,1]
x=(spDB.iloc[:,2:]-Y_True[bestIndex2].reshape(-1,)).abs().sum(axis=1)
x=pandas.to_numeric(x).argmin()
bestVTIndex2=spDB.iloc[x,0]
bestSMILES2=spDB.iloc[x,1]
x=(spDB.iloc[:,2:]-Y_True[worstIndex1].reshape(-1,)).abs().sum(axis=1)
x=pandas.to_numeric(x).argmin()
worstVTIndex1=spDB.iloc[x,0]
worstSMILES1=spDB.iloc[x,1]
x=(spDB.iloc[:,2:]-Y_True[worstIndex2].reshape(-1,)).abs().sum(axis=1)
x=pandas.to_numeric(x).argmin()
worstVTIndex2=spDB.iloc[x,0]
worstSMILES2=spDB.iloc[x,1]
# Plot best case
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,Y_True[bestIndex1].reshape(-1,),'-k',label='Exp.')
plt.plot(sigma,Y_Pred[bestIndex1].reshape(-1,),'--r',label='Pred.')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
#plt.title('Best-Performing Prediction ( '+str(bestVTIndex)+')')
#plt.legend()
plt.show()
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,Y_True[bestIndex2].reshape(-1,),'-k',label='Exp.')
plt.plot(sigma,Y_Pred[bestIndex2].reshape(-1,),'--r',label='Pred.')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
#plt.title('Best-Performing Prediction ( '+str(bestVTIndex)+')')
#plt.legend()
plt.show()
# Plot worst case
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,Y_True[worstIndex1].reshape(-1,),'-k',label='Exp.')
plt.plot(sigma,Y_Pred[worstIndex1].reshape(-1,),'--r',label='Pred.')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
#plt.title('Worst-Performing Prediction ( '+str(worstVTIndex)+')')
#plt.legend()
plt.show()
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,Y_True[worstIndex2].reshape(-1,),'-k',label='Exp.')
plt.plot(sigma,Y_Pred[worstIndex2].reshape(-1,),'--r',label='Pred.')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
#plt.title('Worst-Performing Prediction ( '+str(worstVTIndex)+')')
#plt.legend()
plt.show()
# Plot L-Phenylalanine
VTIndex=1518
aux=spDB[spDB['VT-2005 Index']==VTIndex]
x=numpy.abs(aux.iloc[0,2:].to_numpy()-Y_True).sum(axis=1).argmin()
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,Y_True[x].reshape(-1,),'-k',label='Exp.')
plt.plot(sigma,Y_Pred[x].reshape(-1,),'--r',label='Pred.')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
#plt.title('Prediction for '+str(VTIndex))
#plt.legend()
plt.show()
# Plot Cinnamic Acid (1197)
VTIndex=1197
aux=spDB[spDB['VT-2005 Index']==VTIndex]
x=numpy.abs(aux.iloc[0,2:].to_numpy()-Y_True).sum(axis=1).argmin()
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,Y_True[x].reshape(-1,),'-k',label='Exp.')
plt.plot(sigma,Y_Pred[x].reshape(-1,),'--r',label='Pred.')
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
#plt.title('Prediction for '+str(VTIndex))
#plt.legend()
plt.show()