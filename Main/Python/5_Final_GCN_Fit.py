# -*- coding: utf-8 -*-
"""
5_Final_GCN_Fit.py performs several random fits of a given GCN architecture
and saves the best model.

Sections:
    . Imports
    . Configuration
    . Main Functions
        . graphDataset()
        . splitTrainVal()
        . GCN_Model_SP()
        . composite()
        . generateGCN()
    . Main Script
    . Plots
    
Last edit: 2023-11-15
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import gc
import random
import pickle
import copy

# Specific
import numpy
import spektral
import tensorflow
from sklearn import metrics
from matplotlib import pyplot as plt
from spektral import transforms
from tensorflow.keras import backend
from skmultilearn.model_selection.iterative_stratification\
    import iterative_train_test_split as stratSplit

# =============================================================================
# Configuration
# =============================================================================

# Force Field used for atom typing
ffType='GAFF' # One of: "El" | "MMFF"| "GAFF"
# Hyperparameters
hp={'conv1_channels': 239,
    'conv2_channels': 112,
    'conv3_channels': 243,
    'L2 coeff.': 0.00012019192186062926,
    'alpha': 0.0024824547416921233,
    'batchSize': 32}
# Path to Model Folder
modelFolder=r'/path/to/Main/Models'
# Path to the "Databases" folder
databasesFolder=r'/path/to/Main/Databases'

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
        conv3_channels=architecture.get('conv3_channels')
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
    earlyStop=tensorflow.keras.callbacks.EarlyStopping(patience=500,
                                                       mode='min',
                                                     restore_best_weights=True)
    # Fit the model
    hist=model.fit(loaderTrain.load(),
                   steps_per_epoch=loaderTrain.steps_per_epoch,
                   epochs=10000, # Arbitrarily large number
                   validation_data=loaderVal.load(),
                   validation_steps=loaderVal.steps_per_epoch,
                   verbose=verbose,
                   callbacks=earlyStop)
    # Output
    return model,hist

# =============================================================================
# Main Script
# =============================================================================

# Load testing set
with open(os.path.join(databasesFolder,
                       ffType+'_Spektral_Testing.pkl'),'rb') as f:
    graphSet_Test=pickle.load(f)
    graphSet_Test.apply(transforms.GCNFilter())
# Obtain random seeds
randomSeeds=numpy.random.randint(0,1000000,100)
# Initialize best metric
bestMetric=10**3
# Intialize loop over seeds
for seed in randomSeeds:
    # Clear memory
    backend.clear_session()
    gc.collect()
    # Seed trial
    os.environ['PYTHONHASHSEED']=str(seed)
    tensorflow.random.set_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    # Load training dataset
    with open(os.path.join(databasesFolder,
                           ffType+'_Spektral_Training.pkl'),'rb') as f:
        graphSet_Train_=pickle.load(f)
    # Split training into training and validation (80/20)
    graphSet_Train,graphSet_Val=splitTrainVal(graphSet_Train_)
    # Apply filters to adjacency matrices
    graphSet_Train.apply(transforms.GCNFilter())
    graphSet_Val.apply(transforms.GCNFilter())
    # Generate GCN model
    model,hist=generateGCN(hp,graphSet_Train,graphSet_Val,verbose=1)
    # Predict Val
    loaderVal=spektral.data.BatchLoader(graphSet_Val,shuffle=False)
    Y_Val=model.predict(loaderVal.load(),steps=loaderVal.steps_per_epoch)
    # Extract val labels as tensors
    nVal=graphSet_Val.n_graphs
    labelTensor_Val=numpy.zeros((nVal,51))
    for n in range(nVal): labelTensor_Val[n,:]=graphSet_Val[n].y
    # Get loss
    loss=composite()
    metric=loss(labelTensor_Val.flatten(),Y_Val.flatten())
    metric=metric.numpy()
    # Check if best metric
    if metric<bestMetric:
        bestMetric=metric
        # Save model weights
        with open(os.path.join(modelFolder,ffType+'_GCN.pkl'),'wb') as file:
            pickle.dump(model.get_weights(),file,pickle.HIGHEST_PROTOCOL)
        # Predict
        loaderTrain=spektral.data.BatchLoader(graphSet_Train,shuffle=False)
        Y_Train=model.predict(loaderTrain.load(),
                                   steps=loaderTrain.steps_per_epoch)
        loaderVal=spektral.data.BatchLoader(graphSet_Val,shuffle=False)
        Y_Val=model.predict(loaderVal.load(),steps=loaderVal.steps_per_epoch)
        loaderTest=spektral.data.BatchLoader(graphSet_Test,shuffle=False)
        Y_Test=model.predict(loaderTest.load(),
                             steps=loaderTest.steps_per_epoch)
        # Extract labels as tensors
        nTrain=graphSet_Train.n_graphs
        labelTensor_Train=numpy.zeros((nTrain,51))
        for n in range(nTrain):
            labelTensor_Train[n,:]=graphSet_Train[n].y
        nVal=graphSet_Val.n_graphs
        labelTensor_Val=numpy.zeros((nVal,51))
        for n in range(nVal):
            labelTensor_Val[n,:]=graphSet_Val[n].y
        nTest=graphSet_Test.n_graphs
        labelTensor_Test=numpy.zeros((nTest,51))
        for n in range(nTest): 
            labelTensor_Test[n,:]=graphSet_Test[n].y
        with open(os.path.join(modelFolder,
                               ffType+'_labelTensor_Train.pkl'),'wb') as file:
            pickle.dump(labelTensor_Train,file,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(modelFolder,
                               ffType+'_Y_Train.pkl'),'wb') as file:
            pickle.dump(Y_Train,file,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(modelFolder,
                               ffType+'_labelTensor_Val.pkl'),'wb') as file:
            pickle.dump(labelTensor_Val,file,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(modelFolder,
                               ffType+'_Y_Val.pkl'),'wb') as file:
            pickle.dump(Y_Val,file,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(modelFolder,
                               ffType+'_labelTensor_Test.pkl'),'wb') as file:
            pickle.dump(labelTensor_Test,file,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(modelFolder,
                               ffType+'_Y_Test.pkl'),'wb') as file:
            pickle.dump(Y_Test,file,pickle.HIGHEST_PROTOCOL)

# =============================================================================
# Plots
# =============================================================================

# Plot Configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['font.size'] = 12
plt.rcParams["savefig.pad_inches"] = 0.02

# --- Plot losses
plt.semilogy(numpy.array(hist.history['loss']),'--k',label='Training Loss')
plt.semilogy(numpy.array(hist.history['val_loss']),'--r',
             label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Logarithmic Error)')
plt.legend()
plt.show()

# --- Plot pred vs exp scatter
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
plt.text(0.03,0.86,'MAE='+'{:.2f}'.format(MAE_Val),
         horizontalalignment='left',transform=plt.gca().transAxes,c='r')
plt.text(0.03,0.79,'MAE='+'{:.2f}'.format(MAE_Test),
         horizontalalignment='left',transform=plt.gca().transAxes,c='b')
plt.text(0.97,0.17,'$R^2=$'+'{:.2f}'.format(R2_Train),
         horizontalalignment='right',transform=plt.gca().transAxes,c='k')
plt.text(0.97,0.10,'$R^2=$'+'{:.2f}'.format(R2_Val),
         horizontalalignment='right',transform=plt.gca().transAxes,c='r')
plt.text(0.97,0.03,'$R^2=$'+'{:.2f}'.format(R2_Test),
         horizontalalignment='right',transform=plt.gca().transAxes,c='b')
if ffType=='El': title='A.N.'
else: title=ffType
plt.text(0.97,0.5,title,
         horizontalalignment='right',transform=plt.gca().transAxes,c='k')
plt.show()