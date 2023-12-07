# -*- coding: utf-8 -*-
"""
4_Hyperparameter_Tuning.py tunes the hyperparameters of the GCN models
developed in this work. 

Sections:
    . Imports
    . Configuration
    . Main Functions
        . graphDataset()
        . splitTrainVal()
        . GCN_Model_SP()
        . composite()
        . generateGCN()
        . TuneModelHPs_BO()
    . Main Script
    
Last edit: 2023-11-15
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import gc
import pickle
import copy

# Specific
import numpy
import spektral
import tensorflow
import keras_tuner
from spektral import transforms
from tensorflow.keras import backend
from skmultilearn.model_selection.iterative_stratification\
    import iterative_train_test_split as stratSplit

# =============================================================================
# Configuration
# =============================================================================

# Force Field used for atom typing
ffType='El' # One of: "El" | "MMFF"| "GAFF"
# Path to HyperparameterSearch Folder
hpFolder=r'/path/to/Main/HyperparameterSearch'
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
    earlyStop=tensorflow.keras.callbacks.EarlyStopping(patience=500,mode='min',
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

class TuneModelHPs_BO(keras_tuner.BayesianOptimization):
    """
    Tuning algorithm.
    """
    def run_trial(self,trial,**kwargs):
        # Clear memory
        backend.clear_session()
        gc.collect()
        # Retrieve hyperparameters for current trial
        trialHP=trial.hyperparameters
        # Define hyperparameters
        hpSet={'conv1_channels':trialHP.get('conv1_channels'),
               'conv2_channels':trialHP.get('conv2_channels'),
               'conv3_channels':trialHP.get('conv3_channels'),
               'L2 coeff.':trialHP.get('L2 coeff.'), 
               'alpha':trialHP.get('alpha'),
               'batchSize':trialHP.get('batchSize')}
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
        model,hist=generateGCN(hpSet,graphSet_Train,graphSet_Val,verbose=0)
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
        # Output
        return metric

# =============================================================================
# Main Script
# =============================================================================

# Define path to hp subfolder
hpFolder=os.path.join(hpFolder,ffType)

# Define fixed fitting hyperparameters to use in the first BO pass of model HPs
best={'conv1_channels': 100,
      'conv2_channels': 100,
      'conv3_channels': 100,
      'L2 coeff.': 1e-5,
      'alpha': 0.001,
      'batchSize': 16}

# Tune hyperparameters
for n in range(5):
    # Define architecture hyperparameter space
    hp=keras_tuner.HyperParameters()
    hp.Int('conv1_channels',0,300)
    hp.Int('conv2_channels',0,300)
    hp.Int('conv3_channels',0,300)
    hp.Fixed('L2 coeff.',best.get('L2 coeff.'))
    hp.Fixed('alpha',best.get('alpha'))
    hp.Fixed('batchSize',best.get('batchSize'))
    # Tune architecture hyperparameters
    modelBO=TuneModelHPs_BO(max_trials=50,
                            hyperparameters=hp,
                            directory=hpFolder,
                            project_name=str(n+1)+'_BO_Architecture',
                            tuner_id=str(n+1)+'_BO_Architecture')
    modelBO.search()
    # Get best hyperparameters
    best=modelBO.get_best_hyperparameters()[0].values
    # Define fitting hyperparameter space
    hp=keras_tuner.HyperParameters()
    hp.Fixed('conv1_channels',best.get('conv1_channels'))
    hp.Fixed('conv2_channels',best.get('conv2_channels'))
    hp.Fixed('conv3_channels',best.get('conv3_channels'))
    hp.Float('L2 coeff.',10**-10,10**0,sampling='log')
    hp.Float('alpha',10**-4,10**-2,sampling='log')
    hp.Choice('batchSize',[4,8,16,32,64,128])
    # Tune fitting hyperparameters
    modelBO=TuneModelHPs_BO(max_trials=50,
                            hyperparameters=hp,
                            directory=hpFolder,
                            project_name=str(n+1)+'_BO_Fitting',
                            tuner_id=str(n+1)+'_BO_Fitting')
    modelBO.search()
    # Get best hyperparameters
    best=modelBO.get_best_hyperparameters()[0].values

