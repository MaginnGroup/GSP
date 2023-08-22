# -*- coding: utf-8 -*-
"""
7_CNN_Test.py plots tests GCN-generated sigma profiles as input to CNNs.

Sections:
    . Imports
    . Configuration
    . Main Functions
        . normalize()
        . graphDataset()
        . GCN_Model_SP()
        . composite()
        . generateGCN()
    . Timer_Start
    . Main Script (S_25)
    . Main Script (BP)
    
Last edit: 2023-07-27
Author: Dinis Abranches
"""

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
# Imports
# =============================================================================

# General
import os
import pickle

# Specific
import numpy
import pandas
import tensorflow
from tensorflow.keras import backend
import spektral
from sklearn import preprocessing
from spektral import transforms
from matplotlib import pyplot as plt
from sklearn import metrics
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# =============================================================================
# Main Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, Array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . LogStand - standardization on the log of the variable,
              (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
              zero; uses a small buffer, (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Defalt: 'Standardization'
    reverse : bool
        Wether to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1: inputArray=inputArray.reshape((-1,1))
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        skScaler=preprocessing.StandardScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        inputArray=skScaler.inverse_transform(inputArray) # Rescale the data back to its original distribution
        # Check method
        if method=='LogStand': inputArray=numpy.exp(inputArray)
        elif method=='Log+bStand': inputArray=numpy.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

class graphDataset(spektral.data.Dataset):
    """
    spektral.data.Dataset object containing the spektral graph objects of the
    molecules in any given dataset.
    """

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
                                          shuffle=False)
    loaderVal=spektral.data.BatchLoader(graphSet_Val,shuffle=False)
    # Define early stopping
    earlyStop=tensorflow.keras.callbacks.EarlyStopping(patience=100,mode='min',
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
# Timer_Start
# =============================================================================

import time
start_time=time.time()

# =============================================================================
# Main Script (S_25)
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
GCN,__=generateGCN(hp,graphSet_Train,graphSet_Test,verbose=1)
GCN.set_weights(weights)
# Apply filters to adjacency matrices
graphSet_Train.apply(transforms.GCNFilter())
graphSet_Test.apply(transforms.GCNFilter())
# Predict Sigma Profiles
loaderTrain=spektral.data.BatchLoader(graphSet_Train,shuffle=False)
SP_Train=GCN.predict(loaderTrain.load(),steps=loaderTrain.steps_per_epoch)
loaderTest=spektral.data.BatchLoader(graphSet_Test,shuffle=False)
SP_Test=GCN.predict(loaderTest.load(),steps=loaderTest.steps_per_epoch)
# Load solubility database
databasePath=os.path.join(databasesFolder,'S_25_mlDatabase_Original.csv')
mlDatabase=pandas.read_csv(databasePath,dtype=str)
# Load CNN model
modelPath=os.path.join(modelFolder,'S_25_mlDatabase.h5')
CNN=tensorflow.keras.models.load_model(modelPath)
# Load CNN normalization weights
scaler_X=pickle.load(open(os.path.join(modelFolder,
                                       'S_25_mlDatabase_X_Scaler.pkl'),'rb'))
scaler_Y=pickle.load(open(os.path.join(modelFolder,
                                       'S_25_mlDatabase_Y_Scaler.pkl'),'rb'))
# Normalize Features
SP_Train=normalize(SP_Train,method='Log+bStand',skScaler=scaler_X)[0]
SP_Test=normalize(SP_Test,method='Log+bStand',skScaler=scaler_X)[0]
SP_Train=SP_Train.reshape(SP_Train.shape[0],SP_Train.shape[1],1)
SP_Test=SP_Test.reshape(SP_Test.shape[0],SP_Test.shape[1],1)
S_Train_Predicted=CNN.predict(SP_Train)
S_Test_Predicted=CNN.predict(SP_Test)
# Initialize solubility containers
S_Train_Exp=[]
S_Test_Exp=[]
# Loop over training set
deleteMask=[]
for n in range(len(graphSet_Train)):
    # Get entry
    entry=graphSet_Train[n]
    # Get VT2005 index
    index=entry.VT2005Index
    # Get experimental solubility and append
    aux=mlDatabase[mlDatabase['Index']==str(index)]
    if aux.empty:
        deleteMask.append(n)
    else:
        S_Train_Exp.append(float(aux.iloc[0,3]))
S_Train_Predicted=numpy.delete(S_Train_Predicted,deleteMask,axis=0)
# Loop over testing set
deleteMask=[]
for n in range(len(graphSet_Test)):
    # Get entry
    entry=graphSet_Test[n]
    # Get VT2005 index
    index=entry.VT2005Index
    # Get experimental solubility and append
    aux=mlDatabase[mlDatabase['Index']==str(index)]
    if aux.empty:
        deleteMask.append(n)
    else:
        S_Test_Exp.append(float(aux.iloc[0,3]))
S_Test_Predicted=numpy.delete(S_Test_Predicted,deleteMask,axis=0)
# Unnormalize
S_Train_Predicted=normalize(S_Train_Predicted,method='LogStand',
                            skScaler=scaler_Y,reverse=True)[0]
S_Test_Predicted=normalize(S_Test_Predicted,method='LogStand',
                           skScaler=scaler_Y,reverse=True)[0]

# Plot Configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 13
plt.rcParams["savefig.pad_inches"] = 0.02

caption='Solubility (25 ºC) /g/kg'
v=[10**-7,10**5]
plt.loglog(S_Train_Exp,S_Train_Predicted,'.k',label='Training Set')
plt.loglog(S_Test_Exp,S_Test_Predicted,'.b',label='Testing Set')
plt.loglog(v,v,'-k',linewidth=1)
plt.xlim(v)
plt.ylim(v)
plt.legend(loc='upper left')
# Get R^2
R2_Train=metrics.r2_score(numpy.log(S_Train_Exp),numpy.log(S_Train_Predicted))
R2_Test=metrics.r2_score(numpy.log(S_Test_Exp),numpy.log(S_Test_Predicted))
# Print R^2 in plot
plt.text(0.97,0.17,'$R^2=$'+'{:.2f}'.format(R2_Train),
         horizontalalignment='right',transform=plt.gca().transAxes,c='k')
plt.text(0.97,0.10,'$R^2=$'+'{:.2f}'.format(R2_Test),
         horizontalalignment='right',transform=plt.gca().transAxes,c='b')
plt.xlabel('Exp. '+caption)
plt.ylabel('Pred. '+caption)
plt.show()

# =============================================================================
# Main Script (BP)
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
GCN,__=generateGCN(hp,graphSet_Train,graphSet_Test,verbose=1)
GCN.set_weights(weights)
# Apply filters to adjacency matrices
graphSet_Train.apply(transforms.GCNFilter())
graphSet_Test.apply(transforms.GCNFilter())
# Predict Sigma Profiles
loaderTrain=spektral.data.BatchLoader(graphSet_Train,shuffle=False)
SP_Train=GCN.predict(loaderTrain.load(),steps=loaderTrain.steps_per_epoch)
loaderTest=spektral.data.BatchLoader(graphSet_Test,shuffle=False)
SP_Test=GCN.predict(loaderTest.load(),steps=loaderTest.steps_per_epoch)
# Load solubility database
databasePath=os.path.join(databasesFolder,'BP_mlDatabase_Original.csv')
mlDatabase=pandas.read_csv(databasePath,dtype=str)
# Load CNN model
modelPath=os.path.join(modelFolder,'BP_mlDatabase.h5')
CNN=tensorflow.keras.models.load_model(modelPath)
# Load CNN normalization weights
scaler_X=pickle.load(open(os.path.join(modelFolder,
                                       'BP_mlDatabase_X_Scaler.pkl'),'rb'))
scaler_Y=pickle.load(open(os.path.join(modelFolder,
                                       'BP_mlDatabase_Y_Scaler.pkl'),'rb'))
# Normalize Features
SP_Train=normalize(SP_Train,method='Log+bStand',skScaler=scaler_X)[0]
SP_Test=normalize(SP_Test,method='Log+bStand',skScaler=scaler_X)[0]
SP_Train=SP_Train.reshape(SP_Train.shape[0],SP_Train.shape[1],1)
SP_Test=SP_Test.reshape(SP_Test.shape[0],SP_Test.shape[1],1)
BP_Train_Predicted=CNN.predict(SP_Train)
BP_Test_Predicted=CNN.predict(SP_Test)
# Initialize solubility containers
BP_Train_Exp=[]
BP_Test_Exp=[]
# Loop over training set
deleteMask=[]
for n in range(len(graphSet_Train)):
    # Get entry
    entry=graphSet_Train[n]
    # Get VT2005 index
    index=entry.VT2005Index
    # Get experimental solubility and append
    aux=mlDatabase[mlDatabase['Index']==str(index)]
    if aux.empty:
        deleteMask.append(n)
    else:
        BP_Train_Exp.append(float(aux.iloc[0,3]))
BP_Train_Predicted=numpy.delete(BP_Train_Predicted,deleteMask,axis=0)
# Loop over testing set
deleteMask=[]
for n in range(len(graphSet_Test)):
    # Get entry
    entry=graphSet_Test[n]
    # Get VT2005 index
    index=entry.VT2005Index
    # Get experimental solubility and append
    aux=mlDatabase[mlDatabase['Index']==str(index)]
    if aux.empty:
        deleteMask.append(n)
    else:
        BP_Test_Exp.append(float(aux.iloc[0,3]))
BP_Test_Predicted=numpy.delete(BP_Test_Predicted,deleteMask,axis=0)
# Unnormalize
BP_Train_Predicted=normalize(BP_Train_Predicted,method='Standardization',
                             skScaler=scaler_Y,reverse=True)[0]
BP_Test_Predicted=normalize(BP_Test_Predicted,method='Standardization',
                            skScaler=scaler_Y,reverse=True)[0]

# Plot Configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 13
plt.rcParams["savefig.pad_inches"] = 0.02

VT_index=[]
for n in range(1):
    index=numpy.array(abs(BP_Test_Predicted.reshape(-1,)-BP_Test_Exp)).argmax()
    aux=mlDatabase.loc[mlDatabase['Boiling Temp. /ºC'].astype('float')\
                        ==BP_Test_Exp[index]]
    VT_index.append(aux.iloc[0,0])
    BP_Test_Exp.pop(index)
    BP_Test_Predicted=numpy.delete(BP_Test_Predicted,index)

caption='Boiling Temperature /ºC'
v=[-500,1000]
plt.plot(BP_Train_Exp,BP_Train_Predicted,'.k',label='Training Set')
plt.plot(BP_Test_Exp,BP_Test_Predicted,'.b',label='Testing Set')
plt.plot(v,v,'-k',linewidth=1)
plt.xlim(v)
plt.ylim(v)
plt.legend(loc='upper left')
# Get R^2
R2_Train=metrics.r2_score(BP_Train_Exp,BP_Train_Predicted)
R2_Test=metrics.r2_score(BP_Test_Exp,BP_Test_Predicted)
# Print R^2 in plot
plt.text(0.97,0.17,'$R^2=$'+'{:.2f}'.format(R2_Train),
         horizontalalignment='right',transform=plt.gca().transAxes,c='k')
plt.text(0.97,0.10,'$R^2=$'+'{:.2f}'.format(R2_Test),
         horizontalalignment='right',transform=plt.gca().transAxes,c='b')
plt.xlabel('Exp. '+caption)
plt.ylabel('Pred. '+caption)
plt.show()

# =============================================================================
# Timer_End
# =============================================================================

print("--- %s seconds ---" % (time.time() - start_time))
print('Total molecules in S_25: '+str(len(S_Train_Exp)+len(S_Test_Exp)))
print('Total molecules in BP: '+str(len(BP_Train_Exp)+len(BP_Test_Exp)))