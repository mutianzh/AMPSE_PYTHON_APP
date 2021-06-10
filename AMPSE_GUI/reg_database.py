#==================================================================
#*******************  Initialization  *****************************
#==================================================================
# import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
# sys.path.insert(0,'/home/mohsen/PYTHON_PHD/GlobalLibrary')


from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow.keras.initializers as init
from tensorflow import concat
import tensorflow as tf
import numpy as np

#==================================================================
#*******************  dataset filtering  **************************
#==================================================================
def comp_filtering(ds):
    
    ds=ds.dropna()                                  #dropout Nan datas in datasets
    ds=ds.reset_index(drop=True)
    

    return ds

def sardac_filtering(ds):
    
    ds=ds.dropna()
    #Convert 'madc' in dataset back to bit number instead 2^bit
    y = np.array(ds.iloc[1:, 1].values,dtype='float64')
    newy=np.log2(y)
    ds.iloc[1:,1] = newy
    ds=ds.reset_index(drop=True)
    
    return ds


def seq1_filtering(ds):
    
    ds=ds.dropna()                                             #dropout Nan datas in datasets
    y = np.array(ds.iloc[1:, 10].values,dtype='float64')
    newy=np.log2(y)
    ds.iloc[1:,10] = newy
    y = np.array(ds.iloc[1:, 13].values,dtype='float64')
    remfilt = [ d>0 for d in y]                               #    remove negative/false datasets
    remfilt =[True]+remfilt
    ds = ds[remfilt]
    ds=ds.reset_index(drop=True)

    return ds

def seq2_filtering(ds):
    
    ds=ds.dropna()                                       #dropout Nan datas in datasets
    ds=ds.reset_index(drop=True)

    return ds



#==================================================================
#*******************  Defining Regression Model  ******************
#==================================================================

#----------Comparator's Regression Model---------- 
class COMP_model(Model):
  def __init__(self):
    super(COMP_model, self).__init__()
    #layers        #number of neurons                                        #type of neuron        #number of inputs
    self.d1 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 14)
    self.d2 = Dense(512, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d3 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d4 = Dense(5, activation='linear') 
                    #number of outputs
    
    # when noise is avaliable
    # self.d4 = Dense(6, activation='linear')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)


#----------SARDAC's Regression Model---------- 
class SARDAC_model(Model):
  def __init__(self):
    super(SARDAC_model, self).__init__()
    #layers        #number of neurons                                        #type of neuron        #number of inputs
    self.d1 = Dense(64, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 6)
    self.d2 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d3 = Dense(64, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d4 = Dense(4, activation='linear')
                    #number of outputs

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)



#----------Seq1's Regression Model---------- 
class SEQ1_model(Model):
  def __init__(self):
    super(SEQ1_model, self).__init__()
    
    #layers        #number of neurons                                        #type of neuron        #number of inputs
    self.d1 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 11)
    self.d2 = Dense(512, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d3 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d4 = Dense(4, activation='linear')
                    #number of outputs
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)

#----------Seq2's Regression Model---------- 
class SEQ2_model(Model):
  def __init__(self):
    super(SEQ2_model, self).__init__()
    #layers        #number of neurons                                        #type of neuron        #number of inputs
    self.d1 = Dense(50, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 2)
    self.d2 = Dense(50, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d3 = Dense(6, activation='linear')
                    #number of outputs

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)