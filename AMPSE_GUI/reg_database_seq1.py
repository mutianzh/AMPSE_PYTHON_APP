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



def seq1_filtering(ds):
    
    ds=ds.dropna()
#    add remove negative datasets
    ds=ds.reset_index(drop=True)
    

    return ds

class SEQ1_model(Model):
  def __init__(self):
    super(SEQ1_model, self).__init__()
    self.d1 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 11)
    self.d2 = Dense(512, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d3 = Dense(256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid')
    self.d4 = Dense(4, activation='linear')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)
