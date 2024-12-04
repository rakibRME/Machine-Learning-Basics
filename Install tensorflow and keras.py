import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense,Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import os
import warnings
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
warnings.filterwarnings('ignore')

train_data = pd.read_csv("D:\ML-Data-Sets\train_Human Activity Recognition With Neural Networks.csv")
test_data = pd.read_csv("D:\ML-Data-Sets\test_Human Activity Recognition With Neural Networks.csv")

print(f'Shape of train data is: {train_data.shape}\nShape of test data is: {test_data.shape}')