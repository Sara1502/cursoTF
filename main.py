import tensorflow as tf
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('arrhythmia.data', header=None)

data  = df[[0,1,2,3,4,5]]
data.columns = ['age', 'sex', 'height', 'weight', 'QRS duration', 'P-R interval']

plt.rcParams['figure.figsize'] = [15, 15]
data.hist();

scatter_matrix(data);