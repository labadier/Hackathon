#%%
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import os
# 

def load_data(path, mode, labeled):

  df = pd.read_csv(os.path.join(path, f'{mode}.csv'), usecols = ['label', 'example_path'] if labeled else ['example_path']).to_numpy()
  
  labels, images = [], []

  if labeled:

    for i in range(len(df)):
      images += [os.path.join(path, df[i,1])]
      labels += [df[i,0]]
  else:
    for i in range(len(df)):
      images += [os.path.join(path, df[i,0])]
  
  if labeled == True:
    return {'images': np.array(images), 'labels': np.array(labels)}
  return {'images': np.array(images)}



def plot_training(history, model, output, measure='loss'):
    
    plt.plot(history[measure])
    plt.plot(history['dev_' + measure])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel(measure)
    plt.xlabel('Epoch')
    if measure == 'loss':
        x = np.argmin(history['dev_loss'])
    else: x = np.argmax(history['dev_f1'])

    plt.plot(x,history['dev_' + measure][x], marker="o", color="red")
    plt.savefig(os.path.join(output, f'train_history_{model}.png'))
