#%%
import argparse, sys, os, numpy as np, torch, random
from pathlib import Path
from sklearn import svm
from matplotlib.pyplot import axis
from utils import load_data, plot_training

from model.model import train_model_CV, evauation
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Deforestation Detector')

  parser.add_argument('-mode', metavar='mode', help='Train or Evaluation')
  parser.add_argument('-model', metavar='model', help='model to encode')
  parser.add_argument('-output', metavar='output', help='output directory')
  parser.add_argument('-wp', metavar='wp', help='Weights path')
  parser.add_argument('-dt', metavar='dt', default='data', help='Data')
  parser.add_argument('-ep', metavar='ep', type=int, default=2, help='Epoches to train')
  parser.add_argument('-bs', metavar='bs', type=int, default=4, help='Batch Size')
  parser.add_argument('-il', metavar='il', type=int, default=64, help='Intermediate Layer Size')
  parser.add_argument('-lr', metavar='lr', type=float, default=1e-3, help='Learning Rate')

  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  mode = parameters.mode
  model = parameters.model
  wp = parameters.wp
  output = parameters.output
  dt = parameters.dt
  ep = parameters.ep
  bs = parameters.bs
  interm_layer_size = parameters.il
  lr = parameters.lr
  
  if mode == 'train':

    Path(output).mkdir(parents=True, exist_ok=True)
    data = load_data(dt, 'train', True)

    history = train_model_CV(model, data, splits = 5, epoches = ep, batch_size = bs, interm_layer_size = interm_layer_size, lr = lr, output=output)
    plot_training(history, model, output, 'loss')

  if mode == 'eval':

    data = load_data(dt, 'test', False)
    evauation(data=data, model=model, splits=5, output=output, wp=wp)

