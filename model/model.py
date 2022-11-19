import numpy as np, pandas as pd, os, torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from pathlib import Path
import cv2
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import f1_score
from statistics import mode
import json

path = 'data'
mode = 'train'

mode = os.path.join(path, f'{mode}.csv')

class ZeroDeforestationDataset(Dataset):
  def __init__(self, data, transform=None, target_transform=None):

    self.img_labels = data['labels']
    self.img_dir = data['images']
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    image = cv2.imread(self.img_dir[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = self.img_labels[idx]

    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return {'images':image, 'labels':label, 'index':self.img_dir[idx].split('/')[-1].split('.')[0]}

def train_model_CV(model_name, data, splits, epoches, batch_size, interm_layer_size, lr, output):
  
  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  for i, (train_index, test_index) in enumerate(skf.split(data['images'], data['labels'])):  
    
    history.append({'loss': [], 'f1':[], 'dev_loss': [], 'dev_f1': []})
   
    if model_name == 'vgg16':
      model_ft = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
      num_ftrs = model_ft.classifier[-1].in_features
      model_ft.classifier[-1] = torch.nn.Linear(num_ftrs, 3)
      model_ft = model_ft.to('cuda')
      optimizer_ft = torch.optim.Adagrad(model_ft.parameters(), lr=lr)

    if model_name == 'resnet':
      model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc =  torch.nn.Linear(num_ftrs, 2)
      model_ft = model_ft.to('cuda')
      optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=lr)


    
    loss_ft = torch.nn.CrossEntropyLoss()

    
    train_loader = DataLoader( ZeroDeforestationDataset( {'images':data['images'][train_index], 'labels':data['labels'][train_index]}, 
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ])),
                  batch_size=batch_size, 
                  shuffle=True)
     
    dev_loader = DataLoader( ZeroDeforestationDataset( {'images':data['images'][test_index], 'labels':data['labels'][test_index]}, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])),
                batch_size=batch_size, 
                shuffle=True)

    history.append(train_model(model = model_ft, optimizer = optimizer_ft, loss_function= loss_ft, train_loader= train_loader,
                    dev_loader= dev_loader, epoches = epoches, output = os.path.join(output, f'{model_name}_{i+1}.pt')))
    print(f'Training Finished Split: {i+1}')
    
    del train_loader
    del model_ft
    del dev_loader

  return history

def compute_f1(ground_truth, predictions):
  out = torch.max(predictions, 1).indices.detach().cpu().numpy()
  return f1_score(out, ground_truth.detach().cpu().numpy(), average='macro')

def train_model(model, optimizer, loss_function, train_loader, dev_loader, epoches, output):

  eloss, ef1, edev_loss, edev_f1= [], [], [], []
  best_f1 = None

  for epoch in range(epoches):
    running_loss = 0.0
    f1 = 0

    model.train()

    iter = tqdm(enumerate(train_loader, 0))
    iter.set_description(f'Epoch: {epoch}')
    for j, data_batch in iter:

      torch.cuda.empty_cache()         
      labels = data_batch['labels'].to('cuda')     
      
      optimizer.zero_grad()
      outputs = model(data_batch['images'].to('cuda'))
      loss = loss_function(outputs, labels)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          f1 = compute_f1(labels, outputs)
          running_loss = loss.item()
        else: 
          f1 = (f1 + compute_f1(labels, outputs))/2.0
          running_loss = (running_loss + loss.item())/2.0

      iter.set_postfix({'f1_train':f1, 'loss_train':running_loss}) 

    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      for k, data_batch_dev in enumerate(dev_loader, 0):
        torch.cuda.empty_cache() 

        labels = data_batch_dev['labels'].to('cuda')  
        dev_out = model(data_batch_dev['images'].to('cuda'))

        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)

        dev_loss = loss_function(out, log).item()
        dev_f1 = compute_f1(log, out)
        ef1.append(f1)
        edev_loss.append(dev_loss)
        edev_f1.append(dev_f1) 

    if best_f1 is None or best_f1 < dev_f1:
      torch.save(model.state_dict(), output) 
      best_f1 = dev_f1

    print(f'f1_train:{f1}, loss_train:{running_loss}, f1_dev:{dev_f1}, loss_dev:{dev_loss}') 
    

  return {'loss': eloss, 'f1': ef1, 'dev_loss': edev_loss, 'dev_f1': edev_f1}

    
def evauation(data, model, splits, output, wp):

  if model == 'vgg16':
    model_ft = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = torch.nn.Linear(num_ftrs, 3)

  dev_loader = DataLoader( ZeroDeforestationDataset( {'images':data['images'], 'labels':np.zeros(len(data['images'],))}, 
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ])),
              batch_size=64)

  dic = {}

  itr = tqdm(range(splits))
  with torch.no_grad():
    for i in itr:
      
      model_ft.load_state_dict(torch.load(f'{os.path.join(wp, model)}_{i+1}.pt', map_location='cuda'))
      model_ft = model_ft.to('cuda')

      model.eval()
      for k, data_batch_dev in enumerate(dev_loader, 0):

        labels = data_batch_dev['index']
        torch.cuda.empty_cache() 
        dev_out = model_ft(data_batch_dev['images'].to('cuda'))

        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)
      
      if not i:
        for index, pred in zip(labels, dev_out):
          dic[index] = [torch.argmax(pred).item()]
      else:
        for index, pred in zip(labels, dev_out):
          dic[index] += [torch.argmax(pred).item()]

  for i in dic:
    dic[i] = mode(dic[i])

  ans = {'target': dic}
  with open(os.path.join(output, 'predictions.json'), 'w') as fp:
    json.dump(ans, fp)

  ans = pd.DataFrame(dic)
  ans.to_csv(os.path.join(output, 'predictions.csv'))



