#%%
import torchvision.models as models
import torch

from model.model import ZeroDeforestationDataset
from utils import load_data
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

#%%

model_ft = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
num_ftrs = model_ft.classifier[-1].in_features
model_ft.classifier[-1] = torch.nn.Linear(num_ftrs, 3)
model_ft = model_ft.to('cuda')

# for now, a simple optimizer
optimizer_ft = torch.optim.Adagrad(model_ft.parameters(), lr=0.001)#, momentum=0.9)
# %%

data = load_data('data', 'train', True)

####
loss_ft = torch.nn.CrossEntropyLoss()
from collections import Counter

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 23)

for i, (train_index, test_index) in enumerate(skf.split(data['images'], data['labels'])):  
  print(Counter(data['labels'][train_index]))
  
  train_loader = DataLoader( ZeroDeforestationDataset( {'images':data['images'][train_index], 'labels':data['labels'][train_index]}, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])),
     batch_size=2, 
     shuffle=True)
  
  
  torch.cuda.empty_cache()      

  ##epoches
  for i in range(2):

    for j, data_batch in enumerate(train_loader, 0):

      labels = data_batch['labels'].to('cuda')     
      
      optimizer_ft.zero_grad()
      outputs = model_ft(data_batch['images'].to('cuda'))
      loss = loss_ft(outputs, labels)
   
      loss.backward()
      optimizer_ft.step()
      print('step')



# %%
# %%
