# Import Modules
import time
import pandas as pd
from glob import glob
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

# Import Custom Modules
from dataset import CustomDataset, getDataLoader
from bert import littleBERT

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing
dat_list = sorted(glob('./data/*.xls'), reverse=True)
for i, data in enumerate(dat_list):
    if i == 0:
        dat = pd.read_excel(data)
    else:
        dat = pd.concat([dat, pd.read_excel(data)])
dat = dat.iloc[::-1]
dat.index = range(len(dat))
dat['month'] = dat['date'].map(lambda x: int(x[5:7]))
dat['day'] = dat['date'].map(lambda x: int(x[8:10]))
dat['time'] = dat['date'].map(lambda x: int(x[-2:]))
dat = dat.drop(columns = 'date')
dat = dat.dropna()
x1 = list()
x2 = list()
y = list()
for t in tqdm(range(5, len(dat) - 8)):
    y.append(dat.iloc[t+8]['pm10'])
    # x1.append(dat.iloc[t][['Pm2.5', 'ozone', 'nitrogen', 'carbon', 'sulfur']].tolist())
    x1.append(dat['Pm2.5'].loc[t-5:t-1].tolist())
    x2.append(dat['pm10'].loc[t-5:t-1].tolist())

# CustomDataset and DataLoader
train_dataset = CustomDataset(x1, x2, y)
train_loader = getDataLoader(train_dataset, 4, True)

model = littleBERT(n_head=12, d_model=768, d_embedding=768, n_layers=24,
                   dim_feedforward=768 * 4, dropout=0.1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.L1Loss()
torch_utils.clip_grad_norm_(model.parameters(), 5)
start_time = time.time()
for epoch in range(10):
    print('Epoch {}/{}'.format(epoch + 1, 5))
    print('-' * 100)
    model.train()
    running_loss = 0.0
    
    for src1, src2, trg in tqdm(train_loader):
        src1 = src1.to(device)
        src2 = src2.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled('train' == 'train'):
            outputs = model(src1, src2, device)
            output_last_token = outputs[5].squeeze(1)
            loss = criterion(output_last_token, trg)
            loss.backward()
            optimizer.step()

        # Statistics
        running_loss += loss.item() * src1.size(0)

    # Epoch loss calculate
    epoch_loss = running_loss / len(train_dataset)
    scheduler.step(epoch)
    spend_time = (time.time() - start_time) / 60
    print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}min'.format('train', epoch_loss, 0, spend_time))