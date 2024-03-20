#!/usr/bin/env python
# coding: utf-8

# 
# X_test = np.load('../dataset/X_test4.npy')
# 
# X_test = torch.tensor(X_test, dtype=torch.float)
# 
# y_test = np.load('../dataset/y_test4.npy')
# 
# y_test = torch.tensor(y_test, dtype=torch.float)
# 

# In[ ]:





# In[2]:


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from math import nan
from loading_data import CustomData
from loading_data import DeviceDataLoader
from neural_network import NeuralNetwork
from utils import get_default_device
from utils import to_device
from utils import evaluate_raw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pyranges as pr
from utils import load_model
from utils import get_aucs
from utils import format_4AUCplot
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8


# In[3]:


X_test = np.load('../dataset/X_test.npy')
# 
X_test = torch.tensor(X_test, dtype=torch.float)
# 
y_test = np.load('../dataset/y_test.npy')
# 
y_test = torch.tensor(y_test, dtype=torch.float)



# In[ ]:


# get into torch
# get_loader
batch_size= 128
device = get_default_device()
test_data = CustomData(X_test, y_test)
test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)
test_loader = DeviceDataLoader(test_loader, device)


# In[ ]:





# In[ ]:


params = np.load('../saved_models/params_2tis_2057.npy', allow_pickle='TRUE').item() #change here
model = NeuralNetwork(4,
                          params["h"],
                          params["f"],
                          2,
                          params["fcs"],
                          params["p"],
                          params["mha_p"])

device = get_default_device()
to_device(model, device)
model.load_state_dict(torch.load("../saved_models/model.npy")) #change here
model.eval()



# In[ ]:


def evaluate_raw(model, val_loader):
    """Raw model output"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return outputs


# In[ ]:


device = get_default_device()
outputs = evaluate_raw(model, test_loader)


# In[ ]:




# In[ ]:




# # not working
# cpu_outputs = []
# for out in outputs:
#     cpu_out = {}
#     for k, v in out.items():
#         cpu_out[k] = v.detach().cpu()
#     cpu_outputs.append(cpu_out)
# 

# In[ ]:

for x in outputs:
    for key in x.keys():
        if isinstance(x[key], torch.Tensor):
            x[key] = x[key].detach().cpu()



# In[ ]:

headout_list = [x['head_out'] for x in outputs if 'head_out' in x.keys()] 
heady_list = [x['head_y'] for x in outputs if 'head_y' in x.keys()] 
testisout_list = [x['testis_out'] for x in outputs if 'testis_out' in x.keys()] 
testisy_list = [x['testis_y'] for x in outputs if 'testis_y' in x.keys()] 



# In[ ]:

headout_tensor = torch.cat(headout_list)
heady_tensor = torch.cat(heady_list)
testisout_tensor = torch.cat(testisout_list)
testisy_tensor = torch.cat(testisy_list)

print('head tissue')


# roc and pr 
fpr, tpr, thresholds = metrics.roc_curve(heady_tensor.numpy(), headout_tensor.numpy())
pr, re, thresholds = metrics.precision_recall_curve(heady_tensor.numpy(), headout_tensor.numpy())
# calculate
roc_auc = metrics.auc(fpr, tpr)
pr_auc = metrics.auc(re, pr)

result_roc = []
result_pr = []

df1 = pd.DataFrame({'True positive rate': tpr, 'False positive rate': fpr})
result_roc.append(df1)
print(df1)

df2 = pd.DataFrame({'Precision': pr, 'Recall': re})
result_pr.append(df2)
print(df2)


print('head roc-auc is: ' + str(roc_auc))
print('head pr-auc is: ' + str(pr_auc))



# In[ ]:

print('testis tissue')
# roc and pr 
fpr2, tpr2, thresholds = metrics.roc_curve(testisy_tensor.numpy(), testisout_tensor.numpy())
pr2, re2, thresholds = metrics.precision_recall_curve(testisy_tensor.numpy(), testisout_tensor.numpy())
# calculate
roc_auc2 = metrics.auc(fpr2, tpr2)
pr_auc2 = metrics.auc(re2, pr2)

result_roc2 = []
result_pr2 = []


df3 = pd.DataFrame({'True positive rate': tpr2, 'False positive rate': fpr2})
result_roc2.append(df3)
print(df3)


df4 = pd.DataFrame({'Precision': pr2, 'Recall': re2})
result_pr2.append(df4)
print(df4)


print('testis roc-auc is: ' + str(roc_auc2))
print('testis pr-auc is: ' + str(pr_auc2))


# In[ ]:

f, ax = plt.subplots(2, 2, figsize=(5, 5))

sns.lineplot(data=df1,  
                       x="False positive rate", 
                       y="True positive rate",
                       palette="Dark2", 
                       linewidth=2, 
                       ax=ax[0, 0])

ax[0, 0].set_title('Head', fontsize=8)
ax[0, 0].text(0.6, 0.1, 'auc=' + str(round(roc_auc, 2)), fontsize=9)
ax[0, 0].plot([0, 1], [0, 1], linestyle='dotted', color='k')



sns.lineplot(data=df3,  
                       x="False positive rate", 
                       y="True positive rate",
                       palette="Dark2", 
                       linewidth=2, 
                       ax=ax[0, 1])
                       
ax[0, 1].set_title('Testis', fontsize=8)
ax[0, 1].text(0.6, 0.1, 'auc=' + str(round(roc_auc2, 2)), fontsize=9)
ax[0, 1].plot([0, 1], [0, 1], linestyle='dotted', color='k')



sns.lineplot(data=df2,  
                       x="Recall", 
                       y="Precision", 
                       palette="Dark2", 
                       linewidth=2, 
                       ax=ax[1, 0])
                       
ax[1, 0].text(0.1, 0.3, 'auc=' + str(round(pr_auc, 2)), fontsize=9)
ax[1, 0].set_title('Head', fontsize=8)



sns.lineplot(data=df4,  
                       x="Recall", 
                       y="Precision",
                       palette="Dark2", 
                       linewidth=2, 
                       ax=ax[1, 1])
                       
ax[1, 1].text(0.1, 0.1, 'auc=' + str(round(pr_auc2, 2)), fontsize=9)
ax[1, 1].set_title('Testis', fontsize=8)


plt.ylim([-0.05, 1.05])



plt.legend(loc='lower left')


plt.tight_layout()
plt.show()
f.savefig('../plot/Fig.pdf', dpi = 400 , bbox_inches='tight')





# In[ ]:




# In[ ]:



