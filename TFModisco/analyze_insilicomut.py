#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pyranges as pr
import pandas as pd
import pyranges.genomicfeatures as gf
from Bio import SeqIO
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
import pyBigWig
import scipy.special as sp
import sklearn.metrics as metrics
#%matplotlib inline


# ### First block of cells is initial processing of insilico results ... not necessary to run every time

# In[5]:


insil_testis = pd.read_csv('../run_model/insilico_testismel_3L.csv')
#insil_testis = insil_testis[insil_testis.ref != 'nan'].reset_index()

insil_head = pd.read_csv('../run_model/insilico_headmel_3L.csv')

#insil_head = insil_head[insil_head.ref != 'nan'].reset_index()


# In[ ]:


conv_dict = {0.0: 'a', 1.0: 't', 2.0: 'c', 3.0: 'g'}
insil_head['ref'] = insil_head['ref'].map(conv_dict)
insil_testis['ref'] = insil_testis['ref'].map(conv_dict)


# In[ ]:


insil_head


# In[11]:


test_coor = pd.read_csv('../run_model/test_coors_3L_new')



# In[13]:


def process_sample(df, sample_ind, coor):
    sl = df[df.sample_ind == sample_ind].copy()
    try: 
        ref_base = sl.ref.iloc[0]
        ref_val = sl[ref_base].iloc[0]
    except:
        ref_val = np.nan
    sl['importance_logit'] = sp.logit(ref_val) - sp.logit(sl[['a', 't', 'c', 'g']]).mean(axis = 1)
    sl['importance_raw'] = ref_val - sl[['a', 't', 'c', 'g']].mean(axis = 1)
    sl['ref_val'] = ref_val
    sl['max_dev'] = sl[['a', 't', 'c', 'g']].max(axis=1) - ref_val
    sl['min_dev'] = sl[['a', 't', 'c', 'g']].min(axis=1) - ref_val
    
    # attach coordinate info
    coor = coor.copy()
    coor_slice = coor.iloc[sample_ind]
    sl['chr_pos'] = sl.pos + coor_slice.Start
    sl['label_head'] = coor_slice.label_head
    sl['label_testis'] = coor_slice.label_testis
    return sl


# In[14]:


def add_coorlabs(insil, test_coor):
    test_coor = test_coor.copy()
    proc_insil_l = []
    for i in range(np.max(insil.sample_ind)+1):
        proc_insil_l.append(process_sample(insil, i, test_coor))
    proc_insil = pd.concat(proc_insil_l)
    return proc_insil



# In[15]:


proc_insil_head = add_coorlabs(insil_head, test_coor)
proc_insil_testis = add_coorlabs(insil_testis, test_coor)


# In[3]:


bw = pyBigWig.open('/rugpfs/fs0/zhao_lab/scratch/xwu05/detect_peaks_scripts/run_model/dm6.phyloP27way.bw')


# In[1]:


get_ipython().system(' df -h')


# In[2]:


phyp = []
for index, row in proc_insil_head.iterrows():
    try:
        phyp.append(bw.values("chr3L", int(row.chr_pos), int(row.chr_pos +1))[0])
    except:
        phyp.append(np.nan)


# In[ ]:


proc_insil_head['phyp'] = phyp
proc_insil_testis['phyp'] = phyp
proc_insil_head


# In[ ]:


#proc_insil_head = proc_insil_head.dropna(axis=0)
#proc_insil_testis = proc_insil_testis.dropna(axis=0)


proc_insil_head.to_csv('../run_model/proc_insil_head_logit_3L.csv')
proc_insil_testis.to_csv('../run_model/proc_insil_testis_logit_3L.csv')

