#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import modisco
import modisco.visualization
from modisco.visualization import viz_sequence
import pandas as pd
import scipy.special as sp
from Bio import SeqIO
import h5py
import modisco.util
from collections import Counter
import numpy as np

from modisco.visualization import viz_sequence
from matplotlib import pyplot as plt

import modisco.affinitymat.core

import modisco.cluster.phenograph.core

import modisco.cluster.phenograph.cluster

import modisco.cluster.core

import modisco.aggregator

from utils import load_testdata
from utils import load_model
from loading_data import get_testloader
import torch
from utils import get_default_device
from utils import to_device
from torch.utils.data import Dataset
from loading_data import CustomData
from loading_data import DeviceDataLoader
from neural_network import NeuralNetwork
from utils import evaluate_raw
import seaborn as sns
import sklearn.metrics as metrics

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8


# In[ ]:





# ## Run TF modisco

# In[2]:


chrs = {}
genome_file = "../genomes/dmel-all-chromosome-r6.41.fasta"
with open(genome_file, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        key = record.id.replace('Scf_', '')
        chrs[key] = record.seq


# In[3]:


chrs['3L']
def get_content(nuc, seq): #计算了一个序列中某个核苷酸的概率
    seq = str(seq).lower()
    totlength = 0
    nucn = 0
    for n in seq:
        if n in ['a','t','g','c']:
            totlength += 1
        if n == nuc:
            nucn += 1
    return nucn/totlength
p_a = round(get_content('a',chrs['3L'] ), 2)
p_t = round(get_content('t',chrs['3L'] ), 2)
p_g = round(get_content('g',chrs['3L'] ), 2)
p_c = round(get_content('c',chrs['3L'] ), 2)


# In[5]:


onehot_ref = np.load('../run_model/onehot_ref.npy')
hdf5_results = h5py.File("../run_model/multitask_results.hdf5","r")
metacluster_names = [
    x.decode("utf-8") for x in 
    list(hdf5_results["metaclustering_results"]
         ["all_metacluster_names"][:])]

all_patterns = []
#background = np.mean(onehot_ref, axis=(0,1))


consensus_motifs = dict()
# to convert 1-hots used by tf-modisco and personal code
conv_1hot = {0: 0, 1: 2, 2: 3, 3: 1}
with open("../run_model/meme_file_multitask_TRASH_3L.txt", "w") as file:
    file.write('MEME version 5\n')
    file.write('ALPHABET= ACGT\n')
    file.write('strands: +\n')
    file.write('Background letter frequencies\n')
    file.write('A ' + str(p_a) + 
               ' C '+ str(p_c) + 
               ' G '+ str(p_g) +
               ' T '+ str(p_t) + '\n')
    
    # columns are A C G T
    for metacluster_name in metacluster_names:
        print(metacluster_name)
        metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                       [metacluster_name])
        print("activity pattern:",metacluster_grp["activity_pattern"][:])
        all_pattern_names = [x.decode("utf-8") for x in 
                             list(metacluster_grp["seqlets_to_patterns_result"]
                                                 ["patterns"]["all_pattern_names"][:])]
        # init consensus motifs
        consensus_motifs[metacluster_name] = []
        if (len(all_pattern_names)==0):
            print("No motifs found for this activity pattern")
        for pattern_name in all_pattern_names:
            print(metacluster_name, pattern_name)
            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
            print("Task 0 hypothetical scores:")
            viz_sequence.plot_weights(pattern["head_hypothetical_contribs"]["fwd"])
            print("Task 0 actual importance scores:")
            viz_sequence.plot_weights(pattern["head_contrib_scores"]["fwd"])
            print("onehot, fwd and rev:")
            ppm = np.array(pattern["sequence"]["fwd"])
            file.write('MOTIF ' + metacluster_name + pattern_name  + '\n')
            file.write('letter-probability matrix: alength= 4 w= 10 E= 0\n')
            
            # add to consensus
            #print(consensus_motifs)
            consensus_motifs[metacluster_name].append([conv_1hot[i] for i in ppm.argmax(axis=1)])
            for j in range(10):
                file.write(str(ppm[j, 0].item()) + ' ' + 
                           str(ppm[j, 1].item()) + ' ' + 
                           str(ppm[j, 2].item())  + ' ' + 
                           str(ppm[j, 3].item()) +'\n')



# ## Insert TF-modisco motifs into test examples

# In[13]:


def get_modout(suffix, X_test, y_test):
    # suffix is model name suffix
    """ calculate model output for test data"""
    device = get_default_device()
    test_loader = get_testloader(X_test, y_test, device)
    model = load_model(suffix)
    outputs = evaluate_raw(model, test_loader)
    return outputs
def torchout_2df(outputs, tissue):
    '''convert torch outputs to dataframe'''
    tot_output = torch.cat([x[tissue + '_out'] for x in outputs])
    tot_y = torch.cat([x[tissue + '_y'] for x in outputs])
    return pd.DataFrame({'out': tot_output.cpu().numpy(), 'label': tot_y.cpu().numpy()})


# In[14]:


X_test, y_test = load_testdata('2tis_2057')
# copy used for mutation
X_test_mut = X_test.copy()
orig_out = get_modout("2tis_2057", X_test, y_test)


# ### Choose binarization threshold

# In[7]:


def get_threshold(df):
    '''maximized product of sensitivity and specificity'''
    fpr, tpr, thresholds = metrics.roc_curve(df.label,df.out)
    return thresholds[np.sqrt(tpr*(1-fpr)).argmax()] #threshold是maximized product of sensitivity and specificity


suffix = "2tis_2057"
X_val =  np.load('../run_model/X_val.npy')
y_val =  np.load('../run_model/y_val.npy')
outputs_val = get_modout("2tis_2057", X_val, y_val)
head_df = torchout_2df(outputs_val, "head")
testis_df = torchout_2df(outputs_val, "testis")

thresh_head = get_threshold(head_df)
thresh_testis = get_threshold(testis_df)


# In[8]:


import torch


# In[7]:


# need to double check this since it was done manually
motif_category = {'inc_head': ['metacluster_1', 'metacluster_4'], #增加表达
                  'dec_head': ['metacluster_0', ], #减少表达
                  'inc_testis': ['metacluster_1', 'metacluster_3'],
                  'dec_testis': ['metacluster_0', 'metacluster_2']}
def onehot_2nuc(onehot):
    if onehot == 0:
        out = 'A'
    elif onehot == 1:
        out = 'T'
    elif onehot == 2:
        out = 'C'
    elif onehot == 3:
        out = 'G'
        
    return out

def get_modout_mut(cat, tissue):
    '''calculate model output difference between 
    reference and sequences with TF-modisco
    motifs inserted
    '''
    out_df = pd.DataFrame({'out':[],  
                           'orig_out': [], 
                           'Sequence': []})
    metaclusters = motif_category[cat]
    for mcluster in metaclusters:
        for i in range(len(consensus_motifs[mcluster])):
        #for i in range(1):
            print(i)
            X_test_mut[:, :, np.arange(495, 505)] = 0 
            
            X_test_mut[:, consensus_motifs[mcluster][i], np.arange(495, 505)] = 1
            mutate_out = get_modout("2tis_2057", X_test_mut, y_test)
            mutate_out_df = torchout_2df(mutate_out, tissue).drop("label", axis=1)
            orig_out_df = torchout_2df(orig_out, tissue).rename(columns={'out': 'orig_out'}).drop("label", axis=1)

            #mutate_out_df['Difference'] = mutate_out_df.out - orig_out_df.out
            motif_string = [onehot_2nuc(n) for n in consensus_motifs[mcluster][i]]
            
            mutate_out_df['Sequence'] = ''.join(motif_string)
            mutate_out_df = mutate_out_df.merge(orig_out_df , left_index=True, right_index=True)
            out_df = pd.concat([out_df, mutate_out_df])
    return out_df


# In[10]:


inc_head_df = get_modout_mut('inc_head', "head")
inc_testis_df = get_modout_mut('inc_testis', "testis")
dec_head_df = get_modout_mut('dec_head', "head")
dec_testis_df = get_modout_mut('dec_testis', "testis")


# In[11]:


inc_head_df


# In[12]:


def top5_motifs(df, direction, thresh):
    
    df = df.copy()
    if direction == "increasing": #这个方向是高于threthold的
        df = df[df.orig_out < thresh]
        df = df[df.Sequence.isin(df.groupby('Sequence').out.apply(lambda x: sum(x>=thresh)/len(x)).nlargest(5).index)]
        out = df.groupby('Sequence').out.apply(lambda x: 1- sum(x >= thresh )/len(x)).reset_index()

    elif direction == "decreasing": #这个方向是低于threthold的
        
        df = df[df.orig_out >= thresh]
        df = df[df.Sequence.isin(df.groupby('Sequence').out.apply(lambda x: sum(x<thresh)/len(x)).nlargest(5).index)]
        out = df.groupby('Sequence').out.apply(lambda x: 1- sum(x < thresh )/len(x)).reset_index()
    
    return out


# In[14]:


top5_inchead = top5_motifs(inc_head_df, "increasing", thresh_head)
top5_inctestis = top5_motifs(inc_testis_df, "increasing", thresh_testis)
top5_dechead = top5_motifs(dec_head_df, "decreasing", thresh_head)
top5_dectestis = top5_motifs(dec_testis_df, "decreasing", thresh_testis)


# In[ ]:


top5_dectestis


# In[15]:


def plot_difdist(df, title):

        
    order = df.Sequence[df.out.sort_values(ascending=True).index]
  
    ax = sns.catplot(data=df, 
                y='Sequence', 
                x='out', 
                palette="tab10", 
                order=order, 
                kind="bar", height=2.5)


#     for l in ax.lines:
#         l.set_linestyle('-')
    
    plt.title(title)
    plt.xlim([0, 1])
    ax.set_xlabels("Fraction of examples\nretaining original state")
    ax.set_ylabels("Motif")
    plt.show()
    return ax


# In[16]:


def order_motifs(df):
    return df.Sequence[df.out.sort_values(ascending=True).index]

fig, axs = plt.subplots(2, 2, figsize=(5, 3.5))
    
sns.barplot(data=top5_dechead, 
            y='Sequence', 
            x='out', 
            palette="tab10", 
            order= top5_dechead.pipe(order_motifs), 
            ax=axs[0, 0])

axs[0,0].set_title('Head | peak')
axs[0,0].set_ylabel('Motif')

sns.barplot(data=top5_dectestis, y='Sequence', x='out', 
            palette="tab10", 
            order=top5_dectestis.pipe(order_motifs), 
            ax=axs[0, 1])
axs[0,1].set_title('Testis | peak')
axs[0,1].set_ylabel('')

sns.barplot(data=top5_inchead, 
            y='Sequence', 
            x='out', 
            palette="tab10", 
            order= top5_inchead.pipe(order_motifs), 
            ax=axs[1, 0])
axs[1, 0].set_title('Head | no peak')
axs[1,0].set_ylabel('Motif')


sns.barplot(data=top5_inctestis, y='Sequence', x='out', 
            palette="tab10", 
            order=top5_inctestis.pipe(order_motifs), 
            ax=axs[1, 1])

axs[1, 1].set_title('Testis | no peak')
axs[1,1].set_ylabel('')


for i in range(2):
    for j in range(2):
        axs[i,j].set_xlim([0,1])
        axs[i,j].set_xlabel('')
        axs[i,j].set_ylabel('')

#plt.xlabel("Fraction of examples\nretaining original state")
midx = (fig.subplotpars.right + fig.subplotpars.left)/2
midy = (fig.subplotpars.top + fig.subplotpars.bottom)/2
fig.supxlabel('Fraction of examples\nretaining original state', x=midx+.07)
fig.supylabel('Motif chr3L', y=midy+.07)

plt.tight_layout()
plt.show()
fig.savefig('../run_model/TF_modisco_insertions.pdf', bbox_inches='tight')


# ### Find number of occurences of each motif in each example

# In[6]:


from Bio import AlignIO


# In[11]:


consensus_motifs['metacluster_1'][0]


# In[ ]:





# In[12]:


def onehot_2nuc(onehot):
    if onehot == 0:
        out = 'A'
    elif onehot == 1:
        out = 'T'
    elif onehot == 2:
        out = 'C'
    elif onehot == 3:
        out = 'G'
        
    return out


# In[19]:


X_test[0]


# In[17]:


onehot_2nuc(X_test[0])


# In[ ]:




