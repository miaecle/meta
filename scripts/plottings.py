#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:09:09 2019

@author: zqwu
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from samples import valid_samples, merge_samples

n_samples = {}
with open('cohort_samples.csv', 'r') as f:
    for line in f:
        n_samples[line.split(',')[0]] = int(line.split(',')[1])
        
        
def plot_similarity_matrix(f):
  df = pd.read_csv(f)
  ref_ARI = np.array(df['Ref'])
  ARIs = np.array(df)[:, 2:]
  inds = [list(df['Name']).index(n) for n in merge_samples]
  
  
  plot_mat = np.concatenate([np.expand_dims(ref_ARI[inds], 1), ARIs[inds][:, inds]], 1).astype(float)
  plot_mat[np.where(plot_mat == -1)] = 0.
  plt.imshow(plot_mat, vmin=0.5, vmax=1.0)
  plt.yticks(np.arange(len(merge_samples)), merge_samples, rotation=20, fontsize=6)
  plt.xticks([0], ['ref ARI'], fontsize=6)
  plt.colorbar()
  plt.savefig('temp.png', dpi=600)
  