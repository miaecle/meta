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

################################################
import pickle

prior, thetas = pickle.load(open('../MM_init_on_147_k=4004.pkl', 'rb'))
Z = pickle.load(open('../Z.pkl', 'rb'))

Z_, cts = np.unique(Z, return_counts=True)
Z_ = Z_[np.where(cts > 100)]

for i, theta in enumerate(thetas):
  mat = np.exp(theta[Z_])
  order = sorted(np.arange(mat.shape[0]), key=lambda x: np.argmax(mat[x]))
  plt.clf()
  plt.imshow(mat[order])
  plt.savefig('%d.png' % i, dpi=600)

#############################################
import numpy as np
import pickle
from copy import deepcopy

X, gene_names = pickle.load(open('../summary/mspminer_X.pkl', 'rb'))
ref = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))
genes2 = set()
for k2 in ref:
  genes2 |= set(ref[k2])
inds = [i for i, g in enumerate(gene_names) if g in genes2]

X_ = X[np.array(inds)]
gene_names_ = [gene_names[i] for i in inds]

results = []
np.random.seed(123)
for _ in range(10000):
  i = np.random.randint(0, len(X_))
  c_id = np.random.choice(np.where(X_[i] >= 0)[0])
  js = np.where(X_[:, c_id] == X_[i, c_id])[0]
  n_js = int(np.ceil(0.1 * min(200, len(js))))
  js = list(np.random.choice(js, size=n_js, replace=False))
  
  result = np.zeros((len(js), 5))
  for ct, j in enumerate(js):
    effective = sum((X_[i] >= 0) & (X_[j] >= 0))
    consistent = sum((X_[i] >= 0) & (X_[j] >= 0) & (X_[i] == X_[j]))
    result[ct] = np.array([i, j, effective, consistent, 0])
  
  check = set(js)
  for k2 in ref:
    if gene_names_[i] in ref[k2]:
      new_check = deepcopy(check)
      for j in check:
        if gene_names_[j] in ref[k2]:
          result[js.index(j), 4] = 1
          new_check.remove(j)
      if len(new_check) == 0:
        break
      check = new_check
  results.append(result)

results = np.concatenate(results, 0)
#with open('./consistency_precision.pkl', 'wb') as f:
#  pickle.dump(results, f)

d = {}
d2 = {}
for line in results:
  if line[3] not in d:
    d[line[3]] = [0, 0]
  if line[2] not in d2:
    d2[line[2]] = [0, 0]
  d[line[3]][0] += 1
  d[line[3]][1] += line[4]
  d2[line[2]][0] += 1
  d2[line[2]][1] += line[4]

line = []
for k in sorted(d.keys()):
  line.append((k, d[k][1]/d[k][0]))
line = np.array(line)

line2 = []
for k in sorted(d2.keys()):
  line2.append((k, d2[k][1]/d2[k][0]))
line2 = np.array(line2)
plt.plot(line[:, 0], line[:, 1], '.-', label='n consistent')
plt.plot(line2[:, 0], line2[:, 1], '.-', label='n co-appear')
plt.legend()
plt.xlabel('N_studies')
plt.ylabel('Precision')
plt.savefig('consistency_precision.png', dpi=300)
