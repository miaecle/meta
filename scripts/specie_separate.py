#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:40:03 2019

@author: zqwu
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from MM import build_clusters

ref_specie = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))
ref_strain = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

X, gene_names = pickle.load(open('../summary/mspminer_X.pkl', 'rb'))

inds = np.where(X[:, 0] >= 0)[0]
#X_c = build_clusters(X[inds, 0], [gene_names[i] for i in inds])
#X_c2 = {k:X_c[k] for k in X_c if len(X_c[k]) > 100}

Z_c = pickle.load(open('../Z.pkl', 'rb'))
Z_c = build_clusters(Z_c, gene_names)
X_c2 = {k:Z_c[k] for k in Z_c if len(Z_c[k]) > 100}

mapping = {}
for spec in ref_specie.keys():
  strains = [st for st in ref_strain.keys() if st.startswith(spec)]
  if len(strains) > 1:
    mapping[spec] = strains

keys1 = sorted(ref_specie.keys())
genes1 = []
for k1 in keys1:
  genes1.extend(list(ref_specie[k1]))
genes1 = set(genes1)
  
keys2 = sorted(X_c2.keys())
genes2 = []
for k2 in keys2:
  genes2.extend(list(X_c2[k2]))    
genes2 = set(genes2)

shared_genes = genes1 & genes2

for speci in mapping:
  ref_c = set(ref_specie[speci]) & shared_genes
  
  ref_substrains = {k: ref_strain[k] for k in mapping[speci]}
  strain_names = sorted(list(ref_substrains.keys()))
  
  sample_cs_coverage = {}
  for k2 in X_c2:
    if len(set(X_c2[k2]) & shared_genes)> 0:
      sample_cs_coverage[k2] = len(set(X_c2[k2]) & ref_c)/float(len(set(X_c2[k2]) & shared_genes))
  order = sorted(list(sample_cs_coverage.keys()), key=lambda x: -sample_cs_coverage[x])
  
  sample_cs = []
  for i in order:
    if sample_cs_coverage[i] < 0.5:
      break
    sample_cs.append(X_c2[i])
  
  sample_combined = []
  for c in sample_cs:
    sample_combined.extend(c)
    
  print(len(set(ref_specie[speci])))
  print("Recovered\t" + str(len(set(sample_combined) & set(ref_specie[speci]))) + " genes")
  print("Precision\t" + str(len(set(sample_combined) & set(ref_specie[speci]))/(len(set(sample_combined) & shared_genes) + 1e-5)))
  print("Recall\t\t" + str(len(set(sample_combined) & set(ref_specie[speci]))/(len(set(ref_specie[speci]) & shared_genes) + 1e-5)))
  
  overlap_tab = np.zeros((len(sample_cs), len(ref_substrains)+1, 2))
  for i, sample_c in enumerate(sample_cs):
    for j, name in enumerate(strain_names):
      overlap_size = len(set(sample_c) & set(ref_substrains[name]))
      total_possible = len(set(sample_c) & shared_genes)
      overlap_tab[i, j, 0] = overlap_size
      overlap_tab[i, j, 1] = float(overlap_size)/float(total_possible)
    overlap_tab[i, -1, 0] = len(set(sample_c) & set(ref_c))
    overlap_tab[i, -1, 1] = len(set(sample_c) & set(ref_c))/total_possible
  
  plt.clf()
  plt.imshow(np.transpose(overlap_tab[:, :, 1]), vmin=0., vmax=1.)
  ax = plt.gca()
  ax.set_yticks(np.arange(len(strain_names)));
  ax.set_xticks(np.arange(len(sample_cs)));
  ax.set_yticklabels(strain_names, fontsize=3);
  ax.set_xticklabels([''] * len(strain_names));
  
  ax.set_xticks(np.arange(-.5, len(sample_cs) + 0.5, 1), minor=True);
  ax.set_yticks(np.arange(-.5, len(strain_names) + 0.5, 1), minor=True);
  
  ax.grid(which='minor', color='w', linestyle='-', linewidth=0.2)
  plt.colorbar()
  plt.savefig('./figs/' + speci + '.png', dpi=1200)