#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:40:03 2019

@author: zqwu
"""
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv
import os
from samples import merge_samples_msp
from MM import build_clusters

ref_species = pickle.load(open('../utils/Species_Gene_Array_0.9_new.pkl', 'rb'))
#ref_strain = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

X, gene_names = pickle.load(open('../summary/mspminer_X_1.pkl', 'rb'))

groups1, order = pickle.load(open('./thr1_results.pkl', 'rb'))
groups2 = pickle.load(open('./thr1_results_953.pkl', 'rb'))

def overlapping_results(ref_species, groups, output_file='./output.csv'):

  f = open(output_file, 'w')
  writer = csv.writer(f)
  writer.writerow(["Species Name", 
                   "N1(All genes in the reference)",
                   "N2(Overlapping genes in the reference also appearing in ensemble clustered gene set)",
                   "N3(All genes in the (ensemble) clusters corresponding to the speci)",
                   "N4(Overlapping genes in the (ensemble) clusters corresponding to the speci also appearing in the reference gene set)",
                   "N5(Overlapping genes)",
                   "N6(Corresponded (ensemble) clusters)",
                   "Precision (N5/N4)",
                   "Recall (N5/N1)",
                   "Micro-Recall (N5/N2)"])
  genes_set = []
  for v in groups.values():
    genes_set.extend(list(v))
  genes_set = set(genes_set)

  ref_genes_set = []
  for v in ref_species.values():
    ref_genes_set.extend(list(v))
  ref_genes_set = set(ref_genes_set)

  results = {}

  for speci in ref_species:
    _gs = set(ref_species[speci])
    n_ref_total = len(_gs) # All genes in the ref species

    set_max_identified = genes_set & _gs
    n_max_identified = len(set_max_identified) # Overlapping genes in the ref species


    assigned_g_ids = []
    restricted_merged_assigned_gs = set()
    for g_id in groups:
      g = set(groups[g_id]) & ref_genes_set
      if len(g & _gs) > 0.5*len(g):
        assigned_g_ids.append(g_id)
        restricted_merged_assigned_gs |= g
    if len(assigned_g_ids) > 0:
      all_merged_assigned_gs = set.union(*[groups[g_id] for g_id in assigned_g_ids])
      n_ens_total = len(all_merged_assigned_gs) # All genes in the matched ensemble groups
    else:
      n_ens_total = 0
    n_total_identified = len(restricted_merged_assigned_gs) # Overlapping genes in the matched ensemble groups
    n_overlapping = len(restricted_merged_assigned_gs & _gs) # Overlapping genes
    n_assigned_clusters = len(assigned_g_ids)
    results[speci] = (n_ref_total, 
                      n_max_identified, 
                      n_ens_total, 
                      n_total_identified, 
                      n_overlapping,
                      n_assigned_clusters,
                      n_overlapping/(n_total_identified+1e-9),
                      n_overlapping/(n_ref_total+1e-9),
                      n_overlapping/(n_max_identified+1e-9))
    writer.writerow([speci] + list(results[speci]))
  f.close()
  return results

      

def tree_distance(ref_species, groups, order, output_file='./output.csv'):

  f = open(output_file, 'w')
  writer = csv.writer(f)
  writer.writerow(["Species Name", 
                   "N(Corresponded (ensemble) clusters",
                   "Step when 50%% of genes are merged",
                   "Step when 60%% of genes are merged",
                   "Step when 70%% of genes are merged",
                   "Step when 80%% of genes are merged",
                   "Step when 90%% of genes are merged",
                   "Step when 100%% of genes are merged"])
  genes_set = []
  for v in groups.values():
    genes_set.extend(list(v))
  genes_set = set(genes_set)

  ref_genes_set = []
  for v in ref_species.values():
    ref_genes_set.extend(list(v))
  ref_genes_set = set(ref_genes_set)

  results = {}

  for speci in ref_species:
    _gs = set(ref_species[speci])
    set_max_identified = genes_set & _gs

    assigned_g_ids = {}
    for g_id in groups:
      g = set(groups[g_id]) & ref_genes_set
      if len(g & _gs) > 0.5*len(g):
        assigned_g_ids[g_id] = len(groups[g_id])
    
    n = len(assigned_g_ids)
    steps = [-1, -1, -1, -1, -1, -1]
    
    if n > 0:
      total_genes = sum(assigned_g_ids.values())

      def fill(step, assigned_g_ids, total_genes, steps):
        n_max = max(assigned_g_ids.values())
        if steps[0] < 0 and n_max > 0.5*total_genes:
          steps[0] = step
        if steps[1] < 0 and n_max > 0.6*total_genes:
          steps[1] = step
        if steps[2] < 0 and n_max > 0.7*total_genes:
          steps[2] = step
        if steps[3] < 0 and n_max > 0.8*total_genes:
          steps[3] = step
        if steps[4] < 0 and n_max > 0.9*total_genes:
          steps[4] = step
        if steps[5] < 0 and n_max == total_genes:
          steps[5] = step
      
      fill(0, assigned_g_ids,total_genes, steps)
      for i, merge_step in enumerate(order):
        if merge_step[1] in assigned_g_ids and merge_step[2] in assigned_g_ids:
          n1 = assigned_g_ids.pop(merge_step[1])
          n2 = assigned_g_ids.pop(merge_step[2])
          assigned_g_ids[merge_step[0]] = n1 + n2
          fill(i, assigned_g_ids, total_genes, steps)
        elif merge_step[1] in assigned_g_ids and not merge_step[2] in assigned_g_ids:
          n1 = assigned_g_ids.pop(merge_step[1])
          assigned_g_ids[merge_step[0]] = n1
        elif not merge_step[1] in assigned_g_ids and merge_step[2] in assigned_g_ids:
          n2 = assigned_g_ids.pop(merge_step[2])
          assigned_g_ids[merge_step[0]] = n2
      


    results[speci] = tuple([n] + steps)
    writer.writerow([speci, n] + steps)
  f.close()
  return results

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def pr_scatter_plot(f_ns, labels=None):
  dfs = [pd.read_csv(f_n) for f_n in f_ns]
  precisions = [df["Precision (N5/N4)"] for df in dfs]
  #recalls = [df["Micro-Recall (N5/N2)"] for df in dfs]
  recalls = [df["Recall (N5/N1)"] for df in dfs]
  sizes = [df["N2(Overlapping genes in the reference also appearing in ensemble clustered gene set)"] for df in dfs]
  n_clusters = [df["N6(Corresponded (ensemble) clusters)"] for df in dfs]

  n_missing_species = [np.where(precision==0)[0].shape[0] for precision in precisions]
  n_missing_genes = [size[np.where(precision==0)[0]].sum() for size, precision in zip(sizes, precisions)]
  
  plt.clf()
  fig, ax=plt.subplots()
  set_size(10, 5, ax)
  for i, precision in enumerate(precisions):
    ax.scatter([i] * len(precision), precision, s=sizes[i]**0.8*0.1, alpha=0.2)
  ax.set_ylabel("Precision")
  ax.set_ylim(0.4, 1.02)
  ax.set_xlim(-0.5, len(f_ns)-0.5)
  if labels is not None:
    ax.set_xticks(np.arange(len(f_ns)-0.5))
    ax.set_xticklabels(labels, rotation=45)
  plt.tight_layout()
  plt.savefig("Precisions.png", dpi=300)

  plt.clf()
  fig, ax=plt.subplots()
  set_size(10, 5, ax)
  for i, recall in enumerate(recalls):
    ax.scatter([i] * len(recall), recall, s=sizes[i]**0.8*0.1, alpha=0.2)
  ax.set_ylabel("Recall")
  ax.set_ylim(-0.02, 1.02)
  ax.set_xlim(-0.5, len(f_ns)-0.5)
  if labels is not None:
    ax.set_xticks(np.arange(len(f_ns))-0.5)
    ax.set_xticklabels(labels, rotation=45)
  plt.tight_layout()
  plt.savefig("Recalls.png", dpi=300)

  plt.clf()
  fig, ax=plt.subplots()
  set_size(10, 5, ax)
  for i, n_cluster in enumerate(n_clusters):
    ax.scatter([i] * len(n_cluster), n_cluster, s=sizes[i]**0.8*0.1, alpha=0.2)
  ax.set_ylabel("Num clusters each species")
  ax.set_ylim(0, 100)
  ax.set_xlim(-0.5, len(f_ns)-0.5)
  if labels is not None:
    ax.set_xticks(np.arange(len(f_ns))-0.5)
    ax.set_xticklabels(labels, rotation=45)
  plt.tight_layout()
  plt.savefig("Num_clusters.png", dpi=300)

  # plt.clf()
  # fig, ax=plt.subplots()
  # set_size(10, 5, ax)
  # for i, precision in enumerate(precisions):
  #   num_unmatched = np.sum(sizes[i][np.where(np.array(precision) == 0)[0]])
  #   ax.scatter([i], 0., s=num_unmatched**0.8*0.1, alpha=0.2)
  # ax.set_xlim(-0.5, len(f_ns)-0.5)
  # if labels is not None:
  #   ax.set_xticks(np.arange(len(f_ns)-0.5))
  #   ax.set_xticklabels(labels, rotation=45)
  # plt.tight_layout()
  # plt.savefig("Unmatched_genes.png", dpi=300)

# for sample_id, name in enumerate(merge_samples_msp):
#   inds = np.where(X[:, sample_id] >= 0)[0]
#   X_c = build_clusters(X[inds, sample_id], [gene_names[i] for i in inds])
#   X_c2 = {k:set(X_c[k]) for k in X_c if len(X_c[k]) > 100}
#   overlapping_results(ref_species, X_c2, '../%s_summary_new.csv' % name)
