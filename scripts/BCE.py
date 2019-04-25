#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 23:46:00 2019

@author: zqwu
"""
import numpy as np
import pickle
from similarity import load_samples, adjusted_rand_index
from samples import valid_samples, merge_samples, merge_samples_msp

import numba
import time
import threading
from multiprocessing import Process, Pool
from functools import partial
import multiprocessing as mp
import argparse


from scipy.special import digamma, polygamma

@numba.jit(cache=True, nopython=True)
def sample_from_discrete(prob):
  prob = prob / prob.sum()
  prob = np.cumsum(prob)
  sample = np.where((prob - np.random.rand()) <= 0)[0].shape[0]
  return sample

def generate_n_matrix(z, x, k):
  N = x.shape[0]
  k_j = len(set(x))
  if -1 in x:
    k_j -= 1
  n_matrix = np.zeros((k, k_j))
  for i in range(N):
    if x[i] < 0:
      continue
    n_matrix[z[i], x[i]] += 1
  return n_matrix, k_j

def preprocess_files(file_list, threshold=0):
  genes = {}
  n_clusterings = len(file_list)
  for i, f_n in enumerate(file_list):
    dat = load_samples(f_n)
    for ind_key, key in enumerate(dat.keys()):
      for g in dat[key]:
        if not g in genes:
          genes[g] = -np.ones((n_clusterings,))
        genes[g][i] = ind_key

  gene_names = sorted(genes.keys())
  selected_genes = [g for g in gene_names if np.where(genes[g] >= 0)[0].size > threshold] # Filter genes with too sparse entries
  B = np.stack([genes[g] for g in selected_genes], 0)
  for j in range(B.shape[1]):
    group_ids = np.unique(B[:, j])
    group_ids = [group_id for group_id in group_ids if group_id >= 0]
    mapping = {group_id: i for i, group_id in enumerate(group_ids)}
    for i in range(B.shape[0]):
      if B[i, j] >= 0:
        B[i, j] = mapping[B[i, j]]
    assert len(np.unique(B[:, j])) == np.max(B[:, j]) + 2 or \
           len(np.unique(B[:, j])) == np.max(B[:, j]) + 1
  print("Number of genes: %d" % len(selected_genes))
  print("Missing entry rate: %f" % (np.where(B < 0)[0].size/B.size))

  return B.astype(int), selected_genes

def initialize_Z(X, seed=None):
  if not seed is None:
    np.random.seed(seed)

  order = np.arange(X.shape[1])
  np.random.shuffle(order)
  Z = np.copy(X[:, order[0]])
  for i in order[1:]:
    update_positions = set(list(np.where(Z < 0)[0])) & \
        set(list(np.where(X[:, i] >= 0)[0]))
    for j in update_positions:
      if Z[j] < 0:
        assign = X[j, i]
        assert assign >= 0
        points_in_this_cluster = np.where(X[:, i] == assign)[0]
        existing_cluster_assignments = [Z[p] for p in points_in_this_cluster if Z[p] >= 0]
        missing_assignments = [p for p in points_in_this_cluster if Z[p] < 0]
        if len(existing_cluster_assignments) < 5:
          existing_cluster_assignments.append(np.max(Z) + 1) # Assignment to a new cluster

        assert not -1 in existing_cluster_assignments
        assert j in missing_assignments

        if len(missing_assignments) > 20: # When large number of unlabelled points are present, assign them to the same cluster
          j_assign = np.random.choice(existing_cluster_assignments)
          for k in missing_assignments:
            Z[k] = j_assign
        else:
          for k in missing_assignments:
            j_assign = np.random.choice(existing_cluster_assignments)
            Z[k] = j_assign
  assert not -1 in Z
  assert len(np.unique(Z)) == np.max(Z) + 1

  k = np.max(Z) + 1
  return Z, k

def initialize_Z_spread(X, seed=None):
  if not seed is None:
    np.random.seed(seed)

  order = np.arange(X.shape[1])
  np.random.shuffle(order)
  Z = np.copy(X[:, order[0]])
  for i in order[1:]:
    update_positions = set(list(np.where(Z < 0)[0])) & \
        set(list(np.where(X[:, i] >= 0)[0]))
    for j in update_positions:
      if Z[j] < 0:
        assign = X[j, i]
        assert assign >= 0
        points_in_this_cluster = np.where(X[:, i] == assign)[0]
        existing_cluster_assignments = [Z[p] for p in points_in_this_cluster if Z[p] >= 0]
        if len(existing_cluster_assignments) < 20:
          existing_cluster_assignments.append(np.max(Z) + 1) # Assignment to a new cluster
        _, existing_cts = np.unique(existing_cluster_assignments, return_counts=True)
        max_ct = np.max(existing_cts)


        missing_assignments = [p for p in points_in_this_cluster if Z[p] < 0]
        assert not -1 in existing_cluster_assignments
        assert j in missing_assignments

        if len(missing_assignments) > 0.3*max_ct: # When large number of unlabelled points are present, assign them to the same cluster
          j_assign = np.max(Z) + 1
          for k in missing_assignments:
            Z[k] = j_assign
        else:
          for k in missing_assignments:
            j_assign = np.random.choice(existing_cluster_assignments)
            Z[k] = j_assign
  assert not -1 in Z
  assert len(np.unique(Z)) == np.max(Z) + 1

  k = np.max(Z) + 1
  return Z, k

def build_clusters(Z, gene_names):
  Z_clusters = {}
  for i, z in enumerate(Z):
    cluster_id = z
    if cluster_id not in Z_clusters:
      Z_clusters[cluster_id] = []
    Z_clusters[cluster_id].append(gene_names[i])
  return Z_clusters

def load_cohort_sizes(sample_lists, path='../utils/cohort_samples.csv'):
  mapping = {}
  with open(path, 'r') as f:
    for line in f:
      line = line[:-1]
      mapping[line.split(',')[0]] = int(line.split(',')[1])
  return [mapping[name] for name in sample_lists]


def BCE(X, k, Z_init=None, alpha=None, betas=None, n_iter=10):
  M = X.shape[1]
  N = X.shape[0]


  if betas is None:
    n_matrices = [generate_n_matrix(Z_init, X[:, i], k)[0] + 1 for i in range(M)]
    betas = [n_matrix/np.expand_dims(n_matrix.sum(1), 1) for n_matrix in n_matrices]
  
  if alpha is None:
    alpha = np.bincount(Z_init) + 1
  
  alpha = alpha/alpha.sum() * k
  
  for it_ct in range(n_iter):
    print(it_ct)
    new_betas = [np.zeros_like(b) for b in betas]
    
    ghs = N*(digamma(alpha.sum()) - digamma(alpha))
    lhs = -N * polygamma(1, alpha)
    v = N * polygamma(1, alpha.sum())
    
    for i in range(N):
      phi_i = np.zeros((M, k))
      phi_i += digamma(alpha) - digamma(alpha.sum())
      for j in range(M):
        if X[i, j] >= 0:
          phi_i[j] += np.log(betas[j][:, X[i, j]])
      
      phi_i = np.exp(phi_i)
      phi_i = phi_i/np.sum(phi_i, 1, keepdims=True)
      
      for j in range(M):
        new_betas[j][:, X[i, j]] += phi_i[j]
        
      gamma_i = alpha + phi_i.sum(0)
      ghs += digamma(gamma_i) - digamma(gamma_i.sum())
      
    betas = [b/b.sum(1, keepdims=True) for b in new_betas]
    c = (ghs/lhs).sum()/(v**(-1.0) + (lhs**(-1.0)).sum())
    alpha = alpha - (ghs - c)/lhs
    
  return alpha, betas

if __name__ == '__main__':

  ground_truth_clusters = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))
  samples = merge_samples_msp
  n_threads = 4

  file_list =['../summary/mspminer_%s.txt' % name for name in samples]
  #X, gene_names = preprocess_files(file_list)
  X, gene_names = pickle.load(open('../summary/mspminer_X_4.pkl', 'rb'))
  cohort_sizes = load_cohort_sizes(samples)

  # Cohorts have weight related to their sizes, 0.3 is a scaling factor
  sample_weights = (np.array(cohort_sizes)/max(cohort_sizes))**(0.3)
  
  #k = 1314
  Z, k = initialize_Z(X, seed=147) # seed=26
  #Z, k = pickle.load(open('../utils/Z_full_init_123_spreaded.pkl', 'rb'))
  #Z = np.random.randint(0, k, (X.shape[0],))
  #Z = None

  alpha = None
  betas = None
  



  #Z = pickle.load(open('../utils/temp_save_1.pkl', 'rb'))
  assert Z.shape == X.shape
  for ct in range(100):
    #alpha = 1/(0.2*k) * 0.96**ct #Annealing
    #w_j_ratio = 0.4 * 0.98**ct #Annealing
    print("Start Iteration %d" % ct, flush=True)
    Z_clusters = {}
    for i, z in enumerate(Z):
      #cluster_id = np.random.choice(z) # Randomly select one out of M assignment\
      vals, counts = np.unique(Z[i], return_counts=True)
      cluster_id = vals[np.argmax(counts)]
      if cluster_id not in Z_clusters:
        Z_clusters[cluster_id] = []
      Z_clusters[cluster_id].append(gene_names[i])
    print("On batch %d" % ct)
    for f in file_list:
      print(f + "\t" + str(adjusted_rand_index(f, Z_clusters)))
    print("Ground Truth\t" + str(adjusted_rand_index(ground_truth_clusters, Z_clusters)), flush=True)

    t1 = time.time()
    alpha, betas = BCE(X, k, Z, n_epochs=10)
    t2 = time.time()
    print("Took %f seconds" % (t2-t1))

