#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 23:46:00 2019

@author: zqwu
"""
import numpy as np
import pickle
from similarity import load_samples, adjusted_mutual_information, adjusted_rand_index
import numba
import time

@numba.jit(cache=True, nopython=True)
def sample_from_discrete(prob):
  prob = prob / prob.sum()
  prob = np.cumsum(prob)
  sample = np.where((prob - np.random.rand()) <= 0)[0].shape[0]
  return sample

@numba.jit(cache=True, nopython=True)
def generate_n_matrix(z, x, k):
  N = x.shape[0]
  k_j = len(set(x))
  n_matrix = np.zeros((k, k_j))
  for i in range(N):
    if x[i] < 0: 
      continue
    n_matrix[z[i], x[i]] += 1
  return n_matrix, k_j

@numba.jit(cache=True, nopython=True)
def BCE(X, k, Z, alpha, w_j_ratio=0.4, n_epochs=10):
  M = X.shape[1]
  N = X.shape[0]

  # Preprocessing 
  alpha = np.ones((k,)) * alpha

  n_matrices = [generate_n_matrix(Z[:, i], X[:, i], k) for i in range(M)]
  k_js = [pair[1] for pair in n_matrices]
  w_js = [N/k_j/k_j*w_j_ratio for k_j in k_js]
  n_matrices = [pair[0] for pair in n_matrices]
  
  # Gibbs Sampler
  n_matrices_sum = [nm.sum(1) for nm in n_matrices]
  for epoch in range(n_epochs):
    print(epoch)
    for i in range(N):
      n_z = np.zeros((k,))
      n_z[:(Z[i].max()+1)] = np.bincount(Z[i])
      for j in range(M):
        if X[i, j] >= 0:
          n_matrices[j][Z[i, j], X[i, j]] -= 1
          n_matrices_sum[j][Z[i, j]] -= 1
        n_z[Z[i, j]] -= 1
        
        if X[i, j] >= 0:
          p1 = (w_js[j] + n_matrices[j][:, X[i, j]]) / (k_js[j] * w_js[j] + n_matrices_sum[j])
        else:
          p1 = np.ones((k,)) # Placeholder for missing entries
        p2 = (alpha + n_z)/(k * alpha + (M - 1))
        prob = p1 * p2
        Z[i, j] = sample_from_discrete(prob)
        
        if X[i, j] >= 0:
          n_matrices[j][Z[i, j], X[i, j]] += 1
          n_matrices_sum[j][Z[i, j]] += 1
        n_z[Z[i, j]] += 1
  return Z

def preprocess_files(file_list):
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
  selected_genes = [g for g in gene_names if np.where(genes[g] >= 0)[0].size > n_clusterings/3]
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

def initialize_Z(X, k, alpha):
  alpha = np.ones((k,)) * alpha
  theta = [np.random.dirichlet(alpha) for _ in range(X.shape[0])]
  Z = np.array([[sample_from_discrete(theta[i]) for _ in range(X.shape[1])] for i in range(X.shape[0])])
  return Z
  
if __name__ == '__main__':
  
  ground_truth_clusters = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))
  X, gene_names = preprocess_files(['../samples/summary_ERP002469.txt', 
                                    '../samples/summary_ERP003612.txt', 
                                    '../samples/summary_ERP005989.txt',
                                    '../samples/summary_ERP008729.txt',
                                    '../samples/summary_ERP010700.txt'])

  k = 1500
  alpha = 1./k
  w_j_ratio = 0.2
  #Z = initialize_Z(X, k, alpha)
  Z = pickle.load(open('../utils/temp_save_1.pkl', 'rb'))
  assert Z.shape == X.shape
  for ct in range(100):
    #alpha = 1/100 * 0.96**ct #Annealing
    #w_j_ratio = 0.4 * 0.98**ct #Annealing
    print("Start Iteration %d" % ct, flush=True)
    if ct > 0:
      Z_clusters = {}
      for i, z in enumerate(Z):
        cluster_id = np.random.choice(z) # Randomly select one out of M assignment
        if cluster_id not in Z_clusters:
          Z_clusters[cluster_id] = []
        Z_clusters[cluster_id].append(gene_names[i])
      print("On batch %d, alpha %f" % (ct, alpha))
      print(adjusted_rand_index('../samples/summary_ERP002469.txt', Z_clusters))
      print(adjusted_rand_index(ground_truth_clusters, Z_clusters), flush=True)
    
    t1 = time.time()
    Z = BCE(X, k, Z, alpha, w_j_ratio=w_j_ratio, n_epochs=20)
    t2 = time.time()
    print("Took %f seconds" % (t2-t1))
    with open('../utils/temp_save_1.pkl', 'wb') as f:
      pickle.dump(Z, f)
