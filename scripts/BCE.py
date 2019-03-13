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
  if -1 in x:
    k_j -= 1
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
    logp = 0.
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
        prob = prob / prob.sum()
        Z[i, j] = sample_from_discrete(prob)
        logp += np.log(prob[Z[i, j]])
        if X[i, j] >= 0:
          n_matrices[j][Z[i, j], X[i, j]] += 1
          n_matrices_sum[j][Z[i, j]] += 1
        n_z[Z[i, j]] += 1
    print(logp)  
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
  selected_genes = [g for g in gene_names if np.where(genes[g] >= 0)[0].size > 0] # Filter genes with too sparse entries
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

  k = np.max(Z) + 50 # Extra groups
  alpha = np.ones((k,))/float(k)
  Z_init = []
  for i in range(X.shape[0]):
    theta = np.random.dirichlet(alpha)
    theta[Z[i]] += 6 # Temperature-like
    line_init = [sample_from_discrete(theta) for _ in range(X.shape[1])]
    Z_init.append(line_init)
  Z_init = np.array(Z_init)
  with open('./Z_init_values_123.pkl', 'wb') as f:
    pickle.dump(Z_init, f)
  return Z_init, k, alpha[0]

if __name__ == '__main__':

  ground_truth_clusters = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))
  file_list =['../summary/mspminer_ERP002469.txt',
              '../summary/mspminer_ERP003612.txt',
              '../summary/mspminer_ERP005989.txt',
              '../summary/mspminer_ERP008729.txt',
              '../summary/mspminer_ERP010700.txt']
  X, gene_names = preprocess_files(file_list)


  w_j_ratio = 0.02
  #Z, k, alpha = initialize_Z(X, seed=123)
  k = 1360
  alpha = 0.1/k
  Z = pickle.load(open('./Z_init_values_123.pkl', 'rb'))

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
    Z = BCE(X, k, Z, alpha, w_j_ratio=w_j_ratio, n_epochs=5)
    t2 = time.time()
    print("Took %f seconds" % (t2-t1))
    with open('../utils/temp_save_%d.pkl' % ct, 'wb') as f:
      pickle.dump(Z, f)

