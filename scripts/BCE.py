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
from scipy.stats import dirichlet

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

class BCE(object):
  def __init__(self, X, k, Z_init=None, alpha=None, betas=None, weights=None):
    self.X = X
    self.k = k
    self.M = X.shape[1] # n_exp
    self.N = X.shape[0] # n_genes

    if betas is None:
      n_matrices = [generate_n_matrix(Z_init, X[:, i], k)[0] + 1 for i in range(self.M)]
      betas = [n_matrix/np.expand_dims(n_matrix.sum(1), 1) for n_matrix in n_matrices]
    self.log_betas = [np.log(b) for b in betas]
    
    if alpha is None:
      alpha = np.bincount(Z_init) + 1
    self.alpha = alpha
    self.prior = digamma(self.alpha) - digamma(self.alpha.sum())
    
    if weights is None:
      weights = [1.] * self.M
    self.weights = weights


  def infer_zi(self, i):
    phi_i = np.zeros((self.M, self.k))
    phi_i += self.prior
    zs = []
    for j in range(self.M):
      if X[i, j] >= 0:
        phi_i[j] += self.log_betas[j][:, X[i, j]]
        zs.append(np.argmax(phi_i[j]))
    return np.random.choice(zs)
  
  def infer_phi_i(self, i):
    phi_i = np.zeros((self.M, self.k))
    phi_i += self.prior
    for j in range(self.M):
      if X[i, j] >= 0:
        phi_i[j] += self.log_betas[j][:, X[i, j]]
    
    phi_i = np.exp(phi_i)
    phi_i = phi_i/np.sum(phi_i, 1, keepdims=True)
    return phi_i


def EM_BCE(n_iter, bce, n_threads=None):
  if n_threads is None:
    n_threads = mp.cpu_count()
  pl = Pool(n_threads)
  inds = np.arange(bce.N)
  cuts = np.linspace(0, len(inds)+1, n_threads+1)
  i_lists = [inds[int(cuts[i]):int(cuts[i+1])] for i in range(n_threads)]
  for epoch in range(n_iter):
    print("Start iteration %d" % epoch)
    threadRoutine = partial(WorkerEM_BCE, bce=bce)
    res = pl.map(threadRoutine, i_lists)

    all_new_betas = [r[0] for r in res]
    new_betas = [sum([p[i] for p in all_new_betas]) for i in range(bce.M)]

    alpha = bce.alpha
    lhs = -bce.N * polygamma(1, alpha)
    v = bce.N * polygamma(1, alpha.sum())    
    ghs = bce.N*(digamma(alpha.sum()) - digamma(alpha))    
    for r in res:
      ghs_segment = r[1]
      ghs += ghs_segment
    c = (ghs/lhs).sum()/(v**(-1.0) + (lhs**(-1.0)).sum())
    
    betas = [b/b.sum(1, keepdims=True) for b in new_betas]
    alpha = alpha - (ghs - c)/lhs

    bce.alpha = alpha
    bce.prior = digamma(alpha) - digamma(alpha.sum())
    bce.log_betas = [np.log(b) for b in betas]

def WorkerEM_BCE(i_list, bce=None):
  M = bce.M
  ghs = 0.
  new_betas = [np.zeros_like(b) for b in bce.log_betas]
  for i in i_list:
    phi_i = bce.infer_phi_i(i)
    gamma_i = alpha + phi_i.sum(0)
    for j in range(M):
      new_betas[j][:, X[i, j]] += phi_i[j]      
    ghs += digamma(gamma_i) - digamma(gamma_i.sum())
  return new_betas, ghs

def InferZ_BCE(bce, n_threads=None):
  if n_threads is None:
    n_threads = mp.cpu_count()
  inds = np.arange(bce.N)
  cuts = np.linspace(0, len(inds)+1, n_threads+1)
  i_lists = [inds[int(cuts[i]):int(cuts[i+1])] for i in range(n_threads)]
  threadRoutine = partial(WorkerInferZ_BCE, bce=bce)
  with Pool(n_threads) as p:
    res = p.map(threadRoutine, i_lists)
  Z = []
  for r in res:
    Z.extend(r)
  return Z

def WorkerInferZ_BCE(i_list, bce=None):
  Zs = [None] * len(i_list)
  for ind, i in enumerate(i_list):
    z_i = bce.infer_zi(i)
    Zs[ind] = z_i
  return Zs

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='MM for ensemble clustering')
  parser.add_argument(
      '-n',
      action='append',
      dest='thr',
      default=[],
      help='Threshold for input genes')
  
  args = parser.parse_args()
  thr = int(args.thr)

  ground_truth_clusters = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))
  samples = merge_samples_msp
  n_threads = 4

  file_list =['../summary/mspminer_%s.txt' % name for name in samples]
  #X, gene_names = preprocess_files(file_list, threshold=thr)
  X, gene_names = pickle.load(open('../summary/mspminer_X_%d.pkl' % thr, 'rb'))
  cohort_sizes = load_cohort_sizes(samples)

  # Cohorts have weight related to their sizes, 0.3 is a scaling factor
  sample_weights = (np.array(cohort_sizes)/max(cohort_sizes))**(0.3)
  
  Z, k = initialize_Z(X, seed=26)
  #Z, k = pickle.load(open('../utils/Z_full_init_123_spreaded.pkl', 'rb'))
  #Z = None

  alpha = None
  betas = None
  
  bce = BCE(X, k, Z_init=Z, sample_weights=sample_weights)

  Z_clusters = build_clusters(InferZ_BCE(bce, n_threads=n_threads), gene_names)
  scores = []
  for f in file_list:
    scores.append(adjusted_rand_index(f, Z_clusters))
    print(f + "\t" + str(scores[-1]))
  print("Mean score\t" + str(np.mean(np.array(scores))))
  print("Mean score\t" + str(np.sum(np.array(scores) * np.array(cohort_sizes))/np.sum(cohort_sizes)))
  print("Ground Truth\t" + str(adjusted_rand_index(ground_truth_clusters, Z_clusters)), flush=True)

  for ct in range(10):
    print("Start fold %d" % ct, flush=True)
    t1 = time.time()
    EM_BCE(1, bce, n_threads=n_threads)
    t2 = time.time()
    print("Took %f seconds" % (t2-t1))
    with open('../utils/BCE_save/BCE_save_%d_%d.pkl' % (thr, ct), 'wb') as f:
      pickle.dump([bce.alpha, bce.log_betas], f)

    Z_clusters = build_clusters(InferZ_BCE(bce, n_threads=n_threads), gene_names)
    scores = []
    for f in file_list:
      scores.append(adjusted_rand_index(f, Z_clusters))
      print(f + "\t" + str(scores[-1]))
    print("Mean score\t" + str(np.mean(np.array(scores))))
    print("Mean score\t" + str(np.sum(np.array(scores) * np.array(cohort_sizes))/np.sum(cohort_sizes)))
    print("Ground Truth\t" + str(adjusted_rand_index(ground_truth_clusters, Z_clusters)), flush=True)

