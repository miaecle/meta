#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:35:05 2019

@author: zqwu
"""
import os
import numpy as np
from scipy.special import comb, perm
from scipy import sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import threading
from multiprocessing import Process, Pool
from functools import partial
import multiprocessing as mp
from bisect import bisect_left

def parse_line(line):
  line = line[:-1].split(',')
  gene_name = line[0]
  cluster_assignments = line[1:]
  assert len(cluster_assignments) > 0
  if len(cluster_assignments) == 1:
    cluster_id = int(cluster_assignments[0][1:])
    return gene_name, cluster_id
  else:
    types = [c[0] for c in cluster_assignments]
    if 'c' in types:
      cluster_id = min([int(c[1:]) for c in cluster_assignments if c[0] == 'c'])
      return gene_name, cluster_id
    if 'a' in types:
      cluster_id = min([int(c[1:]) for c in cluster_assignments if c[0] == 'a'])
      return gene_name, cluster_id
    cluster_id = min([int(c[1:]) for c in cluster_assignments])
    return gene_name, cluster_id

def load_samples(s1):
  clusters = {}
  with open(s1, 'r') as f:
    for line in f:
      gene_name, cluster_id = parse_line(line)
      if not cluster_id in clusters:
        clusters[cluster_id] = []
      clusters[cluster_id].append(gene_name)
  return clusters

def cluster_labels(s1, s2):
  if s1.__class__ is str:
    s1 = load_samples(s1)
  if s2.__class__ is str:
    s2 = load_samples(s2)
  
  genes1 = set()
  for k1 in s1:
    genes1 |= set(s1[k1])
  genes2 = set()
  for k2 in s2:
    genes2 |= set(s2[k2])

  shared_genes = list(genes1 & genes2)
  print("Number of genes: %d\t%d\t%d" % (len(genes1), len(genes2), len(shared_genes)))
  
  cluster1 = -np.ones(len(shared_genes))
  cluster2 = -np.ones(len(shared_genes))
  cl1 = {}
  cl2 = {}
  for i, key in enumerate(s1.keys()):
    for g in set(s1[key]):
        cl1[g] = i
  for i, key in enumerate(s2.keys()):
    for g in set(s2[key]):
        cl2[g] = i
  for i, g in enumerate(shared_genes):
    cluster1[i] = cl1[g]
    cluster2[i] = cl2[g]
  return cluster1, cluster2

def contingency(s1, s2):
  #print("Calculating contingency table for clustering %s & %s" % (s1, s2))
  c1, c2 = cluster_labels(s1, s2)
  classes, class_idx = np.unique(c1, return_inverse=True)
  clusters, cluster_idx = np.unique(c2, return_inverse=True)
  n_classes = classes.shape[0]
  n_clusters = clusters.shape[0]
  # Using coo_matrix to accelerate simple histogram calculation,
  # i.e. bins are consecutive integers
  # Currently, coo_matrix is faster than histogram2d for simple cases
  cont = sp.coo_matrix((np.ones(class_idx.shape[0]),
                        (class_idx, cluster_idx)),
                       shape=(n_classes, n_clusters),
                       dtype=np.int)
  cont = cont.toarray()
  return cont

def adjusted_rand_index(s1, s2):
  c1, c2 = cluster_labels(s1, s2)
  return adjusted_rand_score(c1, c2)


def rand_index(s1, s2):
  cont = contingency(s1, s2)

  n_samples = np.sum(cont)  
  sum_comb_c = sum(comb(n_c, 2) for n_c in np.ravel(cont.sum(axis=1)))
  sum_comb_k = sum(comb(n_k, 2) for n_k in np.ravel(cont.sum(axis=0)))
  sum_comb = sum(comb(n_ij, 2) for n_ij in np.ravel(cont))

  prod_comb = (sum_comb_c * sum_comb_k) / comb(n_samples, 2)
  mean_comb = (sum_comb_k + sum_comb_c) / 2.
    
  return (sum_comb - prod_comb) / (mean_comb - prod_comb)

def restricted_rand_index(clusters, label_clusters, n_threads=None):
  genes1 = set()
  for k1 in clusters:
    genes1 |= set(clusters[k1])
  genes2 = set()
  for k2 in label_clusters:
    genes2 |= set(label_clusters[k2])
  shared_genes = genes1 & genes2
  print("Number of genes: %d\t%d\t%d" % (len(genes1), len(genes2), len(shared_genes)))
  label_clusters = {k: set(label_clusters[k]) for k in label_clusters}
  if n_threads is None:
    n_threads = mp.cpu_count()
  all_keys = list(clusters.keys())
  cuts = np.linspace(0, len(all_keys)+1, n_threads+1)
  keys_list = [all_keys[int(cuts[i]):int(cuts[i+1])] for i in range(n_threads)]
  threadRoutine = partial(worker_restricted_index,
                          shared_genes=shared_genes,
                          clusters=clusters,
                          label_clusters=label_clusters)
  with Pool(n_threads) as p:
    res = p.map(threadRoutine, keys_list)

  total = sum([r[1] for r in res])
  consistent = sum([r[0] for r in res])
  print("Consistent pair: %d" % consistent)
  print("All pair: %d" % total)
  print("Ratio: %f" % (float(consistent)/total))
  return consistent, total

def worker_restricted_index(keys, shared_genes=None, clusters=None, label_clusters=None):
  total = 0.
  consistent = 0.
  for key in keys:
    cl = shared_genes & set(clusters[key])
    cl_list = sorted(list(cl))
    if len(cl) <= 1:
      continue
    mat = np.zeros((len(cl), len(cl)), dtype=bool)
    for cl2 in label_clusters.values():
      shared = cl & cl2
      if len(shared) > 1:
        idx = np.array([bisect_left(cl_list, g) for g in shared])
        mat[idx[:, None], idx] = True
    np.fill_diagonal(mat, False)
    total += comb(len(cl), 2)
    consistent += np.sum(mat)/2
    del mat
  return consistent, total

def cross_consistency(setA, setB, label_clusters, label_genes):
  setA = setA & label_genes
  setB = setB & label_genes
  if len(setA) == 0 or len(setB) == 0:
    return 0., 0.
  total = len(setA) * len(setB)
  consistent = 0.
  mat = np.zeros((len(setA), len(setB)), dtype=bool)
  listA = sorted(list(setA))
  listB = sorted(list(setB))
  for cl in label_clusters.values():
    sharedA = setA & cl
    sharedB = setB & cl
    if len(sharedA) > 0 and len(sharedB) > 0:
      idx_A = np.array([bisect_left(listA, g) for g in sharedA])
      idx_B = np.array([bisect_left(listB, g) for g in sharedB])
      mat[idx_A[:, None], idx_B] = True
  consistent += np.sum(mat)
  del mat
  return consistent, total

def min_phylogeny_tree_dist(g, species, label_clusters, phylogeny_tree=None):
  #TODO: find a phylogenetic tree
  return 1.

def phylogeny_distance(clusters, label_clusters, phylogeny_tree=None):
  label_clusters = {c:set(label_clusters[c]) for c in label_clusters}
  set_label = set()
  for c in label_clusters:
    set_label |= label_clusters[c]
  
  distances = {}
  for c in clusters:
    gs = set(clusters[c]) & set_label
    overlapping_nums = {c2:len(gs & label_clusters[c2]) for c2 in label_clusters}
    assigned_species = sorted(overlapping_nums.keys(), key=lambda x: overlapping_nums[x])[-1]
    dist = [0] * overlapping_nums[assigned_species]
    for g in gs - label_clusters[assigned_species]:
      dist.append(min_phylogeny_tree_dist(g, assigned_species, label_clusters, phylogeny_tree=phylogeny_tree))
    distances[c] = dist
  return distances

#def adjusted_mutual_information(s1, s2):
#  c1, c2 = cluster_labels(s1, s2)
#  return adjusted_mutual_info_score(c1, c2)
#  
#def normalized_mutual_information(s1, s2):
#  contingency_table = contingency(s1, s2)
#  n_genes = np.sum(contingency_table)
#  
#  H1 = np.sum(contingency_table, 1)/float(n_genes)
#  H1 = - np.sum(H1 * np.log(H1 + 1e-9))
#  H2 = np.sum(contingency_table, 0)/float(n_genes)
#  H2 = - np.sum(H2 * np.log(H2 + 1e-9))
#  H_cross = contingency_table/float(n_genes)
#  H_cross = - np.sum(H_cross * np.log(H_cross + 1e-9))
#  
#  NMI = (H1 + H2 - H_cross)/np.sqrt(H1 * H2)
#  print("Normalized mutual information: %f" % NMI)
#  return NMI
#
#def variation_of_information(s1, s2):
#  contingency_table = contingency(s1, s2)
#  n_genes = np.sum(contingency_table)
#  
#  H1 = np.sum(contingency_table, 1)/float(n_genes)
#  H1 = - np.sum(H1 * np.log(H1 + 1e-9))
#  H2 = np.sum(contingency_table, 0)/float(n_genes)
#  H2 = - np.sum(H2 * np.log(H2 + 1e-9))
#  H_cross = contingency_table/float(n_genes)
#  H_cross = - np.sum(H_cross * np.log(H_cross + 1e-9))
#  
#  I = H1 + H2 - H_cross
#  NVI = 1 - 2*I/(H1 + H2)
#  return NVI
  


