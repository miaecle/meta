#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:35:05 2019

@author: zqwu
"""
import os
import numpy as np
from scipy.special import comb, perm
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

def load_samples(s1):
  clusters = {}
  with open(s1, 'r') as f:
    for line in f:
      if line[-2] != ',':
        numbers = line[:-1].split(',')
        numbers = list(map(int, numbers))
        for n in numbers[1:]:
          if not n in clusters:
            clusters[n] = []
          clusters[n].append(numbers[0])
  return clusters

def contingency(s1, s2):
  #print("Calculating contingency table for clustering %s & %s" % (s1, s2))
  if s1.__class__ is str:
    s1 = load_samples(s1)
  if s2.__class__ is str:
    s2 = load_samples(s2)
  contingency_table = np.zeros((len(s1), len(s2)))
  
  keys1 = sorted(s1.keys())
  genes1 = set()
  for k1 in keys1:
    s1[k1] = set(s1[k1])
    genes1 = genes1 | s1[k1]
    
  keys2 = sorted(s2.keys())
  genes2 = set()
  for k2 in keys2:
    s2[k2] = set(s2[k2])
    genes2 = genes2 | s2[k2]

  shared_genes = genes1 & genes2
  print("Number of genes in cohort 1: %d" % len(genes1))
  print("Number of genes in cohort 2: %d" % len(genes2))
  print("Number of genes shared: %d" % len(shared_genes))
  for i, k1 in enumerate(keys1):
    for j, k2 in enumerate(keys2):
      contingency_table[i, j] = len(s1[k1] & s2[k2] & shared_genes)
  return contingency_table

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
  print("Number of genes in cohort 1: %d" % len(genes1))
  print("Number of genes in cohort 2: %d" % len(genes2))
  print("Number of genes shared: %d" % len(shared_genes))
  
  cluster1 = -np.ones(len(shared_genes))
  cluster2 = -np.ones(len(shared_genes))
  cl1 = {}
  cl2 = {}
  for i, key in enumerate(s1.keys()):
    for g in set(s1[key]) & set(shared_genes):
        cl1[g] = i
  for i, key in enumerate(s2.keys()):
    for g in set(s2[key]) & set(shared_genes):
        cl2[g] = i
  for i, g in enumerate(shared_genes):
    cluster1[i] = cl1[g]
    cluster2[i] = cl2[g]
  return cluster1, cluster2

def adjusted_rand_index(s1, s2):
#  contingency_table = contingency(s1, s2)
#  norm1 = np.sum([comb(i, 2) for i in np.sum(contingency_table, 1)])
#  norm2 = np.sum([comb(i, 2) for i in np.sum(contingency_table, 0)])
#  n_genes = np.sum(contingency_table)
#  norm_all = comb(n_genes, 2)
#  
#  sums = 0
#  for n in contingency_table.flatten():
#    sums += comb(n, 2)
#  res = (sums - norm1 * norm2 / norm_all) / \
#      (0.5 * (norm1 + norm2) - norm1 * norm2 / norm_all)
#  print("Adjusted rank index: %f" % res)
  c1, c2 = cluster_labels(s1, s2)
  return adjusted_rand_score(c1, c2)

def adjusted_mutual_information(s1, s2):
  c1, c2 = cluster_labels(s1, s2)
  return adjusted_mutual_info_score(c1, c2)
  
def normalized_mutual_information(s1, s2):
  contingency_table = contingency(s1, s2)
  n_genes = np.sum(contingency_table)
  
  H1 = np.sum(contingency_table, 1)/float(n_genes)
  H1 = - np.sum(H1 * np.log(H1 + 1e-9))
  H2 = np.sum(contingency_table, 0)/float(n_genes)
  H2 = - np.sum(H2 * np.log(H2 + 1e-9))
  H_cross = contingency_table/float(n_genes)
  H_cross = - np.sum(H_cross * np.log(H_cross + 1e-9))
  
  NMI = (H1 + H2 - H_cross)/np.sqrt(H1 * H2)
  print("Normalized mutual information: %f" % NMI)
  return NMI

def variation_of_information(s1, s2):
  contingency_table = contingency(s1, s2)
  n_genes = np.sum(contingency_table)
  
  H1 = np.sum(contingency_table, 1)/float(n_genes)
  H1 = - np.sum(H1 * np.log(H1 + 1e-9))
  H2 = np.sum(contingency_table, 0)/float(n_genes)
  H2 = - np.sum(H2 * np.log(H2 + 1e-9))
  H_cross = contingency_table/float(n_genes)
  H_cross = - np.sum(H_cross * np.log(H_cross + 1e-9))
  
  I = H1 + H2 - H_cross
  NVI = 1 - 2*I/(H1 + H2)
  return NVI
  
