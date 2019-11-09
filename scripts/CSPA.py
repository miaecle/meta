#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:40:39 2019

@author: zqwu
"""
import numpy as np
import pickle
from samples import valid_samples, merge_samples, merge_samples_msp
from similarity import load_samples, adjusted_rand_index, restricted_rand_index, cross_consistency
from BCE import generate_n_matrix, preprocess_files, initialize_Z, initialize_Z_spread, build_clusters, load_cohort_sizes, load_inter_cohort_consistency
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import time
import threading
import argparse
import itertools

def check_consistency(a, b, thr=5):
  a = np.array(a)
  b = np.array(b)
  if np.sum((a != b) & (a >= 0) & (b >= 0)) > 0:
    return False
  if np.sum((a == b) & (a >= 0) & (b >= 0)) < thr:
    return False
  return True

def generate_unique_keys(keys, inds1, inds2, position=0, thr=5):
  merge_pairs = []
  if position == len(keys[0]) - 1 or (len(inds1) <100 and len(inds2) < 100):
    for i in inds1:
      for j in inds2:
        if j > i and check_consistency(keys[i], keys[j], thr=thr):
          merge_pairs.append((i, j))
    return merge_pairs

  choices1 = {-1: []}
  for i in inds1:
    if keys[i][position] not in choices1:
      choices1[keys[i][position]] = []
    choices1[keys[i][position]].append(i)

  choices2 = {-1: []}
  for i in inds2:
    if keys[i][position] not in choices2:
      choices2[keys[i][position]] = []
    choices2[keys[i][position]].append(i)

  cs = set(choices1.keys()) & set(choices2.keys())
  for c in cs:
    if c == -1:
      # unknown-unknown cases
      merge_pairs.extend(generate_unique_keys(keys, choices1[-1], choices2[-1], position+1, thr=thr))
    else:
      # Split into c1-c2, c1-unknown, unknown-c2 cases
      if c in choices1:
        merge_pairs.extend(generate_unique_keys(keys, choices1[c], choices2[-1], position+1, thr=thr))
      if c in choices2:
        merge_pairs.extend(generate_unique_keys(keys, choices1[-1], choices2[c], position+1, thr=thr))
      if c in choices1 and c in choices2:
        merge_pairs.extend(generate_unique_keys(keys, choices1[c], choices2[c], position+1, thr=thr))
  return merge_pairs

def generate_groups(X, Z, thr=5, seed=None):
  if not seed is None:
    np.random.seed(seed)
  Z = np.array(Z)
  clusters, sizes = np.unique(Z, return_counts=True)
  groups = {}
  for ct, cluster in enumerate(clusters):
    IDs = set(np.where(Z==cluster)[0])
    remaining = IDs
    while (len(remaining) > 0):
      order = np.array(list(remaining))
      np.random.shuffle(order)
      sd = order[0]
      group = [sd]
      group_assign = X[sd]
      for i in order[1:]:
        if check_consistency(X[i], group_assign, thr=thr):
          group.append(i)
          group_assign = np.max(np.stack([group_assign, X[i]], 0), 0)

      final_group = set()
      for i in group:
        if check_consistency(X[i], group_assign, thr=thr):
          final_group.add(i)
      if tuple(group_assign) in groups:
        groups[tuple(group_assign)] = groups[tuple(group_assign)] | final_group
      else:
        groups[tuple(group_assign)] = final_group
      remaining = remaining - final_group
    assert sum([len(v) for v in list(groups.values())]) == sum(sizes[:(ct+1)])
  return groups

def merge_keys(all_keys, thr=5, seed=None):
  if not seed is None:
    np.random.seed(seed)

  mappings = {k: [k] for k in all_keys}
  fixed_mappings = {}
  it = 0

  for _ in range(2):
    keys = list(mappings.keys())
    # First round on a smaller threshold
    pairs = generate_unique_keys(keys,
                                 range(len(keys)),
                                 range(len(keys)),
                                 0,
                                 thr=thr)
    if len(pairs) == 0:
      break
    merge_inds = set(np.concatenate([list(p) for p in pairs]))
    fixed_inds = set(range(len(keys))) - merge_inds
    for i in fixed_inds:
      if keys[i] not in fixed_mappings:
        fixed_mappings[keys[i]] = mappings[keys[i]]
      else:
        fixed_mappings[keys[i]] = fixed_mappings[keys[i]] + mappings[keys[i]]
    mappings = {keys[i]: mappings[keys[i]] for i in merge_inds}

    while (len(pairs) > 0):
      print("Iteration %d, N_pairs %d" % (it, len(pairs)))
      it += 1
      np.random.shuffle(pairs)
      existing_inds = set()
      for pair in pairs:
        if pair[0] not in existing_inds and pair[1] not in existing_inds:
          if check_consistency(keys[pair[0]], keys[pair[1]], thr=thr):
            merged = tuple(np.max(np.stack([keys[pair[0]], keys[pair[1]]], 0), 0))
            v0 = mappings.pop(keys[pair[0]])
            v1 = mappings.pop(keys[pair[1]])
            if merged in mappings:
              mappings[merged] = list(set(mappings[merged] + v0 + v1))
            else:
              mappings[merged] = list(set(v0 + v1))
            existing_inds.add(pair[0])
            existing_inds.add(pair[1])
      keys = list(mappings.keys())
      pairs = generate_unique_keys(keys,
                                   range(len(keys)),
                                   range(len(keys)),
                                   0,
                                   thr=thr)
      if len(pairs) > 0:
        merge_inds = set(np.concatenate([list(p) for p in pairs]))
      else:
        merge_inds = set()
      fixed_inds = set(range(len(keys))) - merge_inds
      for i in fixed_inds:
        k = keys[i]
        if k not in fixed_mappings:
          fixed_mappings[k] = mappings[k]
        else:
          fixed_mappings[k] = fixed_mappings[k] + mappings[k]

      if len(pairs) > 0:
        mappings = {keys[i]: mappings[keys[i]] for i in merge_inds}

    mappings = deepcopy(fixed_mappings)
    fixed_mappings.clear()

  assert sum([len(v) for v in mappings.values()]) == len(all_keys)
  for key in mappings:
    assert all([check_consistency(key, v, thr=thr) for v in mappings[key]])
  return mappings

def first_round(X, gene_names, Z, thr, seed=None):
  groups = generate_groups(X, Z, thr=thr, seed=seed)
  refined_mappings = merge_keys(list(groups.keys()), thr=thr, seed=seed)
  refined_groups = {}
  for key in refined_mappings:
    new_group = set()
    for k in refined_mappings[key]:
      new_group |= groups[k]
    if len(new_group) >= 5:
      refined_groups[key] = [gene_names[i] for i in new_group]
  return refined_groups

def second_round(refined_groups, weights, alpha=0.):
  # Generate distance/similarity matrix
  keys = np.array(sorted(refined_groups.keys()))
  similarity_mat = np.zeros((len(refined_groups), len(refined_groups), 2))
  for i, key in enumerate(keys):
    co_appearing = np.sign(key+1).reshape((1, -1)) * np.sign(keys+1)
    inconsistent = np.sign(np.abs((key+1).reshape((1, -1)) - (keys+1)) * co_appearing)
    similarity_mat[i, :, 0] = (inconsistent * weights.reshape((1, -1))).sum(1)
    similarity_mat[i, :, 1] = (co_appearing * weights.reshape((1, -1))).sum(1)
  sim_mat = alpha * (27 - (similarity_mat[:, :, 1] - similarity_mat[:, :, 0]))/27. + \
            similarity_mat[:, :, 0]/(similarity_mat[:, :, 1] + 1e-10) # Up to tune
  sim_mat = sim_mat/sim_mat.max()
  np.fill_diagonal(sim_mat, 0.)

  # Clustering
  cl = AgglomerativeClustering(n_clusters=2,
                               #memory='agg_merge_log.txt',
                               affinity='precomputed',
                               linkage='average') # Up to tune
  cl.fit(sim_mat)
  ii = itertools.count(sim_mat.shape[0])
  order = [(next(ii), x[0], x[1]) for x in cl.children_]
  return order

def second_round_scipy(refined_groups, weights, alpha=0.):
  # Generate distance/similarity matrix
  keys = np.array(sorted(refined_groups.keys()))
  similarity_mat = np.zeros((len(refined_groups), len(refined_groups), 2))
  for i, key in enumerate(keys):
    co_appearing = np.sign(key+1).reshape((1, -1)) * np.sign(keys+1)
    inconsistent = np.sign(np.abs((key+1).reshape((1, -1)) - (keys+1)) * co_appearing)
    similarity_mat[i, :, 0] = (inconsistent * weights.reshape((1, -1))).sum(1)
    similarity_mat[i, :, 1] = (co_appearing * weights.reshape((1, -1))).sum(1)
  sim_mat = alpha * (27 - (similarity_mat[:, :, 1] - similarity_mat[:, :, 0]))/27. + \
            similarity_mat[:, :, 0]/(similarity_mat[:, :, 1] + 1e-10) # Up to tune
  sim_mat = sim_mat/sim_mat.max()
  np.fill_diagonal(sim_mat, 0.)

  condensed_dist = np.concatenate([sim_mat[i, (i+1):] for i in range(sim_mat.shape[0])])
  link = linkage(condensed_dist, "average")
  return link

def report(refined_groups, order, label, n_clusters, ending_thr=0.97):
  # Label preprocessing
  label_clusters = {k: set(v) for k, v in label.items()}
  label_genes = set()
  for k in label_clusters:
    label_genes |= label_clusters[k]

  # Merge clusters
  keys = np.array(sorted(refined_groups.keys()))
  generated = False
  groups = {}
  for i, key in enumerate(keys):
    groups[i] = refined_groups[tuple(key)]

  consistent, total = restricted_rand_index(groups, label)
  trajectory = [(len(groups), consistent, total)]
  for i, merge in enumerate(order):
    merge_A = groups.pop(merge[1])
    merge_B = groups.pop(merge[2])
    _consistent, _total = cross_consistency(set(merge_A), set(merge_B), label_clusters, label_genes)
    groups[merge[0]] = merge_A + merge_B
    consistent += _consistent
    total += _total
    trajectory.append((len(groups), consistent, total))
    if len(groups) == n_clusters:
      output_groups = deepcopy(groups)
      generated = True
    if float(consistent)/total < ending_thr and generated:
      break

  # Line in the order of: [n_clusters, total pairs, consistent pairs(recall), precision]
  xs = [t[0] for t in trajectory]
  cs = [t[1] for t in trajectory]
  ts = [t[2] for t in trajectory]
  ratios = [float(c)/t for c, t in zip(cs, ts)]
  line = np.stack([xs, ts, cs, ratios], 1)
  return output_groups, line


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='CSPA for ensemble clustering')
  parser.add_argument(
      '-n',
      action='append',
      dest='thr',
      default=[],
      help='Threshold for input genes')

  args = parser.parse_args()
  thr = int(args.thr[0])
  first_round_thr = 4
  n_threads = 4
  seed = 123
  
  ground_truth_clusters = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))
  samples = merge_samples_msp
  file_list =['../summary/mspminer_%s.txt' % name for name in samples]
  #X, gene_names = preprocess_files(file_list, threshold=thr)
  X, gene_names = pickle.load(open('../summary/mspminer_X_%d.pkl' % thr, 'rb'))
  cohort_sizes = load_cohort_sizes(samples)
  consistency_mat = np.mean(load_inter_cohort_consistency(merge_samples_msp), 0)

  # Cohorts have weight related to their sizes, 0.3 is a scaling factor
  #sample_weights = (np.array(cohort_sizes)/max(cohort_sizes))**(0.3)
  sample_weights = ((consistency_mat ** 4)/(consistency_mat.max() ** 4)).astype(float)

  label1 = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))
  label2 = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

  ### FIRST ROUND ###
  # Z, k = initialize_Z(X, seed=147)
  # refined_groups = first_round(X, gene_names, Z, first_round_thr, seed=seed)
  # with open('./CSPA_first_round_merge_%d_%d_%d.pkl' % (thr, first_round_thr, seed), 'wb') as f:
  #   pickle.dump(refined_groups, f)
  refined_groups = pickle.load(open('./CSPA_first_round_merge_%d_%d_%d.pkl' % (thr, first_round_thr, seed), 'rb'))

  ### SECOND ROUND ###
  merge_order = second_round(refined_groups, sample_weights, alpha=0.)
  _, pr_line_species = report(refined_groups, merge_order, label1, 1000, ending_thr=0.97)
  _, pr_line_strain = report(refined_groups, merge_order, label2, 1000, ending_thr=0.85)