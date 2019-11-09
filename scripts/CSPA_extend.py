import numpy as np
import pickle
from samples import valid_samples, merge_samples, merge_samples_msp
from similarity import load_samples, adjusted_rand_index
from BCE import generate_n_matrix, preprocess_files, initialize_Z, initialize_Z_spread, build_clusters, load_cohort_sizes, load_inter_cohort_consistency
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
import time
import threading
import argparse
from CSPA import first_round, second_round, report, check_consistency, generate_unique_keys

class key_locator(object):
  def __init__(self, keys, thr=3):
    self.thr = 3
    self.keys = np.array(keys)
    self.mappings = [{} for _ in range(self.keys.shape[1])]
    for i in range(self.keys.shape[1]):
      unique_assigns = np.unique(self.keys[:, i])
      for assign in unique_assigns:
        if assign >= 0:
          self.mappings[i][assign] = set(np.where(self.keys[:, i] == assign)[0])

  def locate(self, k, ratio=1.0):
    res = []
    for i, a in enumerate(k):
      if a >= 0:
        if not a in self.mappings[i]:
          continue
        res.extend(list(self.mappings[i][a]))
    if len(res) == 0:
      #print("No match")
      return None
    n_nonzero = (k >= 0).sum()
    assigns, cts = np.unique(res, return_counts=True)
    if cts.max() < int(n_nonzero * ratio):
      #print("Too few matches")
      return None
    else:
      k_ind = np.random.choice(assigns[np.where(cts == cts.max())])
      return tuple(self.keys[k_ind])

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
  assert thr <= 3
  n_threads = 4
  seed = 123
  
  samples = merge_samples_msp
  file_list =['../summary/mspminer_%s.txt' % name for name in samples]
  #X, gene_names = preprocess_files(file_list, threshold=thr)
  X, gene_names = pickle.load(open('../summary/mspminer_X_%d.pkl' % thr, 'rb'))
  cohort_sizes = load_cohort_sizes(samples)
  consistency_mat = np.mean(load_inter_cohort_consistency(merge_samples_msp), 0)

  # Cohorts have weight related to their sizes, 0.3 is a scaling factor
  #sample_weights = (np.array(cohort_sizes)/max(cohort_sizes))**(0.3)
  sample_weights = (consistency_mat ** 4)/(consistency_mat.max() ** 4)

  label1 = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))
  label2 = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

  ### FIRST ROUND ###
  selected_inds = [i for i in range(len(X)) if (X[i] >= 0).sum() >= first_round_thr]
  selected_X = X[np.array(selected_inds)]
  selected_gene_names = [gene_names[i] for i in selected_inds]
  selected_Z, k = initialize_Z(selected_X, seed=147)
  refined_groups = first_round(selected_X, 
                               selected_gene_names, 
                               selected_Z, 
                               first_round_thr, 
                               seed=seed)

  with open('./CSPA_first_round_merge_%d_%d_%d.pkl_bkp' % (thr, first_round_thr, seed), 'wb') as f:
    pickle.dump(refined_groups, f)

  ### Add Extra Genes ###
  assignments = list(refined_groups.keys())
  locator = key_locator(assignments)
  included = set()
  for key in assignments:
    included |= set(refined_groups[key])

  ct = 0
  for x, name in zip(X, gene_names):
    if name not in included:
      key = locator.locate(x)
      if key is not None:
        refined_groups[key].append(name)
      else:
        ct += 1
  print("%d genes excluded" % ct)

  with open('./CSPA_first_round_merge_%d_%d_%d.pkl' % (thr, first_round_thr, seed), 'wb') as f:
    pickle.dump(refined_groups, f)

  ### SECOND ROUND ###
  # merge_order = second_round(refined_groups, sample_weights)
  # _, pr_line_species = report(refined_groups, merge_order, label1, 1000, ending_thr=0.97)
  # _, pr_line_strain = report(refined_groups, merge_order, label2, 1000, ending_thr=0.85)