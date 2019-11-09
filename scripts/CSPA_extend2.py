import numpy as np
import pickle
from samples import valid_samples, merge_samples, merge_samples_msp
from similarity import load_samples, adjusted_rand_index, restricted_rand_index, cross_consistency, worker_restricted_index
from BCE import generate_n_matrix, preprocess_files, initialize_Z, initialize_Z_spread, build_clusters, load_cohort_sizes, load_inter_cohort_consistency
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
import time
import threading
import argparse
import itertools
from CSPA import first_round, second_round, report, check_consistency, generate_unique_keys

### Extra Genes ###
def find_matching_groups(x, name, reversed_mapping, original_groups):
  choices = []
  gs = []
  assert len(x) == len(original_groups)
  for i, assign in enumerate(x):
    if assign >= 0:
      for g in original_groups[i][assign]:
        if g in reversed_mapping:
          choices.append(reversed_mapping[g])
          gs.append(g)
  total_ct = len(choices)
  _, inds, cts = np.unique(choices, return_index=True, return_counts=True)
  return {gs[ind]: ct/total_ct for ind, ct in zip(inds, cts)}

### Add Extra Genes ###
def add_extra_genes(final_groups, extra_mappings, include_thr=0.8):
  reversed_mapping = {}
  for key in final_groups:
    for g in final_groups[key]:
      reversed_mapping[g] = key
  
  enlarged_final_groups = {k:[] for k in final_groups}
  unmapped_ct = 0
  for extra_g in extra_mappings:
    mappings = extra_mappings[extra_g]
    new_mappings = {}
    for g in mappings:
      if reversed_mapping[g] not in new_mappings:
        new_mappings[reversed_mapping[g]] = mappings[g]
      else:
        new_mappings[reversed_mapping[g]] += mappings[g]
    for k, v in new_mappings.items():
      if v > include_thr:
        enlarged_final_groups[k].append(extra_g)
        break
    else:
      unmapped_ct += 1
  print("%d genes excluded" % unmapped_ct)
  return enlarged_final_groups

def second_round_extended(refined_groups, 
                          X, 
                          gene_names, 
                          weights):
                    
  # Process raw inputs
  X_groups = []
  for i in range(X.shape[1]):
    valid_inds = np.where(X[:, i] >= 0)[0]
    X_groups.append(build_clusters(X[valid_inds][:, i], gene_names[valid_inds]))

  # Build reversed mapping
  keys = np.array(sorted(refined_groups.keys()))
  included = set()
  reversed_mapping = {}
  for i, key in enumerate(keys):
    included |= set(refined_groups[tuple(key)])
    for g in refined_groups[tuple(key)]:
      reversed_mapping[g] = i

  # Extra genes profiles
  extra_mappings = {}
  n_processed = 0
  for x, name in zip(X, gene_names):
    if name not in included:
      if n_processed % 10000 == 0:
        print("Processed %d genes" % n_processed)
      extra_mappings[name] = find_matching_groups(x, name, reversed_mapping, X_groups)
      n_processed += 1

  order = second_round(refined_groups, weights, 0.)
  return extra_mappings, order

def report_extended(refined_groups, 
                    extra_mappings,
                    order, 
                    label, 
                    n_clusters=1000,
                    include_thr=0.8,
                    ending_thr=0.97):
  # Label preprocessing
  label_clusters = {k: set(v) for k, v in label.items()}
  label_genes = set()
  for k in label_clusters:
    label_genes |= label_clusters[k]
  extra_shared = extra_mappings.keys() & label_genes
  label_ = {k:set(v) for k, v in label.items()}

  # Merge clusters
  keys = np.array(sorted(refined_groups.keys()))
  generated = False
  groups = {}
  for i, key in enumerate(keys):
    groups[i] = refined_groups[tuple(key)]

  consistent, total = restricted_rand_index(groups, label)
  trajectory = []
  for i, merge in enumerate(order):
    if len(groups) < n_clusters:
      break
    merge_A = groups.pop(merge[1])
    merge_B = groups.pop(merge[2])
    _consistent, _total = cross_consistency(set(merge_A), set(merge_B), label_clusters, label_genes)
    groups[merge[0]] = merge_A + merge_B
    consistent += _consistent
    total += _total
    


    if i%200 == 0 or len(groups) == n_clusters:
      print("%d clusters" % len(groups))
      enlarged_groups = add_extra_genes(groups, extra_mappings, include_thr=include_thr)

      consistent_temp = deepcopy(consistent)
      total_temp = deepcopy(total)
      _consistent, _total = worker_restricted_index(enlarged_groups.keys(), 
                                                    extra_shared, 
                                                    enlarged_groups, 
                                                    label_)
      consistent_temp += _consistent
      total_temp += _total
      for k in groups:
        if len(enlarged_groups[k]) > 0:
          _consistent, _total = cross_consistency(set(groups[k]), 
                                                  set(enlarged_groups[k]), 
                                                  label_clusters, 
                                                  label_genes)
          consistent_temp += _consistent
          total_temp += _total
      trajectory.append((len(groups), consistent_temp, total_temp))
      if len(groups) == n_clusters:
        generated = True
        out_group = deepcopy(groups)
        for k in out_group:
          out_group[k].extend(enlarged_groups[k])
  # Line in the order of: [n_clusters, total pairs, consistent pairs(recall), precision]
  xs = [t[0] for t in trajectory]
  cs = [t[1] for t in trajectory]
  ts = [t[2] for t in trajectory]
  ratios = [float(c)/t for c, t in zip(cs, ts)]
  line = np.stack([xs, ts, cs, ratios], 1)
  return out_group, line






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

  samples = merge_samples_msp
  file_list =['../summary/mspminer_%s.txt' % name for name in samples]
  #X, gene_names = preprocess_files(file_list, threshold=thr)
  X, gene_names = pickle.load(open('../summary/mspminer_X_%d.pkl' % thr, 'rb'))
  gene_names = np.array(gene_names)

  # Cohorts weights
  # cohort_sizes = load_cohort_sizes(samples)
  # sample_weights = (np.array(cohort_sizes)/max(cohort_sizes))**(0.3)
  # consistency_mat = np.mean(load_inter_cohort_consistency(merge_samples_msp), 0)
  # sample_weights = (consistency_mat ** 4)/(consistency_mat.max() ** 4)
  sample_weights = np.ones((27,))

  label1 = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))
  label2 = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

  ### FIRST ROUND ###
  refined_groups = pickle.load(open('./CSPA_first_round_merge_%d_4_123.pkl_bkp' % thr, 'rb'))

  extra_mappings, merge_order = second_round_extended(refined_groups, X, gene_names, sample_weights)
  _, line_species = report_extended(refined_groups, 
                                    extra_mappings, 
                                    merge_order, 
                                    label1, 
                                    n_clusters=1000, 
                                    include_thr=0.8, 
                                    ending_thr=0.97)
  _, line_strain = report_extended(refined_groups, 
                                   extra_mappings, 
                                   merge_order, 
                                   label2, 
                                   n_clusters=1000, 
                                   include_thr=0.8, 
                                   ending_thr=0.85)
  #[(10090, 779269545.0, 796197811.0), (9890, 779270169.0, 796938264.0), (9690, 781209578.0, 799338716.0), (9490, 793260126.0, 812053929.0), (9290, 802001193.0, 820910445.0), (9090, 805640058.0, 824596452.0), (8890, 807638064.0, 826680821.0), (8690, 817880774.0, 837351264.0), (8490, 828154918.0, 847829631.0), (8290, 833286201.0, 853044687.0), (8090, 837646610.0, 857561292.0), (7890, 840404034.0, 860385023.0), (7690, 855922455.0, 876272973.0), (7490, 870951361.0, 891953024.0), (7290, 880234412.0, 901542256.0), (7090, 886158979.0, 907573589.0), (6890, 895527267.0, 917144358.0), (6690, 916903959.0, 938967684.0), (6490, 931359447.0, 954140086.0), (6290, 938145384.0, 961182237.0), (6090, 952497328.0, 976304692.0), (5890, 955918563.0, 980084727.0), (5690, 967934271.0, 992352116.0), (5490, 979702406.0, 1004441983.0), (5290, 994966360.0, 1020371428.0), (5090, 1010390350.0, 1036399495.0), (4890, 1020840589.0, 1047119868.0), (4690, 1046162037.0, 1073375909.0), (4490, 1050795432.0, 1078241361.0), (4290, 1074418110.0, 1102772248.0), (4090, 1091668885.0, 1121078821.0), (3890, 1104556178.0, 1134566325.0), (3690, 1119351045.0, 1149882329.0), (3490, 1145112691.0, 1176482226.0), (3290, 1168899208.0, 1202034726.0), (3090, 1203857007.0, 1238361737.0), (2890, 1241310592.0, 1276994370.0), (2690, 1266655356.0, 1305428298.0), (2490, 1308495971.0, 1353790880.0), (2290, 1329380271.0, 1376274296.0), (2090, 1341865827.0, 1389711741.0), (1890, 1371792391.0, 1421733575.0), (1690, 1412005010.0, 1470754366.0), (1490, 1457060946.0, 1519314553.0), (1290, 1516256976.0, 1592366995.0)]
  #[(10090, 706692207.0, 796197811.0), (9890, 706692801.0, 796938264.0), (9690, 708164829.0, 799338716.0), (9490, 716340560.0, 812053929.0), (9290, 724909680.0, 820910445.0), (9090, 728420923.0, 824596452.0), (8890, 730337567.0, 826680821.0), (8690, 739181043.0, 837351264.0), (8490, 747542725.0, 847829631.0), (8290, 752239361.0, 853044687.0), (8090, 756245556.0, 857561292.0), (7890, 758744808.0, 860385023.0), (7690, 772111714.0, 876272973.0), (7490, 783978762.0, 891953024.0), (7290, 791757216.0, 901542256.0), (7090, 796547534.0, 907573589.0), (6890, 804374738.0, 917144358.0), (6690, 819424352.0, 938967684.0), (6490, 833079874.0, 954140086.0), (6290, 839447194.0, 961182237.0), (6090, 851632551.0, 976304692.0), (5890, 854559176.0, 980084727.0), (5690, 865010455.0, 992352116.0), (5490, 876030110.0, 1004441983.0), (5290, 888417209.0, 1020371428.0), (5090, 899526197.0, 1036399495.0), (4890, 907207732.0, 1047119868.0), (4690, 927800813.0, 1073375909.0), (4490, 932029277.0, 1078241361.0), (4290, 953535980.0, 1102772248.0), (4090, 967636849.0, 1121078821.0), (3890, 977699806.0, 1134566325.0), (3690, 991313902.0, 1149882329.0), (3490, 1011320416.0, 1176482226.0), (3290, 1029056007.0, 1202034726.0), (3090, 1054990113.0, 1238361737.0), (2890, 1083760424.0, 1276994370.0), (2690, 1107558431.0, 1305428298.0), (2490, 1139065410.0, 1353790880.0), (2290, 1156070525.0, 1376274296.0), (2090, 1166211281.0, 1389711741.0), (1890, 1189011363.0, 1421733575.0), (1690, 1221028497.0, 1470754366.0), (1490, 1254241294.0, 1519314553.0), (1290, 1296336511.0, 1592366995.0)]
  
