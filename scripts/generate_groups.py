#from dynamicTreeCut import cutreeHybrid
from scipy.spatial.distance import pdist
import numpy as np
from CSPA import second_round, second_round_scipy
from samples import merge_samples_msp
from MM import load_cohort_sizes, load_inter_cohort_consistency
import pickle
import matplotlib.pyplot as plt
import pandas as pd

thr = 1
first_round_thr = 4
n_threads = 4
seed = 123

samples = merge_samples_msp
file_list =['../summary/mspminer_%s.txt' % name for name in samples]
cohort_sizes = load_cohort_sizes(samples)
consistency_mat = np.mean(load_inter_cohort_consistency(merge_samples_msp), 0)
sample_weights = ((consistency_mat ** 4)/(consistency_mat.max() ** 4)).astype(float)

label1 = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))

refined_groups = pickle.load(open('./CSPA_first_round_merge_%d_%d_%d.pkl' % (thr, first_round_thr, seed), 'rb'))
links = second_round_scipy(refined_groups, sample_weights, alpha=0.)

sim_thr = 0.2
groups = {i: refined_groups[tuple(k)] for i, k in enumerate(np.array(sorted(refined_groups.keys())))}
conf = {i: 1.0 for i in groups.keys()}
group_id = max(groups.keys()) + 1
for l in links:
  if l[2] > sim_thr:
    break
  set1 = groups.pop(l[0])
  set2 = groups.pop(l[1])
  groups[group_id] = set1 + set2
  conf[group_id] = 1 - l[2]
  group_id += 1
groups = {k: groups[k] for k in groups if len(groups[k]) > 500 and len(groups[k]) < 5000}


df = np.array(pd.read_csv('./all_cluster_quality.csv', header=None))
all_completeness = {line[0]: float(line[1]) for line in df}
all_contamination = {line[0]: float(line[2]) for line in df}

contam = {k: all_contamination['cluster%d' % k] for k in groups if 'cluster%d' % k in all_contamination}
comp = {k: all_completeness['cluster%d' % k] for k in groups if 'cluster%d' % k in all_contamination}

ct = 0
for k in contam:
  if contam[k] <= 5 and comp[k] >= 70:
    ct += 1
print("%d, %f" % (ct, ct/len(groups)))
len(groups)

comp_vals = [comp[k] for k in comp]
contam_vals = [contam[k] for k in comp]
inds = list(reversed(np.argsort(comp_vals)))

comp_vals = np.array([comp_vals[i] for i in inds])
contam_vals = np.array([contam_vals[i] for i in inds])
plt.scatter(np.arange(len(comp_vals)), comp_vals, s=2)
plt.scatter(np.arange(len(comp_vals)), contam_vals, s=2)
print(np.where(contam_vals > 20)[0].shape[0])
print(np.where(contam_vals > 20)[0].shape[0]/len(contam_vals))

groups = {k: groups[k] for k in groups if contam[k] < 20}
conf = {k: conf[k] for k in groups}
contam = {k: all_contamination['cluster%d' % k] for k in groups}
comp = {k: all_completeness['cluster%d' % k] for k in groups}
assembled_data = {'clusterings': groups,
                  'clustering_confidences': conf,
                  'contamination': contam,
                  'completeness': comp}

high_quality = {k: groups[k] for k in groups if contam[k] < 5 and comp[k] > 70}
X, gene_names = pickle.load(open('../summary/mspminer_X_1.pkl', 'rb'))
X_ = {g_n: x_ for g_n, x_ in zip(gene_names, X)}

potential_studies = {}
for k in high_quality:
  potential_studies[k] = []
  g = groups[k]
  xs = np.stack([X_[g_n] for g_n in g], 0)
  ids, cts = np.unique(np.where(xs < 0)[1], return_counts=True)
  for i, ct in zip(ids, cts):
    if ct < xs.shape[0] * 0.1:
      potential_studies[k].append((merge_samples_msp[i], ct, len(xs)))
  if len(potential_studies[k]) == 0:
    for i, ct in zip(ids, cts):
      if ct < xs.shape[0] * 0.15:
        potential_studies[k].append((merge_samples_msp[i], ct, len(xs)))
assembled_data['potential_studies'] = potential_studies

with open('./thr1_results_1070_final.pkl', 'wb') as f:
  pickle.dump(assembled_data, f)
