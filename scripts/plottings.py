# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 12 15:09:09 2019

# @author: zqwu
# """

# ### Plot thetas in the MM model ###
# import pickle

# prior, thetas = pickle.load(open('../MM_init_on_147_k=4004.pkl', 'rb'))
# Z = pickle.load(open('../Z.pkl', 'rb'))

# Z_, cts = np.unique(Z, return_counts=True)
# Z_ = Z_[np.where(cts > 100)]

# for i, theta in enumerate(thetas):
#   mat = np.exp(theta[Z_])
#   order = sorted(np.arange(mat.shape[0]), key=lambda x: np.argmax(mat[x]))
#   plt.clf()
#   plt.imshow(mat[order])
#   plt.savefig('%d.png' % i, dpi=600)

### Plot pr curve for base clusterings and CSPA ###
import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import itertools
import threading
from multiprocessing import Process, Pool
from functools import partial
import multiprocessing as mp
from bisect import bisect_left
from similarity import restricted_rand_index
import pandas as pd
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from CSPA import second_round, report
from MM import build_clusters
from samples import merge_samples_msp
import csv

labels = pickle.load(open('../utils/Species_Gene_Array_0.9_new.pkl', 'rb'))
X, gene_names = pickle.load(open('../summary/mspminer_X_1.pkl', 'rb'))

bases_precs = []
bases_recas = []

f = open('../utils/base_clusters_pr_new.csv', 'w')
writer = csv.writer(f)
writer.writerow(['Name', 'Consistent pairs - species', 'Total pairs', 'Ratio - species'])

for sample_id, name in enumerate(merge_samples_msp):  
  inds = np.where(X[:, sample_id] >= 0)[0]
  X_c = build_clusters(X[inds, sample_id], [gene_names[i] for i in inds])
  X_c2 = {k:set(X_c[k]) for k in X_c if len(X_c[k]) > 100}
  consistent, total = restricted_rand_index(X_c2, labels)
  bases_precs.append(consistent/total)
  bases_recas.append(consistent)
  writer.writerow([name, consistent, total, consistent/total])

f.close()

lines = []
for thr in range(1, 11):
  print("On thr %d" % thr)
  refined_groups = pickle.load(open('./CSPA_first_round_merge_%d_4_123.pkl' % thr, 'rb'))
  merge_order = second_round(refined_groups, np.ones((27,)), alpha=0.)
  _, pr_line_species = report(refined_groups, merge_order, labels, 1000, ending_thr=0.97)

  lines.append(pr_line_species)

labels = [str(i) for i in range(1, 11)]



# MM_prec_strain = [0.672, 0.776, 0.804, 0.812, 0.824, 
#                  0.837, 0.840, 0.849, 0.859, 0.862, 0.861]
# MM_reca_strain = np.array([2161, 1603, 1318, 1166, 1016, 902, 800, 716, 642, 580, 526]) * 1e6
# MM_prec_specie = [0.844, 0.927, 0.945, 0.949, 0.957, 
#                  0.963, 0.962, 0.968, 0.976, 0.977, 0.979]
# MM_reca_specie = np.array([2714, 1914, 1549, 1363, 1180, 1037, 917, 816, 730, 657, 598]) * 1e6
#
plt.clf()
plt.plot(bases_precs, bases_recas, '.', label='base')
# plt.plot(MM_prec_specie[1:], MM_reca_specie[1:], 'r.-', label='MM')
for i, l in enumerate(lines):
  plt.plot(l[:-100, 3], l[:-100, 2], label=labels[i])
plt.legend()
plt.savefig('pr_species2.png', dpi=300)

# plt.clf()
# plt.plot(bases_prec_strain, bases_reca_strain, '.', label='base')
# plt.plot(MM_prec_strain[1:], MM_reca_strain[1:], 'r.-', label='MM')
# for i, l in enumerate(lines2):
#  plt.plot(l[:, 3], l[:, 2], label=labels[i])
# plt.legend()
# plt.savefig('pr_strain.png', dpi=300)

# ### Compare different second round methods ###
# import pickle
# import numpy as np
# import itertools
# import threading
# from multiprocessing import Process, Pool
# from functools import partial
# import multiprocessing as mp
# from bisect import bisect_left
# from similarity import restricted_rand_index
# import pandas as pd
# import matplotlib
# matplotlib.use('AGG')
# import matplotlib.pyplot as plt
# from CSPA import second_round, report
# from CSPA_extend2 import second_round_extended, report_extended

# # label1 = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))
# label1 = pickle.load(open('../utils/Species_Gene_Array_0.9_new.pkl', 'rb'))
# label2 = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

# # bases = np.array(pd.read_csv('../utils/base_clusterings_pr.csv'))
# bases = np.array(pd.read_csv('../utils/base_clusterings_pr_2.csv'))
# bases_prec_strain = [line[5] for line in bases]
# bases_reca_strain = [float(line[2][:-1]) * 1e6 for line in bases]
# bases_prec_specie = [line[6] for line in bases]
# bases_reca_specie = [float(line[3][:-1]) * 1e6 for line in bases]


# MM_prec_strain = [0.672, 0.776, 0.804, 0.812, 0.824, 
#                   0.837, 0.840, 0.849, 0.859, 0.862, 0.861]
# MM_reca_strain = np.array([2161, 1603, 1318, 1166, 1016, 902, 800, 716, 642, 580, 526]) * 1e6
# MM_prec_specie = [0.844, 0.927, 0.945, 0.949, 0.957, 
#                   0.963, 0.962, 0.968, 0.976, 0.977, 0.979]
# MM_reca_specie = np.array([2714, 1914, 1549, 1363, 1180, 1037, 917, 816, 730, 657, 598]) * 1e6

# thr = 1

# refined_groups = pickle.load(open('./CSPA_first_round_merge_%d_4_123.pkl_bkp' % thr, 'rb'))
# n_genes_refined = sum([len(v) for v in refined_groups.values()])
# merge_order = second_round(refined_groups, np.ones((27,)), alpha=0.)
# _, pr_line_species = report(refined_groups, merge_order, label1, 1000, ending_thr=0.97)
# _, pr_line_strain = report(refined_groups, merge_order, label2, 1000, ending_thr=0.85)

# full_groups = pickle.load(open('./CSPA_first_round_merge_%d_4_123.pkl' % thr, 'rb'))
# n_genes_full = sum([len(v) for v in full_groups.values()])
# merge_order = second_round(full_groups, np.ones((27,)), alpha=0.)
# _, pr_line_species2 = report(full_groups, merge_order, label1, 1000, ending_thr=0.97)
# _, pr_line_strain2 = report(full_groups, merge_order, label2, 1000, ending_thr=0.85)

# X, gene_names = pickle.load(open('../summary/mspminer_X_%d.pkl' % thr, 'rb'))
# gene_names = np.array(gene_names)
# extra_mappings, merge_order = second_round_extended(refined_groups, X, gene_names, np.ones((27,)))
# _, pr_line_species3 = report_extended(
#   refined_groups, extra_mappings, merge_order, label1, n_clusters=900, include_thr=0.8, ending_thr=0.97)
# _, pr_line_strain3 = report_extended(
#   refined_groups, extra_mappings, merge_order, label2, n_clusters=900, include_thr=0.8, ending_thr=0.85)

# plt.clf()
# plt.plot(bases_prec_specie, bases_reca_specie, '.', label='base')
# plt.plot(MM_prec_specie[1], MM_reca_specie[1], 'r.', label='MM')
# plt.plot(pr_line_species[:, 3], pr_line_species[:, 2], label='core set')
# plt.plot(pr_line_species2[:, 3], pr_line_species2[:, 2], label='extended before')
# plt.plot(pr_line_species3[:, 3], pr_line_species3[:, 2], label='extended after')
# plt.legend()
# plt.plot()
# plt.savefig('extend1species.png', dpi=300)

# plt.clf()
# plt.plot(bases_prec_strain, bases_reca_strain, '.', label='base')
# plt.plot(MM_prec_strain[1], MM_reca_strain[1], 'r.', label='MM')
# plt.plot(pr_line_strain[:, 3], pr_line_strain[:, 2], label='core set')
# plt.plot(pr_line_strain2[:, 3], pr_line_strain2[:, 2], label='extended before')
# plt.plot(pr_line_strain3[:, 3], pr_line_strain3[:, 2], label='extended before')
# plt.legend()
# plt.plot()
# plt.savefig('extend1strain.png', dpi=300)
