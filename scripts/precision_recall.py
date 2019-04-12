# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
from MM import adjusted_rand_index, build_clusters
import numpy as np

ref = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))
ref = {k:ref[k] for k in ref if len(ref[k]) > 100}
cts = {k:len(ref[k]) for k in ref}

X, gene_names = pickle.load(open('../summary/mspminer_X.pkl', 'rb'))
Z_c = pickle.load(open('../utils/Z_full_init_136.pkl', 'rb'))[0]
Z_c = build_clusters(Z_c, gene_names)
Z_c2 = {k:Z_c[k] for k in Z_c if len(Z_c[k]) > 100}


print("ARI: %f" % adjusted_rand_index(Z_c, ref))

keys1 = sorted(ref.keys())
genes1 = []
for k1 in keys1:
  genes1.extend(list(ref[k1]))
genes1 = set(genes1)
  
keys2 = sorted(Z_c2.keys())
genes2 = []
for k2 in keys2:
  genes2.extend(list(Z_c2[k2]))    
genes2 = set(genes2)

shared_genes = genes1 & genes2

Z_c2 = {k:set(Z_c2[k]) & shared_genes for k in Z_c2}

results = []
for k in ref:
  ref_c = set(ref[k]) & shared_genes
  if len(ref_c) < 100:
    print("Skipping %s" % k)
    continue
  
#  sample_cs = []
#  for k2 in Z_c2:
#    if len(set(Z_c2[k2]) & ref_c) >= 0.5 * len(set(Z_c2[k2])):
#      sample_cs.extend(list(Z_c2[k2]))
      
  sample_cs_coverage = {}
  for k2 in Z_c2:
    if len(set(Z_c2[k2])) > 0:
      sample_cs_coverage[k2] = len(set(Z_c2[k2]) & ref_c)/float(len(set(Z_c2[k2])))
  order = sorted(list(sample_cs_coverage.keys()), key=lambda x: -sample_cs_coverage[x])
  
  sample_cs = []
  for i in order[:10]:
    if len(sample_cs) >= len(ref_c):
      break
    if sample_cs_coverage[i] < 0.1:
      break
    sample_cs.extend(Z_c2[i])
    
  overlapping = float(len(set(sample_cs) & ref_c))
  precision = overlapping/(len(set(sample_cs)) + 1e-5)
  recall = overlapping/len(ref_c)
  results.append((len(ref_c), len(set(sample_cs)), precision, recall))
#  print("On %s, %d" % (k, len(ref_c)))
#  print("precision %f" % precision)
#  print("recall %f" % recall)

print("Mean precision %f" % np.mean([p[2] for p in results]))
print("Mean recall %f" % np.mean([p[3] for p in results]))
print("Prediction precision %f" % (sum([r[1] *r[2] for r in results])/sum([r[1] for r in results])))
print("Sample recall %f" % (sum([r[0] *r[3] for r in results])/sum([r[0] for r in results])))

#import matplotlib.pyplot as plt
#cts = [p[0] for p in results]
#for point in results:
#  c = (np.log(point[0]) - np.log(min(cts)))/(np.log(max(cts)) - np.log(min(cts)))
#  color = (0, (1-c)*1, c*1, 0.5)
#  plt.plot([point[2]], [point[1]], '.', c=color)
#plt.xlim(-0.02, 1.02)
#plt.xlabel('Recall')
#plt.ylim(-0.02, 1.02)
#plt.ylabel('Precision')
#plt.savefig('pr.png')
