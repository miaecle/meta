# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
from MM import adjusted_rand_index, build_clusters
import numpy as np
import matplotlib.pyplot as plt

ref = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))
ref = {k:ref[k] for k in ref if len(ref[k]) > 100}
cts = {k:len(ref[k]) for k in ref}

X, gene_names = pickle.load(open('../summary/mspminer_X.pkl', 'rb'))
#Z_c = pickle.load(open('../utils/Z_full_init_136.pkl', 'rb'))[0]
#Z_c = build_clusters(Z_c, gene_names)

for study in np.arange(X.shape[1]):
  inds = np.where(X[:, study] >= 0)[0]
  Z_c = build_clusters(X[inds,  study], [gene_names[i] for i in inds])
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
      #print("Skipping %s" % k)
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
      if sample_cs_coverage[i] < 0.01:
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
  
  plt.clf()
  cts = [p[0] for p in results]
  for point in results:
    c = (np.log(point[0]) - np.log(min(cts)))/(np.log(max(cts)) - np.log(min(cts)))
    color = (0, (1-c)*1, c*1, 0.5)
    plt.plot([point[3]], [point[2]], '.', c=color)
  plt.xlim(-0.02, 1.02)
  plt.xlabel('Recall')
  plt.ylim(-0.02, 1.02)
  plt.ylabel('Precision')
  plt.savefig('pr%d.png' % study)

"""
Number of genes: 801272 2078341 370240
ARI: 0.524765
Mean precision 0.374895
Mean recall 0.779249
Prediction precision 0.382102
Sample recall 0.840051

Number of genes: 1421093        2078341 510564
ARI: 0.476230
Mean precision 0.377649
Mean recall 0.772715
Prediction precision 0.376323

Sample recall 0.811511
Number of genes: 1774997        2078341 670239
ARI: 0.486955
Mean precision 0.385123
Mean recall 0.745445
Prediction precision 0.393364
Sample recall 0.805716

Number of genes: 1332499        2078341 570675
ARI: 0.523668
Mean precision 0.386897
Mean recall 0.760699
Prediction precision 0.405386
Sample recall 0.826968

Number of genes: 861117 2078341 510877
ARI: 0.528375
Mean precision 0.394979
Mean recall 0.861903
Prediction precision 0.379157
Sample recall 0.863855

Number of genes: 1223918        2078341 661008
ARI: 0.467444
Mean precision 0.354767
Mean recall 0.767469
Prediction precision 0.335474
Sample recall 0.832925

Number of genes: 1236625        2078341 555831
ARI: 0.521362
Mean precision 0.377125
Mean recall 0.776029
Prediction precision 0.377464
Sample recall 0.831517

Number of genes: 842428 2078341 384427
ARI: 0.445278
Mean precision 0.332414
Mean recall 0.652101
Prediction precision 0.366817
Sample recall 0.730887

Number of genes: 1748330        2078341 635987
ARI: 0.474985
Mean precision 0.382353
Mean recall 0.745997
Prediction precision 0.390965
Sample recall 0.809206

Number of genes: 1031616        2078341 558671
ARI: 0.539986
Mean precision 0.391936
Mean recall 0.828078
Prediction precision 0.390879
Sample recall 0.859697

Number of genes: 1730884        2078341 625071
ARI: 0.465384
Mean precision 0.383685
Mean recall 0.749792
Prediction precision 0.378433
Sample recall 0.793870

Number of genes: 1162572        2078341 652428
ARI: 0.522551
Mean precision 0.385724
Mean recall 0.814312
Prediction precision 0.392355
Sample recall 0.837267

Number of genes: 1101707        2078341 488209
ARI: 0.516416
Mean precision 0.387793
Mean recall 0.784798
Prediction precision 0.394181
Sample recall 0.834946

Number of genes: 886875 2078341 541867
ARI: 0.555282
Mean precision 0.395625
Mean recall 0.843092
Prediction precision 0.389476
Sample recall 0.866064

Number of genes: 707632 2078341 456176
ARI: 0.563895
Mean precision 0.390472
Mean recall 0.853254
Prediction precision 0.373452
Sample recall 0.871738

Number of genes: 1095893        2078341 631821
ARI: 0.528867
Mean precision 0.390172
Mean recall 0.788805
Prediction precision 0.401477
Sample recall 0.837425

Number of genes: 666292 2078341 471216
ARI: 0.530791
Mean precision 0.363333
Mean recall 0.808839
Prediction precision 0.344946
Sample recall 0.858103

Number of genes: 136262 2078341 110154
ARI: 0.384790
Mean precision 0.267900
Mean recall 0.739057
Prediction precision 0.257363
Sample recall 0.795097

Number of genes: 210857 2078341 106586
ARI: 0.406694
Mean precision 0.328232
Mean recall 0.805064
Prediction precision 0.290454
Sample recall 0.813026

Number of genes: 317414 2078341 185038
ARI: 0.536932
Mean precision 0.368223
Mean recall 0.821316
Prediction precision 0.371889
Sample recall 0.849455

Number of genes: 156081 2078341 113315
ARI: 0.312776
Mean precision 0.247591
Mean recall 0.725367
Prediction precision 0.251806
Sample recall 0.815238

Number of genes: 316939 2078341 228852
ARI: 0.438209
Mean precision 0.267355
Mean recall 0.781305
Prediction precision 0.244079
Sample recall 0.853078

Number of genes: 659115 2078341 271516
ARI: 0.470891
Mean precision 0.354790
Mean recall 0.793640
Prediction precision 0.348220
Sample recall 0.833203

Number of genes: 237166 2078341 160404
ARI: 0.447723
Mean precision 0.309465
Mean recall 0.884003
Prediction precision 0.250597
Sample recall 0.888461

Number of genes: 770545 2078341 325610
ARI: 0.422799
Mean precision 0.348558
Mean recall 0.763502
Prediction precision 0.325348
Sample recall 0.800170

Number of genes: 345673 2078341 235774
ARI: 0.557497
Mean precision 0.374064
Mean recall 0.828116
Prediction precision 0.354297
Sample recall 0.862374

Number of genes: 527225 2078341 337793
ARI: 0.523594
Mean precision 0.382205
Mean recall 0.846761
Prediction precision 0.360981
Sample recall 0.854283
"""