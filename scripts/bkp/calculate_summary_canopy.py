from samples import valid_samples
import os
import numpy as np

canopy_output_dir = "/mnt/osf1/user/wuzhq/meta/output"
summary_dir = "/mnt/osf1/user/wuzhq/meta/summary"

for i_c, cohort in enumerate(valid_samples):
  print("On file %s" % cohort)
  file_path = os.path.join(canopy_output_dir, "canopy_%s" % cohort)
  summary_path = os.path.join(summary_dir, "canopy_%s.txt" % cohort)

  if os.path.exists(summary_path):
    continue
  if not os.path.exists(file_path):
    continue

  genes_list = {}

  with open(os.path.join(file_path, 'clusters.c'), 'r') as f:
    for i, line in enumerate(f):
      line = line.split()
      cluster_name = int(line[0].split('_')[-1])
      gene_name = line[1]
      if not gene_name in genes_list:
        genes_list[gene_name] = []
      genes_list[gene_name].append('c' + str(cluster_name))

  clusters = []
  for g in genes_list:
    clusters.extend(genes_list[g])

  cluster_id, cts = np.unique(clusters, return_counts=True)
  valid_clusters = cluster_id[np.where(cts >= 100)]

  with open(summary_path, 'w') as f:
    for key in sorted(genes_list.keys()):
      if genes_list[key][0] in valid_clusters:
        f.write(key+',' + ','.join(genes_list[key]) + '\n')
