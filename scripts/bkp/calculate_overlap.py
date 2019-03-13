import os
import numpy as np
import csv
from samples import valid_samples

summary_dir = "/mnt/osf1/user/wuzhq/meta/summary"

shared_mat = np.zeros((len(valid_samples), len(valid_samples))) - 1
n_genes = np.zeros((len(valid_samples),)) - 1

for i_c, cohort in enumerate(valid_samples):
  file_path = os.path.join(summary_dir, "mspminer_%s.txt" % cohort)
  genes = []
  with open(file_path, 'r') as f:
    for line in f:
      genes.append(line.split(',')[0])
  assert len(set(genes)) == len(genes)
  genes = set(genes)
  n_genes[i_c] = len(genes)
  for j_c in range(i_c+1, len(valid_samples)):
    file_path2 = os.path.join(summary_dir, "mspminer_%s.txt" % valid_samples[j_c])
    genes2 = []
    with open(file_path2, 'r') as f:
      for line in f:
        genes2.append(line.split(',')[0])
    shared_mat[i_c, j_c] = len(set(genes) & set(genes2))

np.save('shared_mat', shared_mat)
np.save('n_genes', n_genes)
with open('output.csv', 'w') as f:
  writer = csv.writer(f)
  header = ['Name', 'Total_ct'] + valid_samples
  writer.writerow(header)
  for i in range(len(valid_samples)):
    line = [valid_samples[i]] + [n_genes[i]] + list(shared_mat[i])
    writer.writerow(line)
