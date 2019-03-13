import os
import numpy as np
import csv
from samples import valid_samples
from similarity import adjusted_rand_index
import pickle

summary_dir = "/mnt/osf1/user/wuzhq/meta/summary"
ref = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))

ARIs = np.zeros((len(valid_samples), len(valid_samples))) - 1
ref_ARIs = np.zeros((len(valid_samples),)) - 1

for i_c, cohort in enumerate(valid_samples):
  file_path = os.path.join(summary_dir, "mspminer_%s.txt" % cohort)
  ref_ARIs[i_c] = adjusted_rand_index(file_path, ref)
  for j_c in range(i_c+1, len(valid_samples)):
    file_path2 = os.path.join(summary_dir, "mspminer_%s.txt" % valid_samples[j_c])
    ARIs[i_c, j_c] = adjusted_rand_index(file_path, file_path2)


with open('output.csv', 'w') as f:
  writer = csv.writer(f)
  header = ['Name', 'Ref'] + valid_samples
  writer.writerow(header)
  for i in range(len(valid_samples)):
    line = [valid_samples[i]] + [ref_ARIs[i]] + list(ARIs[i])
    writer.writerow(line)
