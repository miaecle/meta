from samples import valid_samples
import os

mspminer_output_dir = "/mnt/osf1/user/wuzhq/meta/output"
summary_dir = "/mnt/osf1/user/wuzhq/meta/summary"

for i_c, cohort in enumerate(valid_samples):
  file_path = os.path.join(mspminer_output_dir, "mspminer_%s" % cohort)
  summary_path = os.path.join(summary_dir, "mspminer_%s.txt" % cohort)

  if os.path.exists(summary_path):
    continue
  genes_list = {}

  clusters = [p for p in os.listdir(file_path) if p.startswith('msp_')]
  for cluster in clusters:
    cluster_id = int(cluster[4:])
    with open(os.path.join(file_path, cluster, 'modules.tsv'), 'r') as f:
      for i, line in enumerate(f):
        if i==0: 
          continue
        gene_name = line.split()[2]
        point_type = line[0]
        if not gene_name in genes_list:
          genes_list[gene_name] = []
        genes_list[gene_name].append(point_type + str(cluster_id))

  with open(summary_path, 'w') as f:
    for key in sorted(genes_list.keys()):
      f.write(key+',' + ','.join(genes_list[key]) + '\n')
