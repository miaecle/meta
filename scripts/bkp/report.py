from MM import *
from similarity import restricted_rand_index

samples = merge_samples_msp
n_threads = 4
file_list =['../summary/mspminer_%s.txt' % name for name in samples]
cohort_sizes = load_cohort_sizes(samples)

# Cohorts have weight related to their sizes, 0.3 is a scaling factor
sample_weights = (np.array(cohort_sizes)/max(cohort_sizes))**(0.3)

label1 = pickle.load(open('../utils/Species_Gene_Array_0.9.pkl', 'rb'))
label2 = pickle.load(open('../utils/Strain_Gene_Array_0.9.pkl', 'rb'))

for thr in range(1):
  print("THRESHOLD: %d" % thr)
  X, gene_names = pickle.load(open('../summary/mspminer_X_%d.pkl' % thr, 'rb'))
  
  for i in range(7, 10):
    print("%d" % i)
    path = '../utils/MM_save/MM_save_%d_%d.pkl' % (thr, i)
    prior, thetas = pickle.load(open(path, 'rb'))
    k = prior.shape[0]
    mm = MM(X, k, Z_init=None, prior=prior, thetas=thetas, weights=sample_weights)
    Z_clusters = build_clusters(InferZ(mm, n_threads=n_threads), gene_names)
    
    print(restricted_rand_index(Z_clusters, label1))
    print(restricted_rand_index(Z_clusters, label2))
  print("\n\n")

results = {}
X, gene_names = pickle.load(open('../summary/mspminer_X_0.pkl', 'rb'))
for i in range(X.shape[1]):
  print(i)
  inds = np.where(X[:, i] >= 0)[0]
  x = X[inds][:, i]
  g_names = [gene_names[j] for j in inds]
  x_clusters = build_clusters(x, g_names)
  score = restricted_rand_index(x_clusters, label1)
  results[merge_samples_msp[i]] = score
  print("\n\n")
