import numpy as np
import pickle
from similarity import load_samples, adjusted_mutual_information, adjusted_rand_index
from BCE import generate_n_matrix, preprocess_files
import numba
import time
import threading
from multiprocessing import Process, Pool
from functools import partial
from samples import valid_samples
import multiprocessing as mp


class MM(object):
  def __init__(self, X, k, Z_init=None, prior=None, thetas=None):
    self.X = X
    self.k = k
    self.M = X.shape[1] # n_exp
    self.N = X.shape[0] # n_genes

    if thetas is None:
      n_matrices = [generate_n_matrix(Z_init, X[:, i], k)[0] + 1 for i in range(self.M)]
      self.thetas = [n_matrix/np.expand_dims(n_matrix.sum(1), 1) for n_matrix in n_matrices]
    if prior is None:
      n_z = np.bincount(Z_init)
      self.prior = n_z/n_z.sum()
      
  def infer_zi(self, i):
    z_i = np.copy(self.prior)
    for j in range(self.M):
      if self.X[i, j] > 0:
        z_i *= self.thetas[j][:, self.X[i, j]]
    z_i = z_i/z_i.sum()
    return z_i

def EM(n_iter, mm, n_threads=None):
  if n_threads is None:
    n_threads = mp.cpu_count()
  pl = Pool(n_threads)
  inds = np.arange(mm.N)
  cuts = np.linspace(0, len(inds)+1, n_threads+1)
  i_lists = [inds[int(cuts[i]):int(cuts[i+1])] for i in range(n_threads)]
  for epoch in range(n_iter):
    threadRoutine = partial(WorkerEM, MM=mm)
    res = pl.map(threadRoutine, i_lists)
    
    new_prior = sum([r[1] for r in res])
    assert np.allclose(new_prior.sum(), mm.N)    
    all_new_thetas = [r[0] for r in res]
    new_thetas = [sum([p[i] for p in all_new_thetas]) for i in range(mm.M)]
    for i, theta in enumerate(new_thetas):
      assert np.allclose(theta.sum(), np.where(mm.X[:, i] >= 0)[0].shape[0])

    mm.prior = new_prior/new_prior.sum()
    mm.thetas = [mat/np.expand_dims(mat.sum(1), 1) for mat in new_thetas]

def WorkerEM(i_list, MM=None):
  new_thetas = [np.zeros_like(item) for item in MM.thetas]
  new_prior = np.zeros_like(MM.prior)
  for i in i_list:
    z_i = MM.infer_zi(i)
    new_prior += z_i
    for j in range(MM.M):
      if MM.X[i, j] >= 0:
        new_thetas[j][:, MM.X[i, j]] += z_i
  return new_thetas, new_prior

  
def InferZ(mm, n_threads=None):
  if n_threads is None:
    n_threads = mp.cpu_count()
  inds = np.arange(mm.N)
  cuts = np.linspace(0, len(inds)+1, n_threads+1)
  i_lists = [inds[int(cuts[i]):int(cuts[i+1])] for i in range(n_threads)]
  threadRoutine = partial(WorkerInferZ, MM=mm)
  with Pool(n_threads) as p:
    res = p.map(threadRoutine, i_lists)
  Z = []
  logProb = 0.
  for r in res:
    Z.extend(r[0])
    logProb += r[1]
  print("LogL: %f" % logProb)
  return Z

def WorkerInferZ(i_list, MM=None):
  Zs = [None] * len(i_list)
  logProb = 0.
  for ind, i in enumerate(i_list):
    z_i = MM.infer_zi(i)
    Zs[ind] = np.argmax(z_i)
    logProb += np.log(np.max(z_i))
  return Zs, logProb

def initialize_Z(X, seed=None):
  if not seed is None:
    np.random.seed(seed)

  order = np.arange(X.shape[1])
  np.random.shuffle(order)
  Z = np.copy(X[:, order[0]])
  for i in order[1:]:
    update_positions = set(list(np.where(Z < 0)[0])) & \
        set(list(np.where(X[:, i] >= 0)[0]))
    for j in update_positions:
      if Z[j] < 0:
        assign = X[j, i]
        assert assign >= 0
        points_in_this_cluster = np.where(X[:, i] == assign)[0]
        existing_cluster_assignments = [Z[p] for p in points_in_this_cluster if Z[p] >= 0]
        missing_assignments = [p for p in points_in_this_cluster if Z[p] < 0]
        if len(existing_cluster_assignments) < 5:
          existing_cluster_assignments.append(np.max(Z) + 1) # Assignment to a new cluster

        assert not -1 in existing_cluster_assignments
        assert j in missing_assignments

        if len(missing_assignments) > 20: # When large number of unlabelled points are present, assign them to the same cluster
          j_assign = np.random.choice(existing_cluster_assignments)
          for k in missing_assignments:
            Z[k] = j_assign
        else:
          for k in missing_assignments:
            j_assign = np.random.choice(existing_cluster_assignments)
            Z[k] = j_assign
  assert not -1 in Z
  assert len(np.unique(Z)) == np.max(Z) + 1
  
  k = np.max(Z) + 1
  return Z, k

def build_clusters(Z):
  Z_clusters = {}
  for i, z in enumerate(Z):
    cluster_id = z
    if cluster_id not in Z_clusters:
      Z_clusters[cluster_id] = []
    Z_clusters[cluster_id].append(gene_names[i])
  return Z_clusters

if __name__ == '__main__':

  ground_truth_clusters = pickle.load(open('../utils/ref_species_clusters.pkl', 'rb'))
  file_list = ["../summary/mspminer_%s.txt" % s for s in valid_samples]
  X, gene_names = preprocess_files(file_list)

  n_threads = 1

  # Seed 147 with initialization on ERP005989 has highest score
  for seed in range(100, 150):
    Z, k = initialize_Z(X, seed=seed)
    with open("Z_full_init_%d.pkl" % seed, "wb") as f:
      pickle.dump([Z, k], f)

    Z_clusters = build_clusters(Z)
    for f in file_list:
      print(f + "\t" + str(adjusted_rand_index(f, Z_clusters)))
    print("Ground Truth\t" + str(adjusted_rand_index(ground_truth_clusters, Z_clusters)), flush=True)
