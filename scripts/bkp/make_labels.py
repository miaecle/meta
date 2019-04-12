import numpy as np
import pickle

mapping = {}
with open('./IGC.annotation11.4M.summary', 'r') as f:
  for i, line in enumerate(f):
    if i==0: continue
    id = line.split()[0]
    name = line.split()[1]
    mapping[name] = id

f1 = 'Species_Gene_Array_0.9.list'
f2 = 'Species_Gene_Array_0.9.pkl'

d = {}
with open(f1, 'r') as f:
  for line in f:
    s_name = line.split('\t')[0]
    gs = line.split('\t')[1][:-1]
    gs = gs.split(',')
    d[s_name] = [mapping[g] for g in gs]

with open(f2, 'wb') as f:
  pickle.dump(d, f)
