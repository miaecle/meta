#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:50:28 2019

@author: zqwu
"""

import numpy as np
import pandas as pd
import pickle

raw_annotation = pd.read_table('../IGC.annotation11.4M.summary')
annotated = np.where(np.array(raw_annotation['Phylum_ano']) != 'Unknown')[0]
annotation = raw_annotation.loc[annotated]
del raw_annotation

species, species_cts = np.unique(annotation['Species_ano'], return_counts=True)

species = list(species[np.where(species_cts > 100)[0]])
species_cts = list(species_cts[np.where(species_cts > 100)[0]])

#################SPECIAL CASES#############################
merge_from = ['Klebsiella pneumoniae', 
              'Klebsiella pneumoniae/Klebsiella variicola group',
              'Klebsiella variicola/pneumoniae']
new_speci = 'Klebsiella variicola/pneumoniae'
new_speci_ct = sum([species_cts[species.index(s)] for s in merge_from])

for s in merge_from:
  ind = species.index(s)
  del species[ind]
  del species_cts[ind]
species.append(new_speci)
species_cts.append(new_speci_ct)

removes = ['Parvimonas sp. oral taxon 110',
           'Parvimonas sp. oral taxon 393',
           'Unknown',
           '[Bacteroides] pectinophilus',
           '[Clostridium] bartlettii',
           '[Clostridium] difficile',
           '[Ruminococcus] gnavus',
           '[Ruminococcus] obeum',
           '[Ruminococcus] torques',
           'butyrate-producing bacterium',
           'butyrate-producing bacterium SSC/2',
           'unclassified Alistipes sp. HGB5',
           'unclassified Capnocytophaga sp. oral taxon 329',
           'unclassified Citrobacter sp. 30_2',
           'unclassified Clostridiales bacterium 1_7_47FAA',
           'unclassified Clostridium sp. 7_2_43FAA',
           'unclassified Clostridium sp. D5',
           'unclassified Clostridium sp. HGF2',
           'unclassified Clostridium sp. L2-50',
           'unclassified Coprobacillus sp. 29_1',
           'unclassified Desulfovibrio sp. 3_1_syn3',
           'unclassified Eggerthella sp. YY7918',
           'unclassified Enterobacteriaceae bacterium 9_2_54FAA',
           'unclassified Erysipelotrichaceae bacterium 3_1_53',
           'unclassified Erysipelotrichaceae bacterium 5_2_54FAA',
           'unclassified Fusobacterium',
           'unclassified Fusobacterium sp. D12',
           'unclassified Lachnospiraceae bacterium 1_4_56FAA',
           'unclassified Lachnospiraceae bacterium 2_1_46FAA',
           'unclassified Lachnospiraceae bacterium 3_1_57FAA_CT1',
           'unclassified Lachnospiraceae bacterium 4_1_37FAA',
           'unclassified Lachnospiraceae bacterium 9_1_43BFAA',
           'unclassified Ruminococcaceae bacterium D16',
           'unclassified Ruminococcus sp. 5_1_39BFAA',
           'unclassified Ruminococcus sp. SR1/5',
           'unclassified Streptococcus oralis',
           'unclassified Streptococcus sp. C150',
           'unclassified Streptococcus sp. oral taxon 071',
           'unclassified Veillonella sp. oral taxon 158',
           'unclassified butyrate-producing bacterium SS3/4',
           'unclassified unclassified Fusobacterium',
           'Klebsiella variicola/pneumoniae']
for s in removes:
  ind = species.index(s)
  del species[ind]
  del species_cts[ind]
##############################################################

ref_clusters = {}
for i, s in enumerate(species):
  genes = np.array(annotation['ID'])[np.where(annotation['Species_ano'] == s)[0]]
  assert len(genes) == species_cts[i]
  ref_clusters[s] = genes

with open("../ref_species_clusters.pkl", "wb") as f:
  pickle.dump(ref_clusters, f)

  

##############################################################
genus, genus_cts = np.unique(annotation['Genus_ano'], return_counts=True)
genus = list(genus[np.where(genus_cts > 100)[0]])
genus_cts = list(genus_cts[np.where(genus_cts > 100)[0]])
removes = ['Unknown']
for s in removes:
  ind = genus.index(s)
  del genus[ind]
  del genus_cts[ind]
  
ref_clusters = {}
for i, s in enumerate(genus):
  genes = np.array(annotation['ID'])[np.where(annotation['Genus_ano'] == s)[0]]
  assert len(genes) == genus_cts[i]
  ref_clusters[s] = genes

with open("../ref_genus_clusters.pkl", "wb") as f:
  pickle.dump(ref_clusters, f)