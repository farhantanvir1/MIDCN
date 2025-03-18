#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================
# Utility functions for data processing, feature extraction, and negative sampling
# ==============================

import datetime
import errno
import numpy as np
import os
import pickle
import random
import torch
import scipy.sparse as sp
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from torch_geometric.data import download_url, extract_zip
from csv import DictReader
from csv import reader
import numpy as np
import pandas as pd
import xlrd
import csv
import os
import csv
import xlrd
import _thread
import time
import torch as th
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
import argparse
import random
import csv
from process_smiles import GraphDataset_v, smile_to_graph, collate

# ==============================
# Drug Feature Processing
# ==============================

def drug_fea_process(smiles_file, drug_num):
    """
    Converts SMILES representations of drugs into graph-based features.
    
    """
    reader = csv.reader(open(smiles_file))
    smile_graph = []
    
    for item in reader:
        smile = item[1]
        g = smile_to_graph(smile)  # Convert SMILES to graph representation
        smile_graph.append(g)

    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False, collate_fn=collate)
    
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_data

# ==============================
# Feature Normalization
# ==============================

def normalize_features(feat):
    """
    Normalizes feature matrix by row-wise sum.
    
    """
    degree = np.asarray(feat.sum(1)).flatten()
    degree[degree == 0] = 1e10  # Avoid division by zero
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

# The configuration below is from the paper.
default_configure = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [32, 8, 2],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout":  0.6,  # result is sensitive to dropout
        "lr":1e-5,
        "weight_decay": 0.001,
        "num_of_epochs": 100,
        "patience": 100,
        'batch_size': 32
    }



def setup(args):
    args.update(default_configure)
    args['hetero']=True
    args['dataset'] = 'DrugBank' if args['hetero'] else 'DrugBank'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

# ==============================
# Negative Sampling
# ==============================

def negsamp_incr(unique_drug_id, unique_microbe_id, unique_disease_id, drug_microbe_disease_list, n_drug_items=270, n_microbe_items=58, n_disease_items=167, n_samp=2763):
    """
    Generates negative samples by randomly selecting triplets that do not exist in the dataset.
    
    """
    neg_inds = []
    neg_drug_id, neg_microbe_id, neg_disease_id = [], [], []
    
    while len(neg_inds) < n_samp:
        drug_samp = np.random.randint(0, n_drug_items)
        microbe_samp = np.random.randint(0, n_microbe_items)
        disease_samp = np.random.randint(0, n_disease_items)
        
        if [drug_samp, microbe_samp, disease_samp] not in drug_microbe_disease_list and \
           [drug_samp, microbe_samp, disease_samp] not in neg_inds:
            neg_drug_id.append(drug_samp)
            neg_microbe_id.append(microbe_samp)
            neg_disease_id.append(disease_samp)
            neg_inds.append([drug_samp, microbe_samp, disease_samp])

    return neg_drug_id, neg_microbe_id, neg_disease_id, neg_inds

# ==============================
# Data Loading Function
# ==============================

def load_drug_data(device):
    """
    Loads drug-microbe-disease triplet data, performs ID mapping, and generates graph features.
    
    """
    
    triplet_path = '/content/drive/MyDrive/MIDCN/adj_del_4mic_myid.txt'
    triplet_df = pd.read_csv(triplet_path, delimiter='\t', header=None)
    triplet_df.columns = ['Drugs', 'Microbes', 'Diseases', 'Connection']

    # Map unique IDs to a continuous range
    unique_drug_id = pd.DataFrame({'drugId': triplet_df["Drugs"].unique(), 'mappedID': pd.RangeIndex(len(triplet_df["Drugs"].unique()))})
    unique_microbe_id = pd.DataFrame({'microbeId': triplet_df["Microbes"].unique(), 'mappedID': pd.RangeIndex(len(triplet_df["Microbes"].unique()))})
    unique_disease_id = pd.DataFrame({'diseaseId': triplet_df["Diseases"].unique(), 'mappedID': pd.RangeIndex(len(triplet_df["Diseases"].unique()))})

    # Map actual triplet data using new IDs
    drug_id = pd.merge(triplet_df, unique_drug_id, left_on='Drugs', right_on='drugId', how='left')['mappedID']
    microbe_id = pd.merge(triplet_df, unique_microbe_id, left_on='Microbes', right_on='microbeId', how='left')['mappedID']
    disease_id = pd.merge(triplet_df, unique_disease_id, left_on='Diseases', right_on='diseaseId', how='left')['mappedID']

    # Convert to tensors
    drug_id, microbe_id, disease_id = torch.from_numpy(drug_id.values).to(device), torch.from_numpy(microbe_id.values).to(device), torch.from_numpy(disease_id.values).to(device)

    # Combine into edge_index tensor
    edge_index = torch.column_stack([drug_id, microbe_id, disease_id]).to(device)

    # Generate random features for entities
    drug_features = torch.tensor(np.random.randint(2, size=(270, 32)), dtype=torch.float)
    microbe_features = torch.tensor(np.random.randint(2, size=(58, 32)), dtype=torch.float)
    disease_features = torch.tensor(np.random.randint(2, size=(167, 32)), dtype=torch.float)

    # Normalize and concatenate features
    all_node_features = torch.cat([drug_features, microbe_features, disease_features], dim=0)
    all_node_features = torch.from_numpy(normalize_features(all_node_features)).to(device)

    # Create heterogeneous graph
    hetero_graph = Data(x=all_node_features, edge_index=edge_index).to(device)

    # Generate negative samples
    drug_microbe_disease_list = list(zip(drug_id.tolist(), microbe_id.tolist(), disease_id.tolist()))
    neg_drug_id, neg_microbe_id, neg_disease_id, neg_drug_microbe_disease_list = negsamp_incr(
        unique_drug_id, unique_microbe_id, unique_disease_id, drug_microbe_disease_list
    )

    return drug_microbe_disease_list, neg_drug_microbe_disease_list, all_node_features, hetero_graph, drug_id, microbe_id, disease_id, unique_drug_id, unique_microbe_id, unique_disease_id



# Early stopping implementation
class EarlyStopping:
    """
    Implements early stopping mechanism to prevent overfitting.
    
    """
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
