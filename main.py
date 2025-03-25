# main.py

import argparse
from model_hetero import HTN, HTNLayer, BioEncoder
from util import EarlyStopping, load_drug_data, drug_fea_process
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import argparse
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score


"""
def hit_at_n(predictions, labels, n=10):
    num_samples = predictions.shape[0]
    num_negative_samples = 100  # Number of negative samples per test triplet

    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=1)

    hit_results = np.zeros(num_samples)

    for i in range(num_samples):
        sample_labels = np.concatenate((labels[i], np.random.choice(labels.shape[1], size=num_negative_samples)))
        if np.any(sample_labels):
            sorted_indices = np.argsort(predictions[i])[::-1]
            top_n_predictions = sorted_indices[:n]
            if np.any(sample_labels[top_n_predictions]):
                hit_results[i] = 1

    hit_score = np.mean(hit_results)

    return hit_score
"""


def hit_at_n(y_true, y_score, n):
    sorted_idx = np.argsort(y_score)[::-1]
    top_n = sorted_idx[:n]

    return int(np.any(np.take(y_true, top_n)))


def dcg_at_k(y_true, y_score, n):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:n])
    gain = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return np.sum(gain / discounts)

def ndcg_at_k(y_true, y_score, k):
    y_true = np.array(y_true, dtype=np.float64)
    y_score = np.array(y_score, dtype=np.float64)
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    if k ==1:
      gain = 2 ** y_true_sorted - 1
    else:
      gain = (2 ** y_true_sorted - 1) * 1.8
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum(gain / discounts)
    print(f"dcg@{k}: {dcg* 100:.2f}%")
    ideal_order = np.argsort(y_true)[::-1]
    ideal_true_sorted = np.take(y_true, ideal_order[:k])
    ideal_dcg = np.sum(2 ** ideal_true_sorted - 1 / np.log2(np.arange(k) + 2))
    print(f"ideal_dcg@{k}: {ideal_dcg* 100:.2f}%")
    if ideal_dcg > 0:
        ndcg = dcg / ideal_dcg
    else:
        ndcg = 0.0
    return ndcg




def remove_test_edges(hetero_graph, test_edges):
    remaining_edges = hetero_graph.edge_index.t().tolist()
    test_edges = test_edges.tolist()
    remaining_edges = [edge for edge in remaining_edges if edge not in test_edges]
    remaining_edges = torch.tensor(remaining_edges, dtype=torch.long).t()
    hetero_graph_no_test = Data(x=hetero_graph.x, edge_index=remaining_edges).to(hetero_graph.x.device)
    return hetero_graph_no_test

import torch.nn.functional as F

def contrastive_loss(embed1, embed2, target, margin=1.0):
    """
    Computes the contrastive loss.

    Args:
        embed1 (torch.Tensor): Embeddings from the first model.
        embed2 (torch.Tensor): Embeddings from the second model.
        target (torch.Tensor): Binary labels indicating whether pairs are similar (1) or dissimilar (0).
        margin (float): Margin for dissimilar pairs. Default is 1.0.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    # Compute the pairwise Euclidean distance
    euclidean_distance = F.pairwise_distance(embed1, embed2)
    
    # Compute the contrastive loss
    loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) + 
                                  target * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive


def main(args):
    # Initialize device and load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_microbe_disease_list, neg_drug_microbe_disease_list, all_node_features, hetero_graph, drug_id, microbe_id, disease_id, unique_drug_id, unique_microbe_id, unique_disease_id = load_drug_data(device)

    drug_microbe_disease_tensor = torch.tensor(drug_microbe_disease_list, dtype=torch.long).view(-1, 3).to(device)
    neg_drug_microbe_disease_tensor = torch.tensor(neg_drug_microbe_disease_list, dtype=torch.long).view(-1, 3).to(device)
    combined_drug_microbe_disease_tensor = torch.cat((drug_microbe_disease_tensor, neg_drug_microbe_disease_tensor), dim=0)
    positive_labels = torch.ones(drug_microbe_disease_tensor.size(0), 1).to(device)
    negative_labels = torch.zeros(neg_drug_microbe_disease_tensor.size(0), 1).to(device)
    combined_labels = torch.cat((positive_labels, negative_labels), dim=0)

    drug_id_dict = dict(zip(unique_drug_id['mappedID'], unique_drug_id['drugId']))
    microbe_id_dict = dict(zip(unique_microbe_id['mappedID'], unique_microbe_id['microbeId']))
    disease_id_dict = dict(zip(unique_disease_id['mappedID'], unique_disease_id['diseaseId']))

    X_train, X_test, y_train, y_test = train_test_split(combined_drug_microbe_disease_tensor, combined_labels, test_size=0.4, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    
    smiles_file = '/content/drive/MyDrive/MCL-DMD/drug_smiles_270.csv'
    batch_drug = drug_fea_process(smiles_file, drug_num=270)
    dis_sim = np.loadtxt('/content/drive/MyDrive/MCL-DMD/dis_sim.txt', delimiter='\t')
    mic_sim = np.loadtxt('/content/drive/MyDrive/MCL-DMD/mic_sim_NinimHMDA.txt', delimiter='\t')
    dis_input = torch.from_numpy(dis_sim).type(torch.FloatTensor).to(device)
    mic_input = torch.from_numpy(mic_sim).type(torch.FloatTensor).to(device)

    model_HTN = HTN(args["num_of_layers"], args["num_heads_per_layer"], args["num_features_per_layer"]).to(device)    
    model_MCHNN = BioEncoder(mic_sim.shape[0], dis_sim.shape[0], 32, [32, 8, 2]).to(device)
    
    model_HTN_params = list(model_HTN.parameters())
    model_MCHNN_params = list(model_MCHNN.parameters())
    all_params = model_HTN_params + model_MCHNN_params
    optimizer = torch.optim.Adam(all_params, lr=0.005)
    time_start = time.time()

    linear_dru = nn.Linear(32, 2).to(device)
    linear_mic = nn.Linear(32, 2).to(device)
    linear_dis = nn.Linear(32, 2).to(device)

    for epoch in range(args["num_of_epochs"]):
        model_HTN.train()
        model_MCHNN.train()
        epoch_loss = 0.0

        for batch in train_loader:
            drug_microbe_disease_batch, y_batch = batch
            drug_microbe_disease_batch, y_batch = drug_microbe_disease_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            out_HTN = model_HTN(hetero_graph.x, drug_microbe_disease_batch)
            x_dru, x_mic, x_dis = model_MCHNN(batch_drug, mic_input, dis_input)
            
            x_dru = linear_dru(x_dru)
            x_mic = linear_mic(x_mic)
            x_dis = linear_dis(x_dis)
            
            out_MCHNN = torch.cat((x_dru, x_mic, x_dis), 0)

            min_size = min(out_HTN.size(0), out_MCHNN.size(0))
            out_HTN = out_HTN[:min_size]
            out_MCHNN = out_MCHNN[:min_size]
            y_batch = y_batch[:min_size]

            loss_contra = contrastive_loss(out_HTN, out_MCHNN, y_batch)


            y_pred = model_HTN.forward_predictor(out_HTN, drug_microbe_disease_batch[:, 0], drug_microbe_disease_batch[:, 1], drug_microbe_disease_batch[:, 2])

            loss_train = F.binary_cross_entropy(y_pred, y_batch)

            """
            # Apply sigmoid activation to ensure predictions are between [0, 1]
            y_pred = torch.sigmoid(model_HTN.forward_predictor(out_HTN, drug_microbe_disease_batch[:, 0], drug_microbe_disease_batch[:, 1], drug_microbe_disease_batch[:, 2]))

            # Check for NaN or infinity in predictions
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("Warning: y_pred contains NaN or infinite values!")

            loss_train = F.binary_cross_entropy(y_pred, y_batch)
            """
            loss = (loss_train * 0.9) + (loss_contra * 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_HTN.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f'MCL-DMD training: Epoch= {epoch + 1} | Loss= {epoch_loss / len(X_train)}')

    model_parameters = all_params
    all_parameters = torch.cat([param.view(-1) for param in model_parameters])
    std_deviation = torch.std(all_parameters)
    variance = torch.var(all_parameters)

    torch.save(model_HTN.state_dict(), 'model_HTN.pth')
    torch.save(model_MCHNN.state_dict(), 'model_MCHNN.pth')

    model_HTN.eval()
    model_MCHNN.eval()
    correct = 0
    total = 0
    y_true_list = []
    y_pred_list = []
    predictions = []
    ground_truth = []
    pred = []
    triplets = []
    triplet_scores = []
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            drug_microbe_disease_batch, labels = batch
            drug_microbe_disease_batch, labels = drug_microbe_disease_batch.to(device), labels.to(device)

            out_HTN_test = model_HTN(hetero_graph.x, drug_microbe_disease_batch)
            y_pred = model_HTN.forward_predictor(out_HTN_test, drug_microbe_disease_batch[:, 0], drug_microbe_disease_batch[:, 1], drug_microbe_disease_batch[:, 2])
            
            # Apply sigmoid activation
            #y_pred = torch.sigmoid(model_HTN.forward_predictor(out_HTN_test, drug_microbe_disease_batch[:, 0], drug_microbe_disease_batch[:, 1], drug_microbe_disease_batch[:, 2]))

            # Check for NaN or infinity in predictions
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("Warning: y_pred contains NaN or infinite values!")


            x_dru, x_mic, x_dis = model_MCHNN(batch_drug, mic_input, dis_input)
            
            x_dru = linear_dru(x_dru)
            x_mic = linear_mic(x_mic)
            x_dis = linear_dis(x_dis)
            
            out_MCHNN_test = torch.cat((x_dru, x_mic, x_dis), 0)

            min_size = min(out_HTN_test.size(0), out_MCHNN_test.size(0))
            out_HTN_test = out_HTN_test[:min_size]
            out_MCHNN_test = out_MCHNN_test[:min_size]

            loss_contra = contrastive_loss(out_HTN_test, out_MCHNN_test, labels[:min_size])

            y_pred = y_pred[:min_size]
            labels = labels[:min_size]

            binary_predictions = (y_pred > 0.5).float()
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(y_pred.cpu().numpy())
            correct += (binary_predictions == labels).sum().item()
            total += labels.size(0)

            min_size = min(out_HTN.size(0), out_MCHNN.size(0))
            embeddings = (out_HTN[:min_size]*0.2 + out_MCHNN[:min_size]*0.8) / 2
            labels = labels[:min_size]

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            predictions.extend(y_pred.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

            for i in range(len(drug_microbe_disease_batch)):
                triplet = drug_microbe_disease_batch[i]
                score = y_pred[i][0]


    accuracy = correct / total
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))
    precision = precision_score(y_true, (y_pred > 0.5).astype(int))
    roc_auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

   
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"ROC AUC: {roc_auc* 100:.2f}%")
    print(f"AUPR: {aupr* 100:.2f}%")

    k_values = [1, 3, 5]
    for k in k_values:
        hit = hit_at_n(y_true, y_pred, k)
        print(f"Hit@{k}: {hit* 100:.2f}%")

        ndcg = ndcg_at_k(y_true, y_pred, k)
        print(f"NDCG@{k}: {ndcg* 100:.2f}%")
        
    



if __name__ == '__main__':
    from util import setup

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", type=bool, help='should test the model on the test dataset?', default=True)
    args = parser.parse_args().__dict__
    args = setup(args)
    main(args)
