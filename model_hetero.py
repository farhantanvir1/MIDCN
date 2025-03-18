import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_scatter import scatter_add
from torch_geometric.nn import GINConv, global_max_pool as gmp

# ==============================
# BioEncoder: Encodes Disease & Microbe Features, and Processes Drug Graph
# ==============================

class BioEncoder(nn.Module):
    def __init__(self, dim_mic, dim_dis, output, num_features_per_layer, predictor_hidden_dim=64):
        """
        BioEncoder is responsible for:
        - Encoding disease features (dis_feature)
        - Encoding microbe features (mic_feature)
        - Processing drug molecule graph using GINConv
        
        """
        super(BioEncoder, self).__init__()

        # Disease and Microbe feature encoding layers
        self.dis_layer1 = nn.Linear(dim_dis, output)
        self.batch_dis1 = nn.BatchNorm1d(output)
        self.mic_layer1 = nn.Linear(dim_mic, output)
        self.batch_mic1 = nn.BatchNorm1d(output)

        # Graph-based embedding using GINConv
        self.conv1 = GINConv(nn.Sequential(nn.Linear(78, output), nn.ReLU(), nn.Linear(output, output)))
        self.bn1 = torch.nn.BatchNorm1d(output)

        self.conv2 = GINConv(nn.Sequential(nn.Linear(output, output), nn.ReLU(), nn.Linear(output, output)))
        self.bn2 = torch.nn.BatchNorm1d(output)

        self.conv3 = GINConv(nn.Sequential(nn.Linear(output, output), nn.ReLU(), nn.Linear(output, output)))
        self.bn3 = torch.nn.BatchNorm1d(output)

        self.fc1_xd = nn.Linear(output, output)

        # Dropout and activation
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.15)

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(3 * num_features_per_layer[-1], predictor_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(predictor_hidden_dim, 1)
        )

        self.reset_para()

    def reset_para(self):
        """
        Initialize weights using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, dru_adj, mic_feature, dis_feature):
        """
        Forward pass for encoding drugs, microbes, and diseases.
        
        """

        # Process drug graph using GINConv layers
        x_d, edge_index, batch = dru_adj.x, dru_adj.edge_index, dru_adj.batch
        x_d = self.relu(self.conv1(x_d, edge_index))
        x_d = self.bn1(x_d)
        x_d = self.relu(self.conv2(x_d, edge_index))
        x_d = self.bn2(x_d)
        x_d = self.relu(self.conv3(x_d, edge_index))
        x_d = self.bn3(x_d)
        x_d = gmp(x_d, batch)  # Global Max Pooling
        x_d = self.relu(self.fc1_xd(x_d))
        x_d = self.drop_out(x_d)

        # Encode microbe and disease features
        x_dis = self.batch_dis1(F.relu(self.dis_layer1(dis_feature)))
        x_mic = self.batch_mic1(F.relu(self.mic_layer1(mic_feature)))

        return x_d, x_mic, x_dis

    def forward_predictor(self, out_nodes_features, drug_indices_batch, microbe_indices_batch, disease_indices_batch):
        """
        Predict interaction scores using the encoded features.
        
        """
        z_drug = out_nodes_features[drug_indices_batch]
        z_microbe = out_nodes_features[microbe_indices_batch]
        z_disease = out_nodes_features[disease_indices_batch]

        interaction_features = torch.cat((z_drug, z_microbe, z_disease), dim=-1)
        interaction_values = self.predictor(interaction_features)
        interaction_values = torch.sigmoid(interaction_values)  # Apply sigmoid activation

        return interaction_values


# ==============================
# Heterogeneous Triplet Network (HTN)
# ==============================

class HTN(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False, predictor_hidden_dim=64):
        """
        HTN (Heterogeneous Triplet Network) applies multiple GAT layers to learn feature representations.
        
        """
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        num_heads_per_layer = [1] + num_heads_per_layer  # First layer uses 1 head

        self.htn_layers = nn.ModuleList()
        for i in range(num_of_layers):
            layer = HTNLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,
                activation=nn.Sigmoid() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            self.htn_layers.append(layer)

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(3 * num_features_per_layer[-1], predictor_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(predictor_hidden_dim, 1)
        )

    def forward(self, in_nodes_features, edge_index):
        """
        Forward pass through multiple HTN layers.

        """
        x = in_nodes_features
        for layer in self.htn_layers:
            x = layer(x, edge_index)
        return x

    def forward_predictor(self, out_nodes_features, drug_indices_batch, microbe_indices_batch, disease_indices_batch):
        """
        Predict interaction values based on learned embeddings.

        """
        z_drug = out_nodes_features[drug_indices_batch]
        z_microbe = out_nodes_features[microbe_indices_batch]
        z_disease = out_nodes_features[disease_indices_batch]

        interaction_features = torch.cat((z_drug, z_microbe, z_disease), dim=-1)
        interaction_values = self.predictor(interaction_features)
        interaction_values = torch.sigmoid(interaction_values)

        return interaction_values



class HTNLayer(nn.Module):
    """
    Heterogeneous Tensor Network (HTN) Layer for Drug-Microbe-Disease interaction modeling.
    
    This layer computes **attention-based message passing** between Drug, Microbe, and Disease nodes.
    It applies Generalized Attention and Compressed Tensor Networks to model interactions.
    
    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, attention_mlp_hidden=64, concat=True,
                 activation=nn.ELU(), dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        # Project input features into multiple attention heads
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        
        # Attention mechanism (Multi-layer Perceptron)
        self.attention_mlp = nn.Sequential(
            nn.Linear(3 * num_out_features, attention_mlp_hidden),
            nn.ReLU(),
            nn.Linear(attention_mlp_hidden, 1)
        )

        # Edge neural network for learning interactions
        self.edge_nn = nn.Sequential(
            nn.Linear(num_out_features * 2, num_out_features),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(num_out_features, num_out_features),
        )

        # Attention weight matrix (Generalized Tensor Factorization)
        self.theta = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        nn.init.xavier_uniform_(self.theta)

        # Optional Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features if concat else num_out_features))
        else:
            self.register_parameter('bias', None)

        # Skip connection
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_attention_weights = log_attention_weights

    def compute_attention_scores(self, node_features_proj, edge_index):
        """
        Compute attention scores for each (Drug, Microbe, Disease) triplet.
        """
        drug_features = node_features_proj[edge_index[0]]
        microbe_features = node_features_proj[edge_index[1]]
        disease_features = node_features_proj[edge_index[2]]

        # Concatenate features and pass through attention MLP
        interaction_features = torch.cat((drug_features, microbe_features, disease_features), dim=-1)
        attention_scores = self.attention_mlp(interaction_features).squeeze(-1)
        return self.leakyReLU(attention_scores)

    def aggregate_neighbors(self, node_features_proj, edge_index, attention_scores):
        """
        Aggregate neighbor information using attention scores.
        """
        attention_scores_softmax = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        concatenated_embeddings = []

        for i in range(3):
            if i == 0:
                neighbor_embedding = torch.cat((node_features_proj[edge_index[1]], node_features_proj[edge_index[2]]), dim=-1)
            elif i == 1:
                neighbor_embedding = torch.cat((node_features_proj[edge_index[0]], node_features_proj[edge_index[2]]), dim=-1)
            else:
                neighbor_embedding = torch.cat((node_features_proj[edge_index[0]], node_features_proj[edge_index[1]]), dim=-1)

            neighbor_product = self.edge_nn(neighbor_embedding)
            weighted_sum = scatter_add(attention_scores_softmax * neighbor_product, edge_index[i], dim=0, dim_size=node_features_proj.size(0))

            # Compute final representation using Tensor Factorization
            node_features = self.theta * node_features_proj + weighted_sum
            concatenated_embeddings.append(node_features)

        return torch.cat(concatenated_embeddings, dim=0)

    def forward(self, node_features, edge_index):
        """
        Forward pass through the HTN layer.
        """
        node_features_proj = self.linear_proj(node_features).view(-1, self.num_of_heads, self.num_out_features)
        attention_scores = self.compute_attention_scores(node_features_proj, edge_index)
        out_features = self.aggregate_neighbors(node_features_proj, edge_index, attention_scores)

        if self.concat:
            out_features = out_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_features = out_features.mean(dim=1)

        if self.bias is not None:
            out_features += self.bias

        if self.activation is not None:
            out_features = self.activation(out_features)

        return out_features
