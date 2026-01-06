import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    """
    Standard Graph Convolution Layer.
    Performs: H = sigma(A * X * W)
    """
    def __init__(self, in_features, out_features, activation=True, dropout=0.0):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.ReLU() if activation else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (Batch, In_Features)
        # adj: (Batch, Batch) - Normalized Adjacency Matrix
        
        # Support = X * W
        support = self.weight(x)
        
        # Output = A * Support
        output = torch.mm(adj, support)
        
        if self.activation:
            output = self.activation(output)
        
        output = self.dropout(output)
        return output

class GCN(nn.Module):
    """
    2-Layer Graph Convolution Network as described in Eq (14).
    Structure: Input -> GCN Layer 1 -> ReLU -> Dropout -> GCN Layer 2 -> ReLU
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(GCN, self).__init__()
        # "Theta1" layer
        self.gc1 = GraphConvolutionLayer(in_dim, hidden_dim, activation=True, dropout=dropout)
        # "Theta2" layer
        self.gc2 = GraphConvolutionLayer(hidden_dim, out_dim, activation=True, dropout=0.0)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

class HybridLoss(nn.Module):
    """
    Hybrid Loss Function incorporating Classification Loss and 
    Supervised Graph Contrastive Loss (L_sgc).
    """
    def __init__(self, feature_dim, hidden_dim=64, out_dim=32, 
                 lambda_weight=0.1, alpha=0.5, k=10, temperature=0.1, beta=1.0):
        """
        Args:
            feature_dim: Dimension of LDCT output features.
            lambda_weight: Weight hyperparameter for L_sgc (default 0.01-1.0 in paper).
            alpha: Parameter for RBF kernel spread.
            k: Number of nearest neighbors for graph construction.
            temperature: Softmax temperature 'epsilon'.
            beta: Scaling factor/Number of classes (used in Eq 16 denominator).
        """
        super(HybridLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.k = k
        self.temperature = temperature
        self.beta = beta
        
        # GCN for Contrastive Learning branch
        self.gcn = GCN(feature_dim, hidden_dim, out_dim)
        
        # Classification Loss
        self.criterion_cls = nn.CrossEntropyLoss()

    def get_adjacency_matrix(self, x):
        """
        Computes the normalized adjacency matrix based on RBF and KNN.
        Eq (12): A_ij = exp(-alpha * ||xi - xj||^2) for KNN, else 0.
        Eq (13): Normalized Laplacian A_hat.
        """
        # Calculate pairwise Euclidean distances
        # x: (B, D)
        dist_sq = torch.cdist(x, x, p=2).pow(2) # ||xi - xj||^2
        
        # Find K-Nearest Neighbors (indices)
        # We select k+1 because the closest is the node itself (dist=0)
        _, indices = torch.topk(dist_sq, k=self.k + 1, largest=False)
        
        # Create Mask for KNN
        mask = torch.zeros_like(dist_sq)
        mask.scatter_(1, indices, 1)
        
        # RBF Kernel weighting (Eq 12)
        # Note: Paper uses "-exp", assuming standard Gaussian similarity "exp" here for numerical stability
        # in graph spectral domain, as negative adjacency is non-standard.
        A = torch.exp(-self.alpha * dist_sq) * mask
        
        # Add self-loops (A + I)
        I = torch.eye(A.size(0), device=A.device)
        A_tilde = A + I
        
        # Normalization (Eq 13): D^(-1/2) * A_tilde * D^(-1/2)
        D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
        D_inv_sqrt = torch.pow(D_tilde, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0 # Handle division by zero
        
        A_hat = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
        return A_hat

    def get_prototypes(self, z, y):
        """
        Calculates class prototypes r_l^p (Eq 15).
        Average of samples with the same label in the batch.
        """
        unique_labels = torch.unique(y)
        prototypes = {}
        for l in unique_labels:
            mask = (y == l)
            if mask.sum() > 0:
                # Average features of class l
                prototypes[l.item()] = z[mask].mean(dim=0)
        return prototypes, unique_labels

    def supervised_graph_contrastive_loss(self, z1, z2, y):
        """
        Computes L_sgc (Eq 16).
        Args:
            z1: GCN Output for View 1.
            z2: GCN Output for View 2.
            y: Labels.
        """
        # Normalize representations for cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Get prototypes for both views
        proto1, classes = self.get_prototypes(z1, y)
        proto2, _ = self.get_prototypes(z2, y)
        
        loss = 0.0
        valid_classes = 0
        
        for l in classes:
            l = l.item()
            if l not in proto1 or l not in proto2:
                continue
                
            r1_l = proto1[l]
            r2_l = proto2[l]
            
            # Positive Pair Similarity: h(r1_l, r2_l)
            sim_pos = torch.exp(torch.dot(r1_l, r2_l) / self.temperature)
            
            # Negative Pairs Summation
            sum_neg = 0.0
            for m in classes:
                m = m.item()
                if m == l:
                    continue
                    
                if m in proto1:
                    sim_neg_1 = torch.exp(torch.dot(r1_l, proto1[m]) / self.temperature)
                    sum_neg += sim_neg_1
                    
                if m in proto2:
                    sim_neg_2 = torch.exp(torch.dot(r1_l, proto2[m]) / self.temperature)
                    sum_neg += sim_neg_2
            
            # Eq (16): -log( pos / (pos + beta * sum_neg) )
            # Note: Paper formula has beta * sum, or just sum over p. 
            # Assuming beta is the weighting factor '1/beta' or similar if intended, 
            # otherwise direct summation as per formula structure:
            denominator = sim_pos + (sum_neg) # + beta * sum_neg if beta is explicit multiplier
            
            loss += -torch.log(sim_pos / (denominator + 1e-8))
            valid_classes += 1
            
        if valid_classes > 0:
            loss = loss / valid_classes
        return loss

    def forward(self, features_view1, features_view2, logits_view1, labels):
        """
        Args:
            features_view1: LDCT features for original view (B, Dim).
            features_view2: LDCT features for augmented (rotated) view (B, Dim).
            logits_view1: MLP Classification logits for View 1 (B, Num_Classes).
            labels: Ground Truth labels (B).
        
        Returns:
            total_loss: L_cls + lambda * L_sgc
            l_cls: Classification Loss
            l_sgc: Graph Contrastive Loss
        """
        # 1. Classification Loss (Eq 17)
        # "Minimize the cross-entropy loss between predicted... and ground-truth"
        l_cls = self.criterion_cls(logits_view1, labels)
        
        # 2. Graph Construction & Convolution
        # "Input features extracted by LDCT into GCN"
        adj1 = self.get_adjacency_matrix(features_view1)
        adj2 = self.get_adjacency_matrix(features_view2)
        
        # Get latent representations from GCN (Eq 14)
        z_gcn1 = self.gcn(features_view1, adj1)
        z_gcn2 = self.gcn(features_view2, adj2)
        
        # 3. Supervised Graph Contrastive Loss (Eq 16)
        l_sgc = self.supervised_graph_contrastive_loss(z_gcn1, z_gcn2, labels)
        
        # 4. Total Loss (Eq 18)
        # "L = L_cls + lambda * L_sgc"
        total_loss = l_cls + self.lambda_weight * l_sgc
        
        return total_loss, l_cls, l_sgc