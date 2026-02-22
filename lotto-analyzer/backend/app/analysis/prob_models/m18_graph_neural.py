"""
M18 - Graph Neural Network Model

Models lottery numbers as nodes in a graph where edges represent
co-occurrence relationships. Uses message passing to learn number embeddings.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class M18Config:
    """Configuration for M18 model"""
    embedding_dim: int = 32
    hidden_dim: int = 64
    n_layers: int = 2
    epochs: int = 100
    learning_rate: float = 0.01
    edge_threshold: float = 0.0  # Min co-occurrence to create edge
    dropout: float = 0.2


class SimpleGNN(nn.Module):
    """Simple Graph Neural Network using message passing."""
    
    def __init__(self, n_nodes: int, embedding_dim: int, hidden_dim: int, 
                 n_layers: int, dropout: float):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        
        # Node embeddings
        self.node_embeddings = nn.Embedding(n_nodes, embedding_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList()
        in_dim = embedding_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embedding_dim
            self.layers.append(nn.Linear(in_dim * 2, out_dim))
            in_dim = out_dim
        
        # Output layer
        self.output = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with message passing.
        
        Args:
            adj_matrix: Adjacency matrix (n_nodes x n_nodes)
            
        Returns:
            Probability for each node
        """
        # Get node embeddings
        node_ids = torch.arange(self.n_nodes, device=adj_matrix.device)
        h = self.node_embeddings(node_ids)  # (n_nodes, embedding_dim)
        
        # Normalize adjacency matrix
        degree = adj_matrix.sum(dim=1, keepdim=True) + 1e-6
        adj_norm = adj_matrix / degree
        
        # Message passing
        for layer in self.layers:
            # Aggregate neighbor messages
            neighbor_msg = torch.mm(adj_norm, h)  # (n_nodes, dim)
            
            # Concatenate self and neighbor
            combined = torch.cat([h, neighbor_msg], dim=1)
            
            # Transform
            h = F.relu(layer(combined))
            h = self.dropout(h)
        
        # Output probabilities
        logits = self.output(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return probs


class M18GraphNeural:
    """
    Graph Neural Network for lottery number prediction.
    
    Models numbers as nodes in a graph where:
    - Nodes: Individual lottery numbers
    - Edges: Co-occurrence relationships between numbers
    - Edge weights: Frequency of co-occurrence
    
    The GNN learns embeddings that capture structural relationships
    between numbers and predicts which numbers are likely to appear.
    """
    
    model_id = "M18"
    model_name = "Graph Neural Network"
    model_type = "Deep Learning"
    
    def __init__(self, config: Optional[M18Config] = None):
        self.config = config or M18Config()
        self.model = None
        self.adj_matrix = None
        self.n_max = None
        self.k = None
        self._fitted = False
        self.device = None
    
    def _build_adjacency_matrix(self, draws: List[List[int]]) -> np.ndarray:
        """
        Build adjacency matrix from co-occurrence counts.
        
        Args:
            draws: List of historical draws
            
        Returns:
            Adjacency matrix (n_max x n_max)
        """
        adj = np.zeros((self.n_max, self.n_max))
        
        for draw in draws:
            for i, num1 in enumerate(draw):
                for num2 in draw[i+1:]:
                    if 1 <= num1 <= self.n_max and 1 <= num2 <= self.n_max:
                        adj[num1-1, num2-1] += 1
                        adj[num2-1, num1-1] += 1
        
        # Normalize
        if adj.max() > 0:
            adj = adj / adj.max()
        
        # Apply threshold
        adj[adj < self.config.edge_threshold] = 0
        
        return adj
    
    def _prepare_labels(self, draws: List[List[int]]) -> np.ndarray:
        """Prepare target labels (frequency-based)."""
        freq = np.zeros(self.n_max)
        for draw in draws:
            for num in draw:
                if 1 <= num <= self.n_max:
                    freq[num - 1] += 1
        
        # Normalize to [0, 1]
        if freq.max() > 0:
            freq = freq / freq.max()
        
        return freq
    
    def fit(self, draws: List[List[int]], n_max: int, k: int) -> None:
        """
        Train the GNN model.
        
        Args:
            draws: List of historical draws
            n_max: Maximum number in the pool
            k: Numbers per draw
        """
        self.n_max = n_max
        self.k = k
        self._last_draws = draws
        
        if not HAS_TORCH:
            print("Warning: PyTorch not available. Using fallback.")
            self._fitted = False
            return
        
        if len(draws) < 50:
            print("Warning: Not enough draws for GNN training.")
            self._fitted = False
            return
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build adjacency matrix
        adj_np = self._build_adjacency_matrix(draws)
        self.adj_matrix = torch.tensor(adj_np, dtype=torch.float32, device=self.device)
        
        # Prepare labels
        labels_np = self._prepare_labels(draws)
        labels = torch.tensor(labels_np, dtype=torch.float32, device=self.device)
        
        # Build model
        self.model = SimpleGNN(
            n_nodes=n_max,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            
            probs = self.model(self.adj_matrix)
            loss = criterion(probs, labels)
            
            loss.backward()
            optimizer.step()
        
        self._fitted = True
    
    def predict_proba(self, draws: List[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Predict probabilities using the trained GNN.
        """
        if draws is None:
            draws = getattr(self, '_last_draws', [])
        
        if not self._fitted or self.model is None:
            return self._fallback_predict(draws)
        
        # Update adjacency matrix with new draws if provided
        if draws != self._last_draws:
            adj_np = self._build_adjacency_matrix(draws)
            self.adj_matrix = torch.tensor(adj_np, dtype=torch.float32, device=self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            probs = self.model(self.adj_matrix).cpu().numpy()
        
        # Normalize
        probs = probs / probs.sum()
        
        return {"main": probs}
    
    def _fallback_predict(self, draws: List[List[int]]) -> Dict[str, np.ndarray]:
        """Fallback prediction using frequency."""
        if not draws or self.n_max is None:
            n_max = self.n_max or 45
            return {"main": np.ones(n_max) / n_max}
        
        freq = np.zeros(self.n_max)
        for draw in draws:
            for num in draw:
                if 1 <= num <= self.n_max:
                    freq[num - 1] += 1
        
        freq = freq + 1
        probs = freq / freq.sum()
        
        return {"main": probs}
    
    def generate_combinations(self, n: int = 1,
                               draws: List[List[int]] = None) -> List[List[int]]:
        """Generate n combinations based on predicted probabilities."""
        probs = self.predict_proba(draws)["main"]
        
        combinations = []
        for _ in range(n):
            selected = np.random.choice(
                range(1, self.n_max + 1),
                size=self.k,
                replace=False,
                p=probs
            )
            combinations.append(sorted(selected.tolist()))
        
        return combinations
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "n_layers": self.config.n_layers,
            "epochs": self.config.epochs,
            "has_pytorch": HAS_TORCH
        }
    
    def get_node_embeddings(self) -> Optional[np.ndarray]:
        """Get learned node embeddings."""
        if not self._fitted or self.model is None:
            return None
        
        self.model.eval()
        with torch.no_grad():
            node_ids = torch.arange(self.n_max, device=self.device)
            embeddings = self.model.node_embeddings(node_ids).cpu().numpy()
        
        return embeddings
    
    def get_adjacency_matrix(self) -> Optional[np.ndarray]:
        """Get the co-occurrence adjacency matrix."""
        if self.adj_matrix is None:
            return None
        return self.adj_matrix.cpu().numpy() if HAS_TORCH else None
