import torch
from torch import nn
import torch.nn.functional as F


class GraphDenseModule(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Graph Dense Module (GDM) implementation

        Args:
        - in_features (int): Input feature dimension
        - out_features (int): Output feature dimension
        """
        super(GraphDenseModule, self).__init__()

        # Linear transformation to adjust input feature dimension
        self.linear_transform = nn.Linear(in_features, out_features)

        # Residual weight matrix
        self.res_weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Dense weight matrix
        self.dense_weight = nn.Parameter(torch.FloatTensor(out_features, out_features))

        # Shared weight matrix
        self.shared_weight = nn.Parameter(torch.FloatTensor(out_features, out_features))

        # Initialize weights
        nn.init.xavier_uniform_(self.res_weight)
        nn.init.xavier_uniform_(self.dense_weight)
        nn.init.xavier_uniform_(self.shared_weight)

    def laplacian_transform(self, adj_matrix):
        """
        Compute normalized Laplacian matrix

        Args:
        - adj_matrix (torch.Tensor): Adjacency matrix

        Returns:
        - torch.Tensor: Normalized Laplacian matrix
        """
        # Compute degree matrix
        degree_matrix = torch.sum(adj_matrix, dim=1)
        degree_matrix = torch.diag(degree_matrix)

        # Compute Laplacian matrix
        laplacian = degree_matrix - adj_matrix

        # Normalize Laplacian
        d_inv_sqrt = torch.pow(degree_matrix, -0.5)
        d_inv_sqrt = torch.nan_to_num(d_inv_sqrt, posinf=0, neginf=0)
        normalized_laplacian = torch.matmul(torch.matmul(d_inv_sqrt, laplacian), d_inv_sqrt)

        return normalized_laplacian

    def forward(self, x, adj_matrix):
        """
        Forward pass of Graph Dense Module

        Args:
        - x (torch.Tensor): Input node features
        - adj_matrix (torch.Tensor): Adjacency matrix

        Returns:
        - torch.Tensor: Transformed node features
        """
        # Normalize input features dimension
        x_transformed = self.linear_transform(x)

        # Compute Laplacian transform
        lap_matrix = self.laplacian_transform(adj_matrix)

        # Graph convolution operation
        graph_conv = torch.matmul(lap_matrix, torch.matmul(x_transformed, self.shared_weight))

        # Residual and dense connections
        residual_term = torch.matmul(x, self.res_weight)
        dense_term = torch.matmul(x_transformed, self.dense_weight)

        # Combine terms with activation
        output = F.relu(graph_conv + residual_term + dense_term)

        return output


class GraphDenseBlock(nn.Module):
    def __init__(self, num_heads, in_features, out_features, num_gdm_layers):
        """
        Graph Dense Block (GDB) implementation

        Args:
        - num_heads (int): Number of attention heads
        - in_features (int): Input feature dimension
        - out_features (int): Output feature dimension
        - num_gdm_layers (int): Number of Graph Dense Module layers
        """
        super(GraphDenseBlock, self).__init__()

        self.heads = nn.ModuleList([
            nn.ModuleList([
                GraphDenseModule(in_features if l == 0 else out_features, out_features)
                for l in range(num_gdm_layers)
            ])
            for _ in range(num_heads)
        ])

    def forward(self, x, adj_matrix):
        """
        Forward pass of Graph Dense Block

        Args:
        - x (torch.Tensor): Input node features
        - adj_matrix (torch.Tensor): Adjacency matrix

        Returns:
        - torch.Tensor: Transformed node features
        """
        # Process each attention head
        head_outputs = []
        for head in self.heads:
            head_x = x
            for gdm in head:
                head_x = gdm(head_x, adj_matrix)
            head_outputs.append(head_x)

        # Sum pooling to aggregate head outputs
        output = torch.sum(torch.stack(head_outputs), dim=0)

        return output


class GraphDenseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_blocks, num_heads, num_gdm_layers):
        """
        Graph Dense Encoder (GDE) implementation

        Args:
        - input_dim (int): Initial input feature dimension
        - hidden_dim (int): Hidden layer feature dimension
        - output_dim (int): Final output feature dimension
        - num_blocks (int): Number of Graph Dense Blocks
        - num_heads (int): Number of attention heads per block
        - num_gdm_layers (int): Number of Graph Dense Module layers per block
        """
        super(GraphDenseEncoder, self).__init__()

        self.blocks = nn.ModuleList()

        # First block adjusts input dimension
        self.blocks.append(
            GraphDenseBlock(num_heads, input_dim, hidden_dim, num_gdm_layers)
        )

        # Subsequent blocks
        for _ in range(num_blocks - 1):
            self.blocks.append(
                GraphDenseBlock(num_heads, hidden_dim, hidden_dim, num_gdm_layers)
            )

        # Final block to output dimension
        self.final_block = GraphDenseBlock(num_heads, hidden_dim, output_dim, num_gdm_layers)

    def forward(self, x, adj_matrix):
        """
        Forward pass of Graph Dense Encoder

        Args:
        - x (torch.Tensor): Input node features
        - adj_matrix (torch.Tensor): Adjacency matrix

        Returns:
        - torch.Tensor: Final node representations
        """
        # Process through intermediate blocks
        for block in self.blocks:
            x = block(x, adj_matrix)

        # Final block
        x = self.final_block(x, adj_matrix)

        return x
