import torch
#from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GAT
 
class ParityGameNetwork(torch.nn.Module):
    def __init__(self, hidden_channels_nodes, hidden_channels_edges, core_iterations):
        super().__init__()
        self.core = GAT(3, hidden_channels_nodes, core_iterations, jk='lstm', flow='target_to_source')
        #self.core = GCN(3, hidden_channels_nodes, core_iterations, jk='lstm', flow='target_to_source')
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels_nodes, hidden_channels_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )
        
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels_nodes, hidden_channels_edges),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x, edge_index):
        x = self.core(x, edge_index)
        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=1)
        
        return (self.node_classifier(x), self.edge_classifier(edge_rep))