import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import pg_parser as parser
import os
    
class InvalidDataException(Exception):
    """ Exception if solutions do not match games"""
    
class ParityGameDataset(InMemoryDataset):

    def __init__(self, root, games_folder, solutions_folder, transform=None, pre_transform=None, pre_filter=None):
        self.games_folder = games_folder
        self.solutions_folder = solutions_folder
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def make_graph(self, game_file, solution_file):
        #print('Process game ' + game_file)
        nodes, edges = parser.get_nodes_and_edges(game_file)
        regions_0, strategy_0, regions_1, strategy_1 = parser.get_solution(solution_file)
        
        y_nodes = torch.zeros(nodes.shape[0], dtype=torch.long)
        y_nodes[regions_1] = 1
        
        y_edges = torch.zeros(edges.shape[0], dtype=torch.long)
        index_0 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_0]
        index_1 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_1]
        y_edges[index_0] = 1
        y_edges[index_1] = 1
        
        return Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(), y_nodes=y_nodes, y_edges=y_edges)
    
    def check_if_files_match(self, game_files, solution_files):
        if not len(game_files) == len(solution_files):
            raise(InvalidDataException(f'There are {len(game_files)} games but {len(solution_files)} solutions.'))
        for i in range(0, len(game_files)):
            if not 'solution_' + game_files[i] == solution_files[i]:
                raise(InvalidDataException(f'File pair {i} ({game_files[i]}, {solution_files[i]} do not match in their names.'))
                
        return True
        
    def process(self):
    
        game_files = os.listdir(self.games_folder)
        solution_files = os.listdir(self.solutions_folder)
        
        if self.check_if_files_match(game_files, solution_files):
            files = zip(game_files, solution_files)
            
        # Read data into huge `Data` list.
        data_list = [self.make_graph(self.games_folder + '/' + game_file, self.solutions_folder + '/' + solution_file) for game_file, solution_file in files]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])