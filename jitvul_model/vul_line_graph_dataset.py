import os
from torch.utils.data import DataLoader, Dataset
import torch
import random


class GraphDataset(Dataset):
    def __init__(self, _datapoint_files, graph_data_path):
        self.graph_dir = graph_data_path
        files = []
        for f in _datapoint_files:
            graph_file = os.path.join(self.graph_dir, f)
            if os.path.isfile(graph_file):
                try:
                    torch.load(graph_file)
                    files.append(graph_file)
                except:
                    print("graph load exception:", graph_file)

        print("data size:", len(files))
        self.datapoint_files = files

    def __getitem__(self, index):
        graph_file = self.datapoint_files[index]

        graph = torch.load(graph_file)
        file_name = graph_file.split("/")[-1]
        return graph, file_name, index

    def __len__(self):
        return len(self.datapoint_files)


# only batch = 1
def collate_batch(batch):
    _data = batch[0]
    return _data
