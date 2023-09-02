
import torch
from gensim.models import KeyedVectors
from torch_geometric.data import Data

change_operations = ["ADD", "DELETE", "REMAIN"]
edge_types = ["AST", "CFG", "CDG", "DDG"]


def load_nodes(nodes, index_col, encoders=None, **kwargs):
    mapping = {index: i for i, index in enumerate(nodes["id"].unique())}
    x = None
    if encoders is not None:
        xs = [encoder(nodes[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    return x, mapping


def load_edges(edges, src_index_col, src_mapping, dst_index_col, dst_mapping, edge_type_col,
               encoders=None, **kwargs):

    src = [src_mapping[index] for index in edges[src_index_col]]
    dst = [dst_mapping[index] for index in edges[dst_index_col]]

    edge_index = torch.tensor([src, dst])
    types = [edge_types.index(edge) for edge in edges[edge_type_col]]

    edge_type = torch.tensor(types)
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(edges[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    return edge_index, edge_type, edge_attr


class ContentEncoder(object):
    def __init__(self, device=None):
        self.device = device
        self.model = KeyedVectors.load("Model/word2vec.wordvectors", mmap='r')


    @torch.no_grad()
    def __call__(self, df):
        df = df.fillna("")
        x = torch.zeros(len(df), 32)

        for i, item in enumerate(df.values):
            tokens = item.split(" ")
            tmp = []
            for token in tokens:
                if token in self.model:
                    tmp.append(torch.from_numpy(self.model[token]))
            if len(tmp) > 0:
                x[i] =  torch.sum(torch.stack(tmp), dim=0)
        return x.cpu()



class OneHotEncoder(object):
    def __init__(self, dicts):
        self.dicts = dicts

    def __call__(self, df):
        x = torch.zeros(len(df), len(self.dicts))
        for i, col in enumerate(df.values):
            try:
                x[i, self.dicts.index(col)] = 1
            except:
                x[i, 0] = 0
        return x



def embed_graph(commit_id, ground_truth, nodes, edges):
    node_x, node_mapping = load_nodes(
        nodes, index_col='id', encoders={
            'ALPHA': OneHotEncoder(change_operations),
            'node_content': ContentEncoder()

        })


    edge_index, edge_type, edge_label = load_edges(
        edges,
        src_index_col='outnode',
        src_mapping=node_mapping,
        dst_index_col='innode',
        dst_mapping=node_mapping,
        edge_type_col = 'etype',
        encoders={'change_operation': OneHotEncoder(change_operations)}
    )


    data = Data()
    data.x = node_x
    data.edge_index = edge_index
    data.edge_attr = edge_label
    data.edge_type = edge_type

    data.y = torch.tensor([ground_truth], dtype = int)
    return data


def get_node_mapping(nodes, edges):

    mapping = {index: i for i, index in enumerate(nodes["id"].unique())}
    return mapping

def load_edge_mapping(edges, src_index_col, src_mapping, dst_index_col, dst_mapping):
    #df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in edges[src_index_col]]
    dst = [dst_mapping[index] for index in edges[dst_index_col]]

    edge_index = torch.tensor([src, dst])
    return edge_index