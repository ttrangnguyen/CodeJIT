import argparse
import json
import os
from os import listdir
from os.path import isfile, join
import torch
import pandas
from graph_embedding.relational_graph import *
from graph_manager.Main_Trim_CTG import generate_node_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--embedding_graph_dir', type=str, help='dir to save embedding graph')
    parser.add_argument('--skiprows', type=int, default=-1)
    parser.add_argument('--nrows', type=int, default=-1)
    args = parser.parse_args()

    file_path = args.file_path
    embedding_graph_dir = args.embedding_graph_dir
    _skiprows = args.skiprows
    _nrows = args.nrows
    if _skiprows != -1 and _nrows != -1:
        data = pandas.read_csv(file_path, skiprows=_skiprows, nrows=_nrows)
    elif _skiprows == -1 and _nrows != -1:
        data = pandas.read_csv(file_path, nrows=_nrows)
    elif _skiprows != -1 and _nrows == -1:
        data = pandas.read_csv(file_path, skiprows=_skiprows)
    else:
        data = pandas.read_csv(file_path)


    data.columns = ['Unnamed: 0.1', 'Unnamed: 0', 'commit_id', 'line_number', 'index_ctg',
         'topic', 'context_nodes', 'context_edges', 'stmt', 'label']

    data = data[data["context_nodes"] != "[]"]
    for idx, row in data.iterrows():
        try:
            nodes = data.at[idx, "context_nodes"]
            nodes = json.loads(nodes)
            nodes = pandas.DataFrame.from_records(nodes)
            nodes = generate_node_content(nodes)
            edges = data.at[idx, "context_edges"]
            edges = json.loads(edges)
            edges = pandas.DataFrame.from_records(edges)
            commit_id = data.at[idx, "commit_id"]
            line_number = data.at[idx, "line_number"]
            ctg_index = data.at[idx, "index_ctg"]
            graph = embed_graph(data.at[idx, "commit_id"], data.at[idx, "label"], nodes, edges)
            torch.save(data, os.path.join(embedding_graph_dir, "data_{}_{}_{}.pt".format(commit_id, ctg_index, line_number)))
        except:
            print("exception")