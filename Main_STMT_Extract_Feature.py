import argparse
import os

import pandas

from graph_manager.parse_joern import get_node_edges
import warnings

warnings.filterwarnings("ignore")


def get_topic(nodes):
    topic = ""
    method = nodes[nodes["_label"] == "METHOD"]
    for idx, row in method.iterrows():
        if "METHOD_1.0:" in method.at[idx, "node_label"] and "<global>" not in method.at[idx, "node_label"]:
            topic += method.at[idx, "code"] + " "
    return topic


def get_context(nodes, edges, line_number):
    line_edges = edges[(edges["line_out"] == line_number) | (edges["line_in"] == line_number)]
    innodes = line_edges.innode.unique()
    outnodes = line_edges.outnode.unique()
    line_nodes = nodes[(nodes["id"].isin(innodes)) | (nodes["id"].isin(outnodes))]
    return line_nodes.to_json(orient="records"), line_edges.to_json(orient="records")


def main(_skiprows, _nrows, _changedline_filepath, _vtc_filepath, _output_filepath):
    separate_token = "=" * 100
    if _skiprows != -1 and _nrows != -1:
        changedline_data = pandas.read_csv(_changedline_filepath, skiprows=_skiprows, nrows=_nrows)
    elif _skiprows == -1 and _nrows != -1:
        changedline_data = pandas.read_csv(_changedline_filepath, nrows=_nrows)
    elif _skiprows != -1 and _nrows == -1:
        changedline_data = pandas.read_csv(_changedline_filepath, skiprows=_skiprows)
    else:
        changedline_data = pandas.read_csv(_changedline_filepath)
    changedline_data.columns = ['Unnamed: 0', 'commit_id', 'idx', 'changed_type', 'label',
                                'raw_changed_line', 'blame_line', 'line_number', 'index_ctg']
    ctg_data = pandas.read_csv(_vtc_filepath)
    for idx, row in changedline_data.iterrows():

        commit_id = changedline_data.at[idx, "commit_id"]
        ctg_index = changedline_data.at[idx, "index_ctg"]
        line_number = changedline_data.at[idx, "line_number"]
        stmt = changedline_data.at[idx, "raw_changed_line"]
        tmp = stmt.replace(" ", "").replace("{", "").replace("}", "")
        if len(tmp) == 0:
            continue
        try:
            ctg = ctg_data[ctg_data['commit_id'] == commit_id]
            sub_graph_nodes = ctg.loc[0, "nodes"].split(separate_token)
            sub_graph_edges = ctg.loc[0, "edges"].split(separate_token)
            nodes = ""
            edges = ""

            for sub_graph_idx in range(0, len(sub_graph_nodes)):
                if (sub_graph_nodes[sub_graph_idx].split("_____")[0] == str(ctg_index)):
                    node_content = sub_graph_nodes[sub_graph_idx].split("_____")[1]
                    edge_content = sub_graph_edges[sub_graph_idx].split("_____")[1]

                    nodes, edges = get_node_edges(edge_content, node_content)
                    edges.to_csv("Data/edge_example.csv")
                    break
            topic = get_topic(nodes)
            line_nodes, line_edges = get_context(nodes, edges, line_number)
            stmt = changedline_data.at[idx, "raw_changed_line"]
            data_of_the_stmt = {"commit_id": commit_id, "line_number": line_number,
                                "topic": topic,
                                "context_nodes": line_nodes,
                                "context_edges": line_edges,
                                "stmt": stmt,
                                "label": changedline_data.at[idx, "label"]
                                }
            vul_data = {1: data_of_the_stmt}
            if not os.path.isfile(_output_filepath):
                pandas.DataFrame.from_dict(data=vul_data, orient='index').to_csv(_output_filepath, header='column_names')
            else:  # else it exists so append without writing the header
                pandas.DataFrame.from_dict(data=vul_data, orient='index').to_csv(_output_filepath, mode='a', header=False)
            del data_of_the_stmt, sub_graph_nodes, sub_graph_edges, nodes, edges
        except:
            print("exception: ", commit_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--changedline_filepath', type=str, default=-1)
    parser.add_argument('--vtc_filepath', type=str, default=-1)
    parser.add_argument('--skiprows', type=int, default=-1)
    parser.add_argument('--nrows', type=int, default=-1)

    parser.add_argument('--output_filepath', type=str, default=-1)

    args = parser.parse_args()
    _skiprows = args.skiprows
    _nrows = args.nrows
    _changedline_filepath = args.changedline_filepath
    _vtc_filepath = args.vtc_filepath
    _output_filepath = args.output_filepath
    main(_skiprows, _nrows, _changedline_filepath, _vtc_filepath, _output_filepath)
