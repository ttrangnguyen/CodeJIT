import pandas

from graph_manager.parse_joern import get_node_edges
from queue import Queue

from graph_manager.Joern_Node import NODE
# from graph_embedding.relational_graph import *

def find_root_node(stmt_edges):
    outnodes = stmt_edges["outnode"].tolist()
    innodes = stmt_edges["innode"].tolist()
    for n in outnodes:
        if n not in innodes:
            return n


def forward_slice_graph(nodes, edges, etype):
    changed_nodes = nodes[nodes["ALPHA"] != "REMAIN"]["id"].tolist()
    nodes_in_changed_edges = edges[(edges["change_operation"] != "REMAIN") & (edges["etype"] == etype)]["outnode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["outnode"] == n_id) & (edges["etype"] == etype)]["innode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited


def backward_slice_graph(nodes, edges, etype):
    changed_nodes = nodes[nodes["ALPHA"] != "REMAIN"]["id"].tolist()
    nodes_in_changed_edges = edges[(edges["change_operation"] != "REMAIN") & (edges["etype"] == etype)]["innode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["innode"] == n_id) & (edges["etype"] == etype)]["outnode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited

def forward_slice_a_node(stmt_id, nodes, edges, etype):
    stmt_nodes = nodes[nodes['lineNumber'] == stmt_id]
    changed_nodes = stmt_nodes[stmt_nodes["ALPHA"] != "REMAIN"]["id"].tolist()
    nodes_in_changed_edges = edges[(edges["change_operation"] != "REMAIN") & (edges["etype"] == etype)]["outnode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["outnode"] == n_id) & (edges["etype"] == etype)]["innode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited


def backward_slice_a_node(stmt_id, nodes, edges, etype):
    stmt_nodes = nodes[nodes['lineNumber'] == stmt_id]
    changed_nodes = stmt_nodes[stmt_nodes["ALPHA"] != "REMAIN"]["id"].tolist()
    nodes_in_changed_edges = edges[(edges["change_operation"] != "REMAIN") & (edges["etype"] == etype)]["innode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["innode"] == n_id) & (edges["etype"] == etype)]["outnode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited

def aggregate_edges(stmt_nodes, edges):
    node_ids = stmt_nodes["id"].to_list()
    stmt_edges = edges[(edges["innode"].isin(node_ids) & edges["outnode"].isin(node_ids)) & (edges["etype"] == "AST")]
    root = find_root_node(stmt_edges)
    if root is not None:
        for n in node_ids:
            if n != root:
                edges.loc[(edges["innode"] == n) & (edges["etype"] != "AST"), "innode"] = root
                edges.loc[(edges["outnode"] == n) & (edges["etype"] != "AST"), "outnode"] = root
    return edges


def generate_node_content(nodes):
    content = []
    for i, n in nodes.iterrows():
        tmp = NODE(nodes.at[i, "id"], nodes.at[i, "_label"], nodes.at[i, "code"],
                   nodes.at[i, "name"], nodes.at[i, "ALPHA"])
        content.append(tmp.print_node())
    nodes["node_content"] = content
    return nodes


def commit_full_graph(df, idx, separate_token):
    sub_graph_nodes = df.at[idx, "nodes"].split(separate_token)
    sub_graph_edges = df.at[idx, "edges"].split(separate_token)
    commit_nodes = []
    commit_edges = []
    for sub_graph_idx in range(0, len(sub_graph_nodes)):
        node_content = sub_graph_nodes[sub_graph_idx].split("_____")[1]
        edge_content = sub_graph_edges[sub_graph_idx].split("_____")[1]

        nodes, edges = get_node_edges(edge_content, node_content)

        nodes = nodes.dropna(subset=['ALPHA'])
        edges = edges.dropna(subset=['change_operation'])

        removed_nodes = nodes[nodes["lineNumber"] == ""]["id"].tolist()
        nodes = nodes[nodes["lineNumber"] != ""]
        edges = edges[~edges["innode"].isin(removed_nodes)]
        edges = edges[~edges["outnode"].isin(removed_nodes)]

        removed_nodes = nodes[nodes["_label"] == "BLOCK"]["id"].tolist()
        nodes = nodes[nodes["_label"] != "BLOCK"]
        edges = edges[~edges["innode"].isin(removed_nodes)]
        edges = edges[~edges["outnode"].isin(removed_nodes)]

        lineNumbers = nodes.lineNumber.unique()
        kept_nodes = set()

        for x in lineNumbers:
            stmt_nodes = nodes[nodes["lineNumber"] == x]
            edges = aggregate_edges(stmt_nodes, edges)
        edges = edges[edges.innode != edges.outnode]
        edges = edges.drop_duplicates(subset=["innode", "outnode", "etype"], keep='first')

        nodes = generate_node_content(nodes)

        node_list = nodes["id"].tolist()

        edges = edges[edges["outnode"].isin(node_list)]
        edges = edges[edges["innode"].isin(node_list)]


        node_ids = nodes["id"]
        node_ids = [str(sub_graph_idx) + "_" + str(s) for s in node_ids]
        nodes["id"] = node_ids

        node_linenumbers = nodes["lineNumber"]
        node_linenumbers = [sub_graph_nodes[sub_graph_idx].split("_____")[0] + "_____" + str(s) for s in node_linenumbers]
        nodes["lineNumber"] = node_linenumbers

        in_nodes = edges["innode"]
        in_nodes = [str(sub_graph_idx) + "_" + str(s) for s in in_nodes]
        edges["innode"] = in_nodes

        out_nodes = edges["outnode"]
        out_nodes = [str(sub_graph_idx) + "_" + str(s) for s in out_nodes]
        edges["outnode"] = out_nodes

        line_ins = edges["line_in"]
        line_ins = [str(sub_graph_idx) + "_____" + str(s) for s in line_ins]
        edges["line_in"] = line_ins

        line_outs = edges["line_out"]
        line_outs = [str(sub_graph_idx) + "_____" + str(s) for s in line_outs]
        edges["line_out"] = line_outs

        commit_nodes.append(nodes)
        commit_edges.append(edges)


    all_nodes = pandas.concat(commit_nodes)
    all_edges = pandas.concat(commit_edges)
    return all_nodes, all_edges

def CTG_main(df, idx, ground_truth, separate_token, mode="embedding"):
        commit_id = df.at[idx, "commit_id"]
        sub_graph_nodes = df.at[idx, "nodes"].split(separate_token)
        sub_graph_edges = df.at[idx, "edges"].split(separate_token)
        commit_nodes = []
        commit_edges = []
        for sub_graph_idx in range(0, len(sub_graph_nodes)):
            node_content = sub_graph_nodes[sub_graph_idx].split("_____")[1]
            edge_content = sub_graph_edges[sub_graph_idx].split("_____")[1]

            nodes, edges = get_node_edges(edge_content, node_content)
            
            nodes = nodes.dropna(subset=['ALPHA'])
            edges = edges.dropna(subset=['change_operation'])

            removed_nodes = nodes[nodes["lineNumber"] == ""]["id"].tolist()
            nodes = nodes[nodes["lineNumber"] != ""]
            edges = edges[~edges["innode"].isin(removed_nodes)]
            edges = edges[~edges["outnode"].isin(removed_nodes)]

            removed_nodes = nodes[nodes["_label"] == "BLOCK"]["id"].tolist()
            nodes = nodes[nodes["_label"] != "BLOCK"]
            edges = edges[~edges["innode"].isin(removed_nodes)]
            edges = edges[~edges["outnode"].isin(removed_nodes)]

            lineNumbers = nodes.lineNumber.unique()
            kept_nodes = set()

            for x in lineNumbers:
                stmt_nodes = nodes[nodes["lineNumber"] == x]
                edges = aggregate_edges(stmt_nodes, edges)
            edges = edges[edges.innode != edges.outnode]
            edges = edges.drop_duplicates(subset=["innode", "outnode", "etype"], keep='first')

            visited = forward_slice_graph(nodes, edges, "CDG")
            kept_nodes.update(visited)
            visited = backward_slice_graph(nodes, edges, "CDG")
            kept_nodes.update(visited)

            visited = forward_slice_graph(nodes, edges, "DDG")
            kept_nodes.update(visited)
            visited = backward_slice_graph(nodes, edges, "DDG")
            kept_nodes.update(visited)

            kept_nodes = list(kept_nodes)
            kept_line_numbers = nodes[nodes["id"].isin(kept_nodes)]["lineNumber"].to_list()
            nodes = nodes[nodes["lineNumber"].isin(kept_line_numbers)]
            kept_nodes = nodes["id"].to_list()
            edges = edges[(edges["innode"].isin(kept_nodes)) | (edges["outnode"].isin(kept_nodes))]
            
            nodes = generate_node_content(nodes)

            node_list = nodes["id"].tolist()
            
            edges = edges[edges["outnode"].isin(node_list)]
            edges = edges[edges["innode"].isin(node_list)]
            

            node_ids = nodes["id"]
            node_ids = [str(sub_graph_idx) + "_" + str(s) for s in node_ids]
            nodes["id"] = node_ids

            node_linenumbers = nodes["lineNumber"]
            node_linenumbers = [sub_graph_nodes[sub_graph_idx].split("_____")[0] + "_____" + str(s) for s in node_linenumbers]
            nodes["lineNumber"] = node_linenumbers

            in_nodes = edges["innode"]
            in_nodes = [str(sub_graph_idx) + "_" + str(s) for s in in_nodes]
            edges["innode"] = in_nodes

            out_nodes = edges["outnode"]
            out_nodes = [str(sub_graph_idx) + "_" + str(s) for s in out_nodes]
            edges["outnode"] = out_nodes

            line_ins = edges["line_in"]
            line_ins = [str(sub_graph_idx) + "_____" + str(s) for s in line_ins]
            edges["line_in"] = line_ins

            line_outs = edges["line_out"]
            line_outs = [str(sub_graph_idx) + "_____" + str(s) for s in line_outs]
            edges["line_out"] = line_outs

            commit_nodes.append(nodes)
            commit_edges.append(edges)

        
        all_nodes = pandas.concat(commit_nodes)
        all_edges = pandas.concat(commit_edges)
        return all_nodes, all_edges
        # if ground_truth == 1:
        #   all_nodes.to_csv("Data/sliced_graph_before_embedding/VTC_Graph/node/node_{}.csv".format(commit_id))
        #   all_edges.to_csv("Data/sliced_graph_before_embedding/VTC_Graph/edge/edge_{}.csv".format(commit_id))
        # else:
        #   all_nodes.to_csv("Data/sliced_graph_before_embedding/VFC_Graph/node/node_{}.csv".format(commit_id))
        #   all_edges.to_csv("Data/sliced_graph_before_embedding/VFC_Graph/edge/edge_{}.csv".format(commit_id))
        # return
        # if mode == "embedding":
        #   data = embed_graph(commit_id, ground_truth, pandas.concat(commit_nodes),  pandas.concat(commit_edges))
        #   return data
        # elif mode == "explaining":
        #   node_infos = pandas.concat(commit_nodes)
        #   edge_infos = pandas.concat(commit_edges)
        #   node_mapping = get_node_mapping(node_infos, edge_infos)
        #   #edge_index = load_edge_mapping(edge_infos, src_index_col="outnode", src_mapping = node_mapping, dst_index_col="innode", dst_mapping= node_mapping)
        #   return node_mapping, node_infos, edge_infos
      


# if __name__ == '__main__':
#     df = pandas.read_csv("Data/vtc_samples.csv", nrows=1)
#     print(df.columns)
#     separate_token = "=" * 33
#     main(df, separate_token)