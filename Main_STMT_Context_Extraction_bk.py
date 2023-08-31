import argparse
import os

import pandas
from graph_manager.Main_Trim_CTG import *
from graph_manager.Joern_Node import *
from graph_manager.Graph_Manager import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_vul_lines_list(vul_lines):
    vul_lines_list = []
    for v in vul_lines:
        func_id = v.split("_____")[0]
        line_numbers = v.split("_____")[1].split(",")
        for l in line_numbers:
            vul_lines_list.append(func_id + "_____" + str(float(l)))
    return vul_lines_list


def get_operation_context(node_infos, edge_infos, lineNumbers):
    ctg_operation_ctx = {}
    change_operation_sequence = {}
    for item in lineNumbers:
        if item != "":
            stmt_nodes = node_infos[node_infos["lineNumber"] == item]

            op_ctx, change_op = print_stmt(stmt_nodes, edge_infos)
            ctg_operation_ctx[item] = op_ctx
            change_operation_sequence[item] = change_op
    return ctg_operation_ctx, change_operation_sequence


def main(_skiprows, _nrows, _vtc_filepath, _output_filepath):

    ground_truth = 1
    separate_token = "=" * 100
    if(_skiprows != -1 and _nrows != -1):
        df = pandas.read_csv(_vtc_filepath, skiprows=_skiprows, nrows=_nrows)
    elif(_skiprows == -1 and _nrows != -1):
            df = pandas.read_csv(_vtc_filepath, nrows=_nrows)
    elif(_skiprows != -1 and _nrows == -1):
        df = pandas.read_csv(_vtc_filepath, skiprows=_skiprows)
    else:
        df = pandas.read_csv(_vtc_filepath)
    df.columns = ['commit_id', 'nodes', 'edges', 'vul_lines']

    for idx, row in df.iterrows():
        commit_id = df.at[idx, "commit_id"]
        if not isinstance(df.at[idx, "vul_lines"], str):
            print("skip commit:", commit_id)
            continue
        try:

            print("handle commit:", commit_id)
            vul_lines_list = get_vul_lines_list(df.at[idx, "vul_lines"].split(separate_token))
            node_infos, edge_infos = CTG_main(df, idx, ground_truth, separate_token, "explaining")
            lineNumbers = node_infos.lineNumber.unique()
            ctg_operation_ctx, change_operation_sequences = get_operation_context(node_infos, edge_infos, lineNumbers)

            for item in lineNumbers:
                if item != "":
                    stmt_nodes = node_infos[node_infos["lineNumber"] == item]
                    stmt_nodes = stmt_nodes[stmt_nodes['ALPHA'] == "ADD"]
                    if (len(stmt_nodes) > 0):

                        # cdg_bw_slicing = directed_backward_dependence(edge_infos, item, "CDG")
                        # cdg_fw_slicing = directed_forward_dependence(edge_infos, item, "CDG")
                        # ddg_bw_slicing = directed_backward_dependence(edge_infos, item, "DDG")
                        # ddg_fw_slicing = directed_forward_dependence(edge_infos, item, "DDG")
                        vul_stmt = 0
                        if item in vul_lines_list:
                            vul_stmt = 1
                        data_of_the_stmt = {"commit_id": commit_id, "line_number": item,
                                            "operation_ctx": ctg_operation_ctx[item],
                                            "change_operation_sequence": change_operation_sequences[item],
                                            # "cdg_bw_slicing": get_list_stmts(cdg_bw_slicing, ctg_operation_ctx),
                                            # "cdg_fw_slicing": get_list_stmts(cdg_fw_slicing, ctg_operation_ctx),
                                            # "ddg_bw_slicing": get_list_stmts(ddg_bw_slicing, ctg_operation_ctx),
                                            # "ddg_fw_slicing": get_list_stmts(ddg_fw_slicing, ctg_operation_ctx),
                                            "vul_stmt": vul_stmt
                                            }
                        vul_data = {1: data_of_the_stmt}
                        if not os.path.isfile(_output_filepath):
                            pandas.DataFrame.from_dict(data=vul_data, orient='index').to_csv(_output_filepath, header='column_names')
                        else:  # else it exists so append without writing the header
                            pandas.DataFrame.from_dict(data=vul_data, orient='index').to_csv(_output_filepath, mode='a', header=False)

        except:
            print("exception: ", commit_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vtc_filepath', type=str, default=-1)
    parser.add_argument('--skiprows', type=int, default=-1)
    parser.add_argument('--nrows', type=int, default=-1)

    parser.add_argument('--output_filepath', type=str, default=-1)

    args = parser.parse_args()
    _skiprows = args.skiprows
    _nrows = args.nrows
    _vtc_filepath = args.vtc_filepath
    _output_filepath = args.output_filepath
    main(_skiprows, _nrows, _vtc_filepath, _output_filepath)
