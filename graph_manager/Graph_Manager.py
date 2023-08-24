G_NODE_ID = "node_id"
G_CDG_IN = "CDG_IN"
G_CDG_OUT = "CDG_OUT"


def backward_slicing(edges, line_id, etype):
    cdg_bw_slicies = [line_id]
    checked_stmts = []
    for s in cdg_bw_slicies:
        if s not in checked_stmts:
            checked_stmts.append(s)
            for e_idx, e_row in edges.iterrows():
                if e_row["line_in"] in cdg_bw_slicies and e_row["etype"] == etype:
                    if e_row["line_out"] not in cdg_bw_slicies:
                        cdg_bw_slicies.append(e_row["line_out"])
    return cdg_bw_slicies


def forward_slicing(edges, line_id, etype):
    cdg_fw_slicies = [line_id]
    checked_stmts = []
    for s in cdg_fw_slicies:
        if s not in checked_stmts:
            checked_stmts.append(s)
            for e_idx, e_row in edges.iterrows():
                if e_row["line_out"] in cdg_fw_slicies and e_row["etype"] == etype:
                    if e_row["line_in"] not in cdg_fw_slicies:
                        cdg_fw_slicies.append(e_row["line_in"])
    return cdg_fw_slicies

def directed_backward_dependence(edges, line_id, etype):
    cdg_bw_slicies = [line_id]
    for e_idx, e_row in edges.iterrows():
        if e_row["line_in"] == line_id and e_row["etype"] == etype:
            if e_row["line_out"] not in cdg_bw_slicies:
                cdg_bw_slicies.append(e_row["line_out"])
    return cdg_bw_slicies


def directed_forward_dependence(edges, line_id, etype):
    cdg_fw_slicies = [line_id]
    for e_idx, e_row in edges.iterrows():
        if e_row["line_out"] == line_id and e_row["etype"] == etype:
            if e_row["line_in"] not in cdg_fw_slicies:
                cdg_fw_slicies.append(e_row["line_in"])
    return cdg_fw_slicies