import numpy as np


def trace_reachable_graph(nodes, output_id):
    """
    Returns reachable node IDs and parent-to-child edges in deterministic order.
    """
    id_to_node = {node["id"]: node for node in nodes}

    # Traverse backward from output_id through parent links.
    reachable = set()
    stack = [output_id]

    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        for parent in id_to_node[nid]["parents"]:
            if parent not in reachable:
                stack.append(parent)

    # Reachable IDs in original node-list order.
    reachable_ids = [node["id"] for node in nodes if node["id"] in reachable]

    # Edges in child input order, then parent-list order.
    edges = []
    for node in nodes:
        child = node["id"]
        if child not in reachable:
            continue
        for parent in node["parents"]:
            edges.append([parent, child])

    return reachable_ids, edges
