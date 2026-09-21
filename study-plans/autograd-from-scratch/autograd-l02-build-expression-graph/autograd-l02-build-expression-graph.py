import numpy as np

def build_expression_graph(leaves, operations):
    nodes = []
    values = {}

    # Add all leaf nodes first, in the given order.
    for leaf in leaves:
        node = {
            "id": leaf["id"],
            "data": leaf["data"],
            "grad": 0.0,
            "op": "",
            "parents": []
        }
        nodes.append(node)
        values[node["id"]] = node["data"]

    final_id = leaves[-1]["id"] if leaves else None

    # Add operation nodes in the given order.
    for rec in operations:
        left_id = rec["left"]
        right_id = rec["right"]
        op = rec["op"]

        left_val = values[left_id]
        right_val = values[right_id]

        # In the examples, '+' is addition and '' / '*' is multiplication.
        if op == "+":
            result = left_val + right_val
        else:
            result = left_val * right_val

        node = {
            "id": rec["id"],
            "data": result,
            "grad": 0.0,
            "op": op,
            "parents": [left_id, right_id]
        }

        nodes.append(node)
        values[node["id"]] = node["data"]
        final_id = node["id"]

    return nodes, final_id
