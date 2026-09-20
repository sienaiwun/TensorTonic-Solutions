import numpy as np

def value_addition_node(left, right, output_id):
    """
    Returns: an addition node that retains the two supplied leaf records as ordered parents
    """
    return  {"id":output_id,
             "data": left["data"] + right["data"],
             "grad": 0.0,
             "op":"+",
             "parents":[left, right]
            }
    
    pass
