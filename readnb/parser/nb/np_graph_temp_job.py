# this is static functions temp work.
from paddle.lite.fbs.proto.VarType_.Type import Type

def temp_feed_job(op, input_shape={}):
    dim_order = tuple([0,2,3,1])
    for idx, output in enumerate(op['outputs']):
        op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = Type.INT8
        op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = dim_order
        if input_shape != {}:
            real_input_name = op["outputs"][idx]["arguments"][0]["name"]
            real_input_name = real_input_name.replace("_tmp_0", "")
            if real_input_name in input_shape.keys():
                op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = input_shape[real_input_name]
            return input_shape
        else:
            output_shape = op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
            output_shape[0] = 1
            return output_shape

def temp_fetch_job(op):
    dim_order = tuple([0,2,3,1])
    for idx, input in enumerate(op["inputs"]):
        op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = dim_order
