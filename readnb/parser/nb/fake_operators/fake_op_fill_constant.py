from typing import Dict, List
from paddle.lite.fbs.proto.VarType_.Type import Type

# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

@register_fake_nodes
class FakeFillConstantVisitor(FakeNodesVisitor):

    target = "fill_constant"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shapes = kwargs.get('input_shapes')
        axis = kwargs.get('axis')

        if input_shapes and axis:
            sum = 0
            for shape in input_shapes:
                sum += shape[axis]

            output_shape = input_shapes[0]
            output_shape[axis] = sum
        else:
            raise ValueError(f"Missing necessary parameters {input_shapes} {axis}")
        return output_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        op_attrs = kwargs.get('attrs')

        output_shape = list(op_attrs['shape'])

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = Type.INT8
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = list([0, 2, 3, 1])

        return output_shape


@register_fake_nodes
class FakeFillConstantBatchSizeLikeVisitor(FakeNodesVisitor):

    target = "fill_constant_batch_size_like"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        attrs = kwargs.get('attrs')

        in_dim_id = attrs.get('input_dim_idx')
        out_dim_id = attrs.get('output_dim_idx')
        output_shape = attrs.get('shape')

        if -1 in output_shape:
            if in_dim_id != -1 and out_dim_id != -1 and output_shape[out_dim_id] == -1:
                output_shape[out_dim_id] = input_shape[in_dim_id]
            else:
                raise ValueError(f"Fill Constant Batch Size like failed in shape:{input_shape} out shape:{output_shape}, in dim id{in_dim_id}, out dim id {out_dim_id}")
        return output_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        op_attrs = kwargs.get('attrs')

        for idx, input in enumerate(op["inputs"]):
            if op["inputs"][idx]["arguments"] == None or len(op["inputs"][idx]["arguments"]) == 0:
                continue
            it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
            # input_dim_order = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]

        output_shape = self.shape_infer(input_shape=it, attrs=op_attrs)

        if len(output_shape) == 4: # Means this is a NHWC Tensor use.
            out_dim_order = list([0, 2, 3, 1])
        else:
            out_dim_order = list(range(len(output_shape)))

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = Type.INT8
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = out_dim_order
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = list([0, 2, 3, 1])

        return output_shape