from typing import Dict, List
import numpy as np
# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)
from parser.nb.nb_utils import get_param_safe
from .op_scale_scan import find_outscale_from_next
# Eelement-wises
@register_fake_nodes
class FakeElementwiseVisitor(FakeNodesVisitor):

    target = "elementwise"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        op = kwargs.get('op')
        input_shapes = kwargs.get('input_shapes')
        attrs = kwargs.get('attrs')
        params = kwargs.get('params')

        axis = attrs['axis']
        res_str, bd_shapes = self.__broadcast(axis, input_shapes)

        # if res_str is True:
        #     for idx, shape in enumerate(input_shapes):
        #         params = kwargs.get('params')
        #         param = get_param_safe(params, op["inputs"][idx]["arguments"][0]["name"])
        #         if param != None:
        #             tensor_data = param.tensor['data']
        #             td = np.array(tensor_data)
        #             td = np.broadcast_to(td, shape)
        #             param.tensor['data'] = td.flatten().tolist()
        #             param.tensor['dims'] = shape

        return res_str, bd_shapes

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attrs = kwargs.get('attrs')
        params = kwargs.get('params')

        input_shapes = []
        in_types = []
        for idx, input in enumerate(op["inputs"]):
            it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
            in_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
            if 'dt_dim_order' in op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]:
                input_dim_order = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]
            else :
                input_dim_order = list(range(len(it)))
            input_shapes.append(it)
            in_types.append(in_type)

        _res, output_shape = self.shape_infer(op=op, input_shapes=input_shapes, attrs=attrs, params=params)

        out_type = self.__type_infer(in_types)

        self.__scale_infer(op, attrs)

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = out_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order

    def __type_infer(self, input_types:list):
        # if all(input_types[0] == t for t in input_types):
            return input_types[0]
        # else:
            # raise ValueError(f"[ERR] Eelementwise data type not same {input_types}")

    def __broadcast(self, axis, shapes:list):
        result_shape = []

        if 2 != len(shapes):
            return f"Only Support 2 tensors"
        elif False == (all(len(s) == len(shapes[0]) for s in shapes)):
            if axis < 0:
                if isinstance(axis, int):
                    t_shape = shapes[0] if len(shapes[0]) > len(shapes[1]) else shapes[1]
                    axis = len(t_shape) + axis
                else:
                    return f"Axis must be int, but got {type(axis)}"
                # return f"Not support Broadcast shapes:{shapes} , axis {axis}", None
            # else:
                # for id, s in enumerate(shapes):
                #     if len(s) < len(shapes[0]):
                #         shapes[id] = shapes[0]
            for id, s in enumerate(shapes):
                    if len(s) < len(shapes[0]):
                        shapes[id] = shapes[0]

        for dim0, dim1 in zip(*shapes):
            if dim0 == dim1:
                result_shape.append(dim0)
            elif dim0 == 1:
                result_shape.append(dim1)
            elif dim1 == 1:
                result_shape.append(dim0)
            else:
                # return f"Not allow to BroadCast {shapes}", None
                return f"Not allow to BroadCast {shapes}", shapes[0], None

        return True, result_shape

    def __scale_infer(self, op, op_attrs):
        nexts = op['next_op']
        alias = op_attrs['alias']
        if alias == 'int8_out':
            pass
        elif alias == 'def':
            in_scales = [value for key, value in op_attrs.items() if key in "X0_scale" or key in "Y0_scale"]
            # TinyPose elementwise_add case
            if len(in_scales) == 0:
                in_scales = [value for key, value in op_attrs.items() if key in "Scale_x" or key in "Scale_y"]
            if False == any(scale for scale in in_scales):
                raise ValueError(f"Element with no X0_scale or Y0_scale")
            out_scales = []
            for next_id, next in enumerate(op['next_op']):
                out_scales.append(find_outscale_from_next(op, op_attrs, next_id, next))

            op['attrs'].append({"name": "Output0_scale", "val": list(set(out_scales)) })
        else:
            raise ValueError(f"Not support other alias type.")


