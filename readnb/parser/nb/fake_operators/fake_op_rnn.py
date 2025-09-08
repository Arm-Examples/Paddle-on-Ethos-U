from typing import Dict, List
from parser.nb.nb_utils import attr_to_dict

# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

# Eelement-wises
@register_fake_nodes
class FakeRnnVisitor(FakeNodesVisitor):

    target = "rnn"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        attrs = kwargs.get('attrs')

        rnn_mode = attrs['mode']
        num_layers = attrs['num_layers']
        is_bidirec = attrs['is_bidirec']

        hidden_size = attrs['hidden_size']
        input_size = attrs['input_size']

        if rnn_mode == 'LSTM':
            if True == is_bidirec:
                output_shape = [input_shape[0], input_shape[1], hidden_size * 2]
            else:
                output_shape = [input_shape[0], input_shape[1], hidden_size]
        else :
            raise ValueError(f"Rnn Not support mode {rnn_mode}")
        return output_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attrs = kwargs.get('attrs')

        input_ = [input for idx, input in enumerate(op["inputs"]) if input['parameter'] == "Input" ]
        it = input_[0]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
        in_type = input_[0]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
        input_dim_order = input_[0]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]


        output_shape = self.shape_infer(input_shape=it, attrs=attrs)

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = in_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order

        return output_shape
