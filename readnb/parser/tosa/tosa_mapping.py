# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# PyTorch to Tosa mapping - simple mapping functions and multi-type extraction
# of key information. These are used by the initial compile stage which captures
# the standardised TOSA representation.
#

import serializer.tosa_serializer as ts
from paddle.lite.fbs.proto.VarType_.Type import Type
# from parser.nb.enums import PrecisionType

UNSUPPORTED_DTYPES = (
    Type.INT64,
    Type.FP64,
    Type.DENSE_TENSOR,
    Type.SELECTED_ROWS,
    Type.FEED_MINIBATCH,
    Type.FETCH_LIST,
    Type.STEP_SCOPES,
    Type.LOD_RANK_TABLE,
    Type.DENSE_TENSOR_ARRAY,
    Type.PLACE_LIST,
    Type.READER,
    Type.RAW,
    Type.TUPLE,
    Type.SIZE_T,
)

DTYPE_MAP = {
    Type.FP32  : ts.DType.FP32,
    Type.INT8  : ts.DType.INT8,
    Type.INT32 : ts.DType.INT32,
    Type.FP16  : ts.DType.FP16,
    Type.BOOL  : ts.DType.BOOL,
    Type.INT16 : ts.DType.INT16,
    Type.UINT8 : ts.DType.UINT8,
}

# DTYPE_MAP = {
#     0: ts.DType.UNKNOWN,  # kUnk
#     1: ts.DType.FP32,     # kFloat
#     2: ts.DType.INT8,     # kInt8
#     3: ts.DType.INT32,    # kInt32
#     5: ts.DType.FP16,     # kFP16
#     6: ts.DType.BOOL,     # kBool
#     8: ts.DType.INT16,    # kInt16
#     9: ts.DType.UINT8,    # kUInt8
# }

def map_dtype(data_type):
    assert data_type not in UNSUPPORTED_DTYPES, f"Unsupported type: {data_type}"
    assert data_type in DTYPE_MAP, f"Unknown type: {data_type}"
    return DTYPE_MAP[data_type]

def extract_tensor_var(var):
    if var['type'] != Type.DENSE_TENSOR:
        RuntimeWarning(f"Extract tensor var dense_tensor type is not 7:DENSE_TENSOR which is:{var['type']}")
    dtype = map_dtype(var['dense_tensor']['dt_type'])
    shape = tuple(var['dense_tensor']['dt_dims'])

    if 'dt_dim_order' in var['dense_tensor'].keys():
        dim_order = var['dense_tensor']['dt_dim_order']
    else:
        dim_order = tuple(range(len(shape)))

    return (dtype, shape, dim_order)


# Class to capture arguments and turn into tensor references for TOSA OPs
class TosaArg:
    def __process_node(self, argument):
        self.name = argument['name']
        self.dtype, self.shape, self.dim_order = extract_tensor_var(argument['type'])

    def __process_list(self, argument):
        self.special = list(argument)

    def __process_number(self, argument):
        self.number = argument

    def set_dtype(self, type_id):
        self.dtype = map_dtype(type_id)

    def __init__(self, argument) -> None:
        self.name = None
        self.dtype = None
        self.shape = None
        self.dim_order = None
        self.special = None

        if argument is None:
            return

        if isinstance(argument, dict):
            self.__process_node(argument)
            return
        if isinstance(argument, list):
            self.__process_list(argument)
            return
        if isinstance(argument, int):
            self.__process_number(argument)
            return
        if isinstance(argument, float):
            self.__process_number(argument)
            return

        RuntimeError(
            f"Unhandled node input argument: {argument}, of type {type(argument)}"
        )


if __name__  == "__main__":


    var_type_dict = {}
    var_type_dict['type'] = 7
    #
    var_dt = {}
    var_dt['lod_lv'] = 0
    var_dt['dt_type'] = 5
    var_dt['dt_dims'] = [-1, 3, 224, 224]

    var_type_dict['dense_tensor'] = var_dt

    tensor = {
        'name': "image",
        'type': var_type_dict,
        'persistable' : False,
        'need_check_feed' : False,
    }
    print(tensor)
    out = TosaArg(tensor)
    print(out.name, out.shape, out.dim_order, out.dtype)
 
