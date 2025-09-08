import flatbuffers
from pathlib import Path
import numpy as np


import tosa.TosaGraph as TosaGraph
import tosa.Version as Version
import tosa.TosaRegion as TosaRegion
import tosa.TosaBasicBlock as TosaBasicBlock
import tosa.TosaOperator as TosaOperator
import tosa.TosaTensor as TosaTensor
import tosa.DType as DType
import tosa.Op as Op

def int_to_op_type(num):
    optypes = {
        0:  ("UNKNOWN", Op.Op.UNKNOWN),
        1:  ("ARGMAX", Op.Op.ARGMAX),
        2:  ("AVG_POOL2D", Op.Op.AVG_POOL2D),
        3:  ("CONV2D", Op.Op.CONV2D),
        4:  ("CONV3D", Op.Op.CONV3D),
        5:  ("DEPTHWISE_CONV2D", Op.Op.DEPTHWISE_CONV2D),
        6:  ("FULLY_CONNECTED", Op.Op.FULLY_CONNECTED),
        7:  ("MATMUL", Op.Op.MATMUL),
        8:  ("MAX_POOL2D", Op.Op.MAX_POOL2D),
        9:  ("TRANSPOSE_CONV2D", Op.Op.TRANSPOSE_CONV2D),
        10: ("CLAMP", Op.Op.CLAMP),
        11: ("RESERVED", Op.Op.RESERVED),
        12: ("SIGMOID", Op.Op.SIGMOID),
        13: ("TANH", Op.Op.TANH),
        14: ("ADD", Op.Op.ADD),
        15: ("ARITHMETIC_RIGHT_SHIFT", Op.Op.ARITHMETIC_RIGHT_SHIFT),
        16: ("BITWISE_AND", Op.Op.BITWISE_AND),
        17: ("BITWISE_OR", Op.Op.BITWISE_OR),
        18: ("BITWISE_XOR", Op.Op.BITWISE_XOR),
        19: ("INTDIV", Op.Op.INTDIV),
        20: ("LOGICAL_AND", Op.Op.LOGICAL_AND),
        21: ("LOGICAL_LEFT_SHIFT", Op.Op.LOGICAL_LEFT_SHIFT),
        22: ("LOGICAL_RIGHT_SHIFT", Op.Op.LOGICAL_RIGHT_SHIFT),
        23: ("LOGICAL_OR", Op.Op.LOGICAL_OR),
        24: ("LOGICAL_XOR", Op.Op.LOGICAL_XOR),
        25: ("MAXIMUM", Op.Op.MAXIMUM),
        26: ("MINIMUM", Op.Op.MINIMUM),
        27: ("MUL", Op.Op.MUL),
        28: ("POW", Op.Op.POW),
        29: ("SUB", Op.Op.SUB),
        30: ("TABLE", Op.Op.TABLE),
        31: ("ABS", Op.Op.ABS),
        32: ("BITWISE_NOT", Op.Op.BITWISE_NOT),
        33: ("CEIL", Op.Op.CEIL),
        34: ("CLZ", Op.Op.CLZ),
        35: ("EXP", Op.Op.EXP),
        36: ("FLOOR", Op.Op.FLOOR),
        37: ("LOG", Op.Op.LOG),
        38: ("LOGICAL_NOT", Op.Op.LOGICAL_NOT),
        39: ("NEGATE", Op.Op.NEGATE),
        40: ("RECIPROCAL", Op.Op.RECIPROCAL),
        41: ("RSQRT", Op.Op.RSQRT),
        42: ("SELECT", Op.Op.SELECT),
        43: ("EQUAL", Op.Op.EQUAL),
        44: ("GREATER", Op.Op.GREATER),
        45: ("GREATER_EQUAL", Op.Op.GREATER_EQUAL),
        46: ("REDUCE_ANY", Op.Op.REDUCE_ANY),
        47: ("REDUCE_ALL", Op.Op.REDUCE_ALL),
        48: ("REDUCE_MAX", Op.Op.REDUCE_MAX),
        49: ("REDUCE_MIN", Op.Op.REDUCE_MIN),
        50: ("REDUCE_PRODUCT", Op.Op.REDUCE_PRODUCT),
        51: ("REDUCE_SUM", Op.Op.REDUCE_SUM),
        52: ("CONCAT", Op.Op.CONCAT),
        53: ("PAD", Op.Op.PAD),
        54: ("RESHAPE", Op.Op.RESHAPE),
        55: ("REVERSE", Op.Op.REVERSE),
        56: ("SLICE", Op.Op.SLICE),
        57: ("TILE", Op.Op.TILE),
        58: ("TRANSPOSE", Op.Op.TRANSPOSE),
        59: ("GATHER", Op.Op.GATHER),
        60: ("SCATTER", Op.Op.SCATTER),
        61: ("RESIZE", Op.Op.RESIZE),
        62: ("CAST", Op.Op.CAST),
        63: ("RESCALE", Op.Op.RESCALE),
        64: ("CONST", Op.Op.CONST),
        65: ("IDENTITY", Op.Op.IDENTITY),
        66: ("CUSTOM", Op.Op.CUSTOM),
        67: ("COND_IF", Op.Op.COND_IF),
        68: ("WHILE_LOOP", Op.Op.WHILE_LOOP),
        69: ("FFT2D", Op.Op.FFT2D),
        70: ("RFFT2D", Op.Op.RFFT2D),
        71: ("ERF", Op.Op.ERF),
        72: ("DIM", Op.Op.DIM)
    }
    return optypes.get(num, Op.Op.UNKNOWN)

def int_to_dtype(num):
    dtypes = {
        0:  ("UNKNOWN", DType.DType.UNKNOWN),
        1:  ("FLOAT32", DType.DType.FLOAT32),
        2:  ("FLOAT16", DType.DType.FLOAT16),
        3:  ("INT32", DType.DType.INT32),
        4:  ("INT16", DType.DType.INT16),
        5:  ("INT8", DType.DType.INT8),
        6:  ("UINT8", DType.DType.UINT8),
        7:  ("UINT16", DType.DType.UINT16),
        8:  ("UINT32", DType.DType.UINT32),
        9:  ("BOOL", DType.DType.BOOL)
    }
    return dtypes.get(num, DType.DType.UNKNOWN)



class TosaData:
    def __init__(self):
        self.version = None
        self.regions = []

    def set_version(self, major, minor, patch, draft):
        self.version = {
            "major": major,
            "minor": minor,
            "patch": patch,
            "draft": draft
        }

    def add_region(self, region_name, blocks):
        self.regions.append({
            "name": region_name,
            "blocks": blocks
        })

    def to_dict(self):
        return {
            "version": self.version,
            "regions": self.regions
        }

class TosaParser:
    def __init__(self, model_bin):
        self.data = TosaData()
        self.graph = TosaGraph.TosaGraph.GetRootAsTosaGraph(model_bin, 0)

    def parser(self, custom_inputs=None, remove_op_id=-1):
        if not self.graph:
            raise ValueError("Graph is not loaded. Call load() first.")

        # Parse Version
        version = self.graph.Version()
        self.data.set_version(version._Major(), version._Minor(), version._Patch(), version._Draft())

        # Parse Regions
        for i in range(self.graph.RegionsLength()):
            region = self.graph.Regions(i)
            region_name = region.Name().decode('utf-8')
            blocks = []

            # Parse Blocks in Region
            for j in range(region.BlocksLength()):
                block = region.Blocks(j)
                block_name = block.Name().decode('utf-8')
                operators = []
                tensors = []
                inputs = []
                outputs = []

                for k in range(block.InputsLength()):
                    input = block.Inputs(k)
                    inputs.append(input)
                for k in range(block.OutputsLength()):
                    output = block.Outputs(k)
                    outputs.append(output)
                # Parse Operators in Block
                for k in range(block.OperatorsLength()):
                    operator = block.Operators(k)
                    operators.append({
                        "op": operator.Op(),
                        "attribute": self.parse_attribute(operator.AttributeType(), operator.Attribute()),
                        "inputs": [operator.Inputs(l).decode('utf-8') for l in range(operator.InputsLength())],
                        "outputs": [operator.Outputs(m).decode('utf-8') for m in range(operator.OutputsLength())]
                    })

                # Parse Tensors in Block
                for l in range(block.TensorsLength()):
                    tensor = block.Tensors(l)

                    t_data = tensor.DataAsNumpy().tobytes() if tensor.DataLength() > 0 else None
                    if tensor.DataLength() > 0:
                        # print("!!!!!!!!!!!", type(t_data), tensor.Type(), tensor.DataLength())
                        t_data = np.frombuffer(t_data, dtype=(self.to_np_type(tensor.Type())))

                    tensors.append({
                        "name": tensor.Name().decode('utf-8'),
                        "shape": [tensor.Shape(m) for m in range(tensor.ShapeLength())],
                        "type": tensor.Type(),
                        "variable": tensor.Variable(),
                        "is_unranked": tensor.IsUnranked(),
                        "data": t_data
                    })

                blocks.append({
                    "name": block_name,
                    "inputs": inputs,
                    "outputs": output,
                    "operators": operators,
                    "tensors": tensors
                })

            self.data.add_region(region_name, blocks)

    def parse_attribute(self, attr_type, attr):
        # Implement attribute parsing logic if necessary
        return {
            "type": attr_type,
            "data": attr
        }

    def get_data(self):
        return self.data.to_dict()

    def to_np_type(self, data_type):
        if data_type == DType.DType.BOOL:
            raise NotImplementedError("BOOL is not supported in this example")
        elif data_type == DType.DType.INT16:
            return np.int16
        elif data_type == DType.DType.INT32:
            return np.int32
        elif data_type == DType.DType.FP16:
            raise NotImplementedError("FP16 is not supported in this example")
        elif data_type == DType.DType.FP32:
            return np.float32
        elif data_type == DType.DType.UINT8:
            return np.uint8
        elif data_type == DType.DType.INT8:
            return np.int8
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def print_model(self):
        print(f"Version: {self.data.version}")
        for region in self.data.regions:
            print(f"Region: {region['name']}")
            for block in region['blocks']:
                print(f"  Block: {block['name']}")
                for input in block['inputs']:
                    print(f"    Input: {input}")
                for output in block['outputs']:
                    print(f"    Output: {output}")
                for operator in block['operators']:
                    print(f"    Operator: {int_to_op_type(operator['op'])[0]}, Inputs: {operator['inputs']}, Outputs: {operator['outputs']}, Attribute: {operator['attribute']}")
                for tensor in block['tensors']:
                    print(f"    Tensor: {tensor['name']}, Shape: {tensor['shape']}, Type: {tensor['type']}, Variable: {tensor['variable']}, Is Unranked: {tensor['is_unranked']}, Data: \n    {tensor['data']}")

if __name__ == "__main__":
    model_path = "/data1/wangteng/work_project/arm_etos_paddlelite/executorch/abc/save/out.tosa"
    # Replace `example.tosa` with the path to your FlatBuffer binary file
    with open(model_path, 'rb') as f:
        data = f.read()
    try:
        reader = TosaFlatBufferReader(data)
        reader.parse()
        reader.print_model()
    except Exception as e:
        print(f"Error: {e}")
