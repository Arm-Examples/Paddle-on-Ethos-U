import os, struct, copy
import flatbuffers
import numpy as np
import json

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict

from paddle.lite.fbs.proto import ProgramDesc, AttrType, ParamDesc
from paddle.lite.fbs.proto.ParamDesc_ import VariableDesc, DenseTensorDesc
from paddle.lite.fbs.proto.VarType_.Type import Type
from parser.nb.nb_graph_verify import refill_program, renew_tesnors, graph_passes


from .enums import DataLayoutType, PrecisionType, TargetType
from utils.utils import read_buf, write_buf, insert_bytes, replace_bytes

from utils.custom_op import CustomOp

class VarParam:
    def __init__(self, totalsize, offset, tensor):
        self.total_size = totalsize
        self.offset = offset
        self.param_size = self.total_size - self.offset
        self.tensor = tensor

    def __str__(self):
        par_str = ""

        par_str += f"=================== Param: {self.tensor['name']} ===================\n"
        par_str += f"Total Size: {self.total_size}\n"
        par_str += f"Total Size: {self.offset}\n"
        par_str += f"Total Size: {self.param_size}\n"
        par_str += f"  Param:\n"
        par_str += f"    Name: {self.tensor['name']}\n"
        par_str += f"    Name: {self.tensor['name']}\n"
        par_str += f"    Var Type: {self.tensor['var_type']}\n"
        par_str += f"    Lod len: {len(self.tensor['lod'])} lod: {self.tensor['lod']}\n"
        par_str += f"    Dims: {self.tensor['dims']}\n"
        par_str += f"    Data : Type {self.tensor['data_type']}, Len {len(self.tensor['data'])}\n"
        par_str += f"{self.tensor['data']}"
        return par_str

    def __repr__(self):
        return self.__str__

class DenseTensor:
    def __init__(self, argument):
        self.name = argument['name']
        self.ori_name = argument['ori_name']
        self.persistable = argument['persistable']
        self.need_check_feed = argument['need_check_feed']

        self.type = argument['type']['type']
        self.dt_type = argument['type']['dense_tensor']['dt_type']
        self.dt_dims = argument['type']['dense_tensor']['dt_dims']

class Op:
    def __init__(self, type, id, inputs, outputs, attrs, target):
        self.type = type
        self.id = id
        self.attrs = attrs
        self.target = target

        self.inputs = []
        for i in inputs:
            dense_tensors = []
            for argu in i['arguments']:
                if argu != None:
                    dense_tensors.append(DenseTensor(argument=argu))
            ins = {
                "parameter" : i['parameter'],
                'dense_tensors' : dense_tensors,
            }
            self.inputs.append(ins)

        self.outputs = []
        for o in outputs:
            dense_tensors = []
            for argu in o['arguments']:
                if argu != None:
                    dense_tensors.append(DenseTensor(argument=argu))
            outs = {
                "parameter" : o['parameter'],
                'dense_tensors' : dense_tensors,
            }
            self.outputs.append(outs)

    def get_input(self, key):
        if isinstance(key, int):
            return self.inputs[key]
        elif isinstance(key, str):
            for i in self.inputs:
                if i['parameter'] == key:
                    return i
        else :
            print(f"Not a input {key} | {type(key)}")

    def get_output(self, key):
        if isinstance(key, int):
            return self.outputs[key]
        elif isinstance(key, str):
            for i in self.outputs:
                if i['parameter'] == key:
                    return i
        else :
            print(f"Not a input {key} | {type(key)}")

class OpBlock:
    def __init__(self, idx, parent_id, forward_block_idx):
        self.id = idx
        self.parent = parent_id
        self.forwardblockidx = forward_block_idx
        self.vars = []
        self.ops = []
        self.ops_ex = []
        self.dag = []

    def add_variable(self, var_name, var_type:dict, persistable, need_check_feed):
        '''
            name: name of var_name. After renew the opblock, name will be the signal of vars.
            ori_name: the original name from NB model.
                None Buffer Tensor in NB model is use repeatly, have to use original name(ori_name)
                for re-generating the new Tensors.

            var_type_dict is multi-package includes which unpack as following

            var_type_dict[] -> dict:
                dense_tensor:
                    lod_lv
                    dt_type
                    dt_dims
                select_row:
                    row_type
                    row_dims
                tensor_array:
                    lod_lv
                    ta_type
                    ta_dims
                reader->list:
                    [
                        lod
                        dims
                        d_type
                    ]
                tuple_ele_type

        '''
        self.vars.append({
            'name': var_name,
            'ori_name': var_name,
            'type': var_type,
            'persistable': persistable,
            'need_check_feed': need_check_feed,
        })

    def get_variable(self, key):
        if isinstance(key, int):
            return self.vars[key]
        elif isinstance(key, str):
            for var in self.vars:
                if var['name'] == key:
                    return var
        else :
            print(f"Not vailable type of vars {key} | {type(key)}")

        print(f"Not found Variable from NB graph {key}")
        return None

    def add_op(self, op_type, op_id, inputs:list, outputs:list, attrs:list, target):
        self.ops.append({
            'type' : op_type,
            'id' : op_id,
            'next_op': [],
            'prev_op': [],
            'inputs' : inputs,
            'outputs' : outputs,
            'attrs' : attrs,
            'target' : target,
        })

        self.ops_ex.append(Op(op_type, op_id, inputs, outputs, attrs, target))


    def get_op(self, key):
        if isinstance(key, int):
            return self.ops[key]
        elif isinstance(key, str):
            for op in self.ops:
                if op['type'] == key:
                    return op
        else :
            print(f"Not OP type of vars {key} | {type(key)}")

        print(f"Not found OP from NBGraph {key}")
        return None

    def add_param(self):
        pass

    def get_param(self, key):
        pass

    def __str__(self):
        var_str = ""
        for id, v in enumerate(self.vars):
            var_str += f"=================== Variable {id} : {v['name']} ===================\n"
            var_str +=(f"    Name: {v['name']}" + "\n"
                     + f"    Type: {v['type']}" + "\n"
                     + f"    Persistable:   {v['persistable']}" + "\n"
                     + f"    NeedCheckFeed: {v['need_check_feed']}" + "\n")
        op_str = ""
        for id, op in enumerate(self.ops):
            op_str += f"=================== Op {id} : {op['type']} ===================\n"

            op_attr_str = ""
            for attr in op['attrs']:
                op_attr_str += f"            {str(attr['name']):<35}  {str(attr['val'])}\n"

            input_str = ""
            for i, t in enumerate(op['inputs']):
                arg_str = ""
                for arg in t['arguments']:
                    arg_str += str(arg)
                input_str += f"            ({i}/{len(op['inputs'])}) Info: {str(t['parameter']):<30} Arguments: {arg_str}\n"

            output_str = ""
            for i, t in enumerate(op['outputs']):
                arg_str = ""
                for arg in t['arguments']:
                    arg_str += str(arg)
                output_str += f"            ({i}/{len(op['outputs'])}) Info: {str(t['parameter']):<30} Arguments: {arg_str}\n"


            op_str += (f"     Op Type:      {op['type']}" + "\n"
                     + f"        Attribute:     \n{op_attr_str}" + "\n"
                     + f"        Input:        \n{input_str}" + "\n"
                     + f"        Output:        \n{output_str}" + "\n")

        return str(f"  Index: {self.id}" + "\n"
                 + f"  Parent Index: {self.parent}" + "\n"
                 + f"  ForwardBlockIdx  Index: {self.forwardblockidx}" + "\n"
                 + f"{var_str}" + "\n"
                 + f"{op_str}")

    def __repr__(self):
        return self.__str__()

@dataclass
class ProgramStru:
    block_nums: int = 0

    param_header_size: int = 0
    param_size: int = 0
    param_max_tensor_size: int = 0

    blocks: List[OpBlock] = field(default_factory=list)
    params: List[VarParam] = field(default_factory=list)

    def print_prgram(self):
        print(f"Number of Blocks: {len(self.blocks)}")

        for block in self.blocks:
            print(block)

        for param in self.params:
            print(param)

    def to_json(self, indent=2):
        """Convert ProgramStru to a JSON string."""
        def serialize(obj):
            if isinstance(obj, np.ndarray):  # if its ndarray, convert to list.
                return obj.tolist()
            if isinstance(obj, (int, float, str, bool, list, dict)) or obj is None:
                return obj
            if isinstance(obj, (np.integer)):
                return int(obj)
            if isinstance(obj, (np.float)):
                return float(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # ops is a Dag Map which include reference of nextops. which is to large to json.
        for block in self.blocks:
            for op in block.ops:
                for idx, next_op in enumerate(op['next_op']):
                    if False == isinstance(op['next_op'][idx], str):
                        op['next_op'][idx] = f"{op['next_op'][idx]['type']}_{op['next_op'][idx]['id']}"
            for op in block.ops:
                for idx, prev_op in enumerate(op['prev_op']):
                    if False == isinstance(op['prev_op'][idx], str):
                        op['prev_op'][idx] = f"{op['prev_op'][idx]['type']}_{op['prev_op'][idx]['id']}"

        program_dict = {
            "block_nums": self.block_nums,
            "blocks": [
                {
                    "id": block.id,
                    "parent": block.parent,
                    "forward_block_idx": block.forwardblockidx,
                    "vars": block.vars,
                    "ops": block.ops,
                }
                for block in self.blocks
            ],
            "params": [
                {
                    "total_size": param.total_size,
                    "offset": param.offset,
                    "param_size": param.param_size,
                    "tensor": param.tensor,
                }
                for param in self.params
            ],
        }

        # return json.dumps(program_dict, default=serialize)
        return json.dumps(program_dict, indent=indent, default=serialize)

    def check_json(self):
        stra = ""
        for block in self.blocks:
            stra += f"block id: {block.id}, type: {type(block.id)}\n"
            stra += f"block parent: {block.parent}, type: {type(block.parent)}\n"
            stra += f"block forward_block_idx: {block.forwardblockidx}, type: {type(block.forwardblockidx)}\n"
            stra += f"block vars: {block.vars}, type: {type(block.vars)}\n"
            stra += f"block ops: {block.ops}, type: {type(block.ops)}\n"
            stra += "-" * 40 + "\n"
        with open("/home/wangteng/storage/work_project/arm_etos_paddlelite/paddlelite_arm/arm-paddle/Paddle-Lite-Ethos/readnb/dump.log", "w") as f:
            f.write(stra)

    @staticmethod
    def from_json(json_str_or_path):
        """Create a ProgramStru from a JSON string or file path."""
        if os.path.exists(json_str_or_path):  # if a file path
            with open(json_str_or_path, 'r') as f:
                json_str = f.read()
        else:
            raise RuntimeError(f"Not json file {json_str_or_path} exists")
            json_str = json_str_or_path

        data = json.loads(json_str)
        program = ProgramStru(block_nums=data["block_nums"])
        for block_data in data["blocks"]:
            block = OpBlock(
                idx=block_data["id"],
                parent_id=block_data["parent"],
                forward_block_idx=block_data["forward_block_idx"]
            )
            block.vars = block_data["vars"]
            block.ops = block_data["ops"]
            for op in block.ops:
                target = op.get('target', "")
                op_ex = Op(op['type'], op['id'], op['inputs'], op['outputs'], op['attrs'], target)
                #op_ex = Op(op['type'], op['id'], op['inputs'], op['outputs'], op['attrs'], op['target'])
                block.ops_ex.append(op_ex)
            program.blocks.append(block)

        for param_data in data["params"]:
            param = VarParam(
                totalsize=param_data["total_size"],
                offset=param_data["offset"],
                tensor=param_data["tensor"]
            )
            program.params.append(param)
        return program


class NbParser:

    def __init__(self, model_bin):
        if not isinstance(model_bin, bytes):
            raise FileNotFoundError("Init NB graph only support byte binary buffer.")
        self.buf = model_bin


    def parser(self, custom_inputs=None, remove_op_id=-1):
        print("==================== Parse Header ====================")

        cur = 0
        cur, self.meta_version = read_buf(self.buf, cur, 2)
        print("Meta Version:", self.meta_version, cur)

        cur, self.paddle_version = read_buf(self.buf, cur, 16, is_str=True)
        print("Paddle Ver: ", self.paddle_version, cur)


        cur, self.topology_size = read_buf(self.buf, cur, 8)
        print("topology_size : ", self.topology_size)


        print("==================== Parse Program Struct ====================")
        print("Seek to: ", cur)

        self.program = ProgramStru()

        self.programDescBuf = self.buf[cur:cur+self.topology_size]
        cur += self.topology_size

        self.parse_program_desc(self.program, self.programDescBuf)

        self.program.print_prgram()
        # exit()

        print("==================== Parse Program params ====================")
        print("Seek to: ", cur)

        # Header
        cur, self.param_version = read_buf(self.buf, cur, 2)
        print("Param Ver: ", self.param_version)

        cur, self.meta_information = read_buf(self.buf, cur, 2)
        print("meta_information: ", self.meta_information) #reserved

        self.paramsBuf = self.buf[cur:]
        custom_op = CustomOp() # TODO: found a new to insert op into nb

        self.program, pos = self.parse_param_desc(self.program, self.paramsBuf)
        # paramsBuf, pos = self.parse_param_deserializer_buf(paramsBuf)
        print("endpos:",pos, len(self.paramsBuf))

        self.program = graph_passes(self.program, custom_inputs, remove_op_id)
        return self

    def parse_program_desc(self, program, buffer):

        kernel_type = "__@kernel_type_attr@__"

        program_desc = ProgramDesc.ProgramDesc.GetRootAsProgramDesc(buffer, 0)
        program.block_nums = program_desc.BlocksLength()

        opblocks = []
        for i in range(program.block_nums):
            block = program_desc.Blocks(i)
            opblock = OpBlock(block.Idx(), block.ParentIdx(), block.ForwardBlockIdx())

            # get program attrubates
            for j in range(block.VarsLength()):
                var = block.Vars(j)
                var_type = var.Type()
                #  ======  var list ======
                #
                var_type_dict = {}
                var_type_dict['type'] = var_type.Type()
                #
                var_dt = {}
                dt = var_type.DenseTensor()
                var_dt['lod_lv'] = dt.LodLevel()
                var_dt['dt_type'] = dt.Tensor().DataType()
                var_dt['dt_dims'] = dt.Tensor().DimsAsNumpy().tolist()

                var_type_dict['dense_tensor'] = var_dt
                #
                var_row = {}
                row = var_type.SelectedRows()
                if row != None:
                    var_row['row_type'] = row.DataType()
                    var_row['row_dims'] = row.DimsAsNumpy().tolist()
                    var_type_dict['select_row'] = var_row

                var_ta = {}
                ta = var_type.TensorArray()
                if ta != None:
                    var_ta['lod_lv']  = ta.LodLevel()
                    var_ta['ta_type'] = ta.Tensor().DataType()
                    var_ta['ta_dims'] = ta.Tensor().DimsAsNumpy().tolist()
                    var_type_dict['tensor_array'] = var_ta

                reader = var_type.Reader()
                if reader != None:
                    reader_dts = []
                    for id in reader.DenseTensorLength():
                        reader_dt = reader.DenseTensor(id)
                        dt['lod'] = reader_dt.LodLevel()
                        dt['dims'] = reader_dt.Tensor().DimAsNumpy().tolist()
                        dt['d_type'] = reader_dt.Tensor().DataType()
                        reader_dts.append(dt)
                    var_type_dict['reader'] = reader_dts

                tup = var_type.Tuple()
                if tup != None:
                    var_type_dict['tuple_ele_type'] = var_type.Tuple().ElementTypeAsNumpy()
                #  ======  var list ======

                opblock.add_variable(var.Name().decode(), var_type_dict, var.Persistable(), var.NeedCheckFeed())

            for i in range(block.OpsLength()):

                op = block.Ops(i)
                # print(f"--------Operation {i} Type: {op.Type()}--------")
                # get inputs
                inputs = []
                for j in range(op.InputsLength()):
                    input_var = op.Inputs(j)
                    # print(f"  Input Parameter {j}/{op.InputsLength()}: {input_var.Parameter()}")
                    argus = []
                    for k in range(input_var.ArgumentsLength()):
                        # print(f"\tArguments {k}/{input_var.ArgumentsLength()}: {input_var.Arguments(k)}")
                        argus_name = input_var.Arguments(k).decode()
                        # TODO: this extra part is not for NB graph
                        argus_var = opblock.get_variable(argus_name)
                        # argus.append({argus_name: argus_var})
                        argus.append(argus_var)
                    if input_var.Parameter().decode() == "ResidualData" or len(argus) == 0:
                        continue
                    inputs.append({
                        'parameter': input_var.Parameter().decode(),
                        'arguments': argus,
                    })

                # get outputs
                outputs = []
                for j in range(op.OutputsLength()):
                    output_var = op.Outputs(j)
                    # print(f"  Output Parameter: {j}/{op.OutputsLength()} {output_var.Parameter()} Len {output_var.ArgumentsLength()}")
                    argus = []
                    for k in range(output_var.ArgumentsLength()):
                        # print(f"\tArguments {k}/{output_var.ArgumentsLength()}: {output_var.Arguments(k)}")
                        argus_name = output_var.Arguments(k).decode()
                        # TODO: this extra part is not for NB graph
                        argus_var = opblock.get_variable(argus_name)
                        # argus.append({argus_name: argus_var})
                        if op.Type().decode() == "calib":
                            # the calib will make output tensor with a no dim
                            arg = opblock.get_variable(op.Inputs(j).Arguments(0).decode())
                            input_dim = arg['type']['dense_tensor']['dt_dims']
                            argus_var['type']['dense_tensor']['dt_dims'] = input_dim
                        argus.append(argus_var)
                    outputs.append({
                        'parameter': output_var.Parameter().decode(),
                        'arguments': argus,
                    })

                # get attribuate
                opattrs = []
                for k in range(op.AttrsLength()):
                    attr = op.Attrs(k)

                    op_attr = {}
                    attr_name = attr.Name().decode('utf-8')

                    if attr.Type()   == AttrType.AttrType.INT:      attr_val = attr.I()
                    elif attr.Type() == AttrType.AttrType.FLOAT:    attr_val = attr.F()
                    elif attr.Type() == AttrType.AttrType.STRING:   attr_val = attr.S().decode('utf-8')
                    elif attr.Type() == AttrType.AttrType.INTS:     attr_val = attr.IntsAsNumpy().tolist()
                    elif attr.Type() == AttrType.AttrType.FLOATS:   attr_val = attr.FloatsAsNumpy().tolist()
                    elif attr.Type() == AttrType.AttrType.STRINGS:
                        attr_val = ""
                        for id in range(attr.StringsLength()):
                            attr_val += attr.Strings(id).decode()
                    elif attr.Type() == AttrType.AttrType.BOOLEAN:  attr_val = attr.B()
                    elif attr.Type() == AttrType.AttrType.BOOLEANS: attr_val = attr.BoolsAsNumpy().tolist()
                    elif attr.Type() == AttrType.AttrType.BLOCK:    attr_val = attr.BlockIdx
                    elif attr.Type() == AttrType.AttrType.LONG:     attr_val = attr.L()
                    elif attr.Type() == AttrType.AttrType.BLOCKS:   attr_val = attr.BlocksIdxAsNumpy().tolist()
                    elif attr.Type() == AttrType.AttrType.LONGS:    attr_val = attr.LongsAsNumpy().tolist()
                    elif attr.Type() == AttrType.AttrType.FLOAT64:  attr_val = attr.Float64
                    elif attr.Type() == AttrType.AttrType.FLOAT64S: attr_val = attr.Float64sAsNumpy().tolist()

                    # for the operator, type to readable.
                    op_attr['name'] = attr_name
                    op_attr['val']  = attr_val
                    if attr_name == kernel_type:
                        op_type, alias, target, precision, layout = attr_val.split("/")
                        op_attr['val'] = f"{op_type}/{alias}/{TargetType.get_type(int(target))}/{PrecisionType.get_type(int(precision))}/{DataLayoutType.get_type(int(layout))}"
                        opattrs.append({'name': 'op_type', 'val': op_type})
                        opattrs.append({'name': 'alias', 'val': alias})
                        opattrs.append({'name': 'target', 'val': target})
                        opattrs.append({'name': 'precision', 'val': precision})
                        opattrs.append({'name': 'layout', 'val': layout})


                        # op_attr['op_type'] = op_type
                        # op_attr['alias'] = alias
                        # op_attr['target'] = target
                        # op_attr['precesion'] = precision
                        # op_attr['layout'] = layout
                    opattrs.append(op_attr)
                    # print(f"    Attribute Name: {str(attr_name):<30} \t Type: {str(attr.Type())} \t Value: {str(attr_val)}")
                    # print(f"    Attribute : {str(op_attr['name']):<35}  {str(op_attr['val'])}")
                opblock.add_op(op.Type().decode(), i, inputs, outputs, opattrs, op.IsTarget())
            # check_ops(opblock)
            program.blocks.append(opblock)

        return program

    def parse_program_desc_buf(self, buffer):

        kernel_type = "__@kernel_type_attr@__"

        program_desc = ProgramDesc.ProgramDesc.GetRootAsProgramDesc(buffer, 0)
        blocks = program_desc.BlocksLength()
        print(f"Number of Blocks: {blocks}")
        for i in range(blocks):
            block = program_desc.Blocks(i)
            print(f"Block {i} Index: {block.Idx()}")
            print(f"Parent Index: {block.ParentIdx()}")
            print(f"ForwardBlockIdx  Index: {block.ForwardBlockIdx()} \n")

            for j in range(block.VarsLength()):
                var = block.Vars(j)
                print(f"  Variable Name: {var.Name()}")
                print(f"  Variable Type: {var.Type()}")
                print(f"  Persistable: {var.Persistable()}")
                print(f"  NeedCheckFeed: {var.NeedCheckFeed()}")


        for i in range(block.OpsLength()):
            op = block.Ops(i)
            print(f"--------Operation {i} Type: {op.Type()}--------")
            for j in range(op.InputsLength()):
                input_var = op.Inputs(j)
                print(f"  Input Parameter {j}/{op.InputsLength()}: {input_var.Parameter()}")
                for k in range(input_var.ArgumentsLength()):
                    print(f"\tArguments {k}/{input_var.ArgumentsLength()}: {input_var.Arguments(k)}")
            for j in range(op.OutputsLength()):
                output_var = op.Outputs(j)
                print(f"  Output Parameter: {j}/{op.OutputsLength()} {output_var.Parameter()} Len {output_var.ArgumentsLength()}")
                for k in range(output_var.ArgumentsLength()):
                    print(f"\tArguments {k}/{output_var.ArgumentsLength()}: {output_var.Arguments(k)}")

            # get type of op data
            for k in range(op.AttrsLength()):
                attr = op.Attrs(k)
                attr_name = attr.Name().decode('utf-8')
                if attr.Type() == AttrType.AttrType.INT:
                    attr_val = attr.I()
                elif attr.Type() == AttrType.AttrType.FLOAT:
                    attr_val = attr.F()
                elif attr.Type() == AttrType.AttrType.STRING:
                    attr_val = attr.S().decode('utf-8')
                elif attr.Type() == AttrType.AttrType.INTS:
                    attr_val = attr.IntsAsNumpy()
                elif attr.Type() == AttrType.AttrType.FLOATS:
                    attr_val = attr.FloatsAsNumpy()
                elif attr.Type() == AttrType.AttrType.STRINGS:
                    attr_val = attr.Strings()
                elif attr.Type() == AttrType.AttrType.BOOLEAN:
                    attr_val = attr.B()
                elif attr.Type() == AttrType.AttrType.BOOLEANS:
                    attr_val = attr.BoolsAsNumpy()
                elif attr.Type() == AttrType.AttrType.BLOCK:
                    attr_val = attr.BlockIdx
                elif attr.Type() == AttrType.AttrType.LONG:
                    attr_val = attr.L()
                elif attr.Type() == AttrType.AttrType.BLOCKS:
                    attr_val = attr.BlocksIdxAsNumpy()
                elif attr.Type() == AttrType.AttrType.LONGS:
                    attr_val = attr.LongsAsNumpy()
                elif attr.Type() == AttrType.AttrType.FLOAT64:
                    attr_val = attr.Float64
                elif attr.Type() == AttrType.AttrType.FLOAT64S:
                    attr_val = attr.Float64sAsNumpy()

                if attr_name == kernel_type:
                    op_type, alias, target, precision, layout = attr_val.split("/")
                    attr_val = f"{op_type}/{alias}/{TargetType.get_type(int(target))}/{PrecisionType.get_type(int(precision))}/{DataLayoutType.get_type(int(layout))}"

                # print(f"    Attribute Name: {str(attr_name):<30} \t Type: {str(attr.Type())} \t Value: {str(attr_val)}")
                print(f"    Attribute : {str(attr_name):<35}  {str(attr_val)}")

        return program_desc

    def parse_param_desc(self, program, buffer):
        all_size = 0
        cur = 0
        offset = 0
        # Fornward read
        cur, program.param_header_size = read_buf(buffer, cur, 2)
        print("HeadSize Ver: ", program.param_header_size)

        cur, program.param_size = read_buf(buffer, cur, 2)
        print("Param Nums: ", program.param_size)

        cur, program.param_max_tensor_size = read_buf(buffer, cur, 4)
        all_size = cur

        for i in range(program.param_size):
            print("----------------------------------------")
            cur, total_size = read_buf(buffer, cur, 4)
            print("Total Size: ", total_size)
            all_size += total_size

            cur, offset = read_buf(buffer, cur, 4)
            print("Offset: ", offset)

            param_data_size = total_size - offset
            print("Param size: ", param_data_size)
            param_desc_buf =  buffer[cur: cur+param_data_size]
            cur += param_data_size
            tensor = self.read_nb_tensor(param_desc_buf)
            param = VarParam(total_size, offset, tensor)
            program.params.append(param)

        print("Total Size: ", all_size)
        return program, cur


    # def parse_param_deserializer_buf(self, buffer):
    #     all_size = 0
    #     cur = 0
    #     offset = 0
    #     # Fornward read
    #     cur, header_size = read_buf(buffer, cur, 2)
    #     print("HeadSize Ver: ", header_size)

    #     cur, params_size = read_buf(buffer, cur, 2)
    #     print("Param Nums: ", params_size)

    #     buffer , pos = write_buf(buffer, cur-2, 2, params_size+1)
    #     cur, params_size_updated = read_buf(buffer, cur-2, 2)
    #     print("Param Nums updated: ", params_size_updated)

    #     cur, max_tensor_size = read_buf(buffer, cur, 4)
    #     all_size = cur
    #     print(f"Seek to {cur}, Max Tensor Size: {max_tensor_size}")
    #     print("=========================================")
    #     for i in range(params_size):
    #         print("----------------------------------------")
    #         cur, total_size = read_buf(buffer, cur, 4)
    #         print("Total Size: ", total_size)
    #         all_size += total_size

    #         cur, offset = read_buf(buffer, cur, 4)
    #         print("Offset: ", offset)

    #         param_data_size = total_size - offset
    #         print("Param size: ", param_data_size)
    #         param_desc_buf =  buffer[cur: cur+param_data_size]
    #         cur += param_data_size
    #         self.read_nb_tensor(param_desc_buf)

    #     print("Total Size: ", all_size)
    #     return buffer, cur

    def read_nb_tensor(self, buffer):
        tensor = {}
        param_desc = ParamDesc.ParamDesc.GetRootAsParamDesc(buffer, 0)
        print(f"Param : \n    Name {param_desc.Name().decode()}")
        tensor['name'] = param_desc.Name().decode()
        if param_desc.Version() != None:
            tensor['ver'] = param_desc.Version().Version()
            tensor['model_version'] = param_desc.Version().ModelVersion()
            print(f"    Ver {tensor['ver']} Model Ver {tensor['model_version']}")

        variable = param_desc.Variable()
        tensor['var_type'] = param_desc.VariableType()
        print(f"    Var Type: {tensor['var_type']}, {type(variable)}")
        if variable is not None:
            if param_desc.VariableType() == VariableDesc.VariableDesc.DenseTensorDesc:
                dense_tensor_desc = DenseTensorDesc.DenseTensorDesc()
                dense_tensor_desc.Init(variable.Bytes, variable.Pos)

                tensor['lod'] = dense_tensor_desc.LodAsNumpy().tolist()
                tensor['dims'] = dense_tensor_desc.DimAsNumpy().tolist()
                tensor['data_type'] = dense_tensor_desc.DataType()

                # TODO: only for struct test
                array_buf = dense_tensor_desc.DataAsNumpy().tobytes()
                tensor['data'] = np.frombuffer(array_buf, dtype=(self.to_np_type(dense_tensor_desc.DataType()))).tolist()

                print(f"    LOD :lv {dense_tensor_desc.LodLevel()} : {dense_tensor_desc.LodAsNumpy()}")
                print(f"    Dim :len {dense_tensor_desc.DimLength()} : {dense_tensor_desc.DimAsNumpy()}")
                print(f"    Data Type: {dense_tensor_desc.DataType()} data_len: {dense_tensor_desc.DataLength()}")

                # print(f"     Data :\n  {tensor['data']}")
            else:
                tensor['lod'] = None
                tensor['dims'] = None
                tensor['data_type'] = None
                tensor['data'] = None
        return tensor

    def to_np_type(self, data_type):
        if data_type == Type.BOOL:
            raise NotImplementedError("BOOL is not supported in this example")
        elif data_type == Type.INT16:
            return np.int16
        elif data_type == Type.INT32:
            return np.int32
        elif data_type == Type.INT64:
            return np.int64
        elif data_type == Type.FP16:
            raise NotImplementedError("FP16 is not supported in this example")
        elif data_type == Type.FP32:
            return np.float32
        elif data_type == Type.FP64:
            return np.float64
        elif data_type == Type.UINT8:
            return np.uint8
        elif data_type == Type.INT8:
            return np.int8
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

# this funciton for adding a bin to nb model.
def convert_customop_to_nb_param(operator : CustomOp):

    op = operator
    print(len(op.buf), op.type)
    # print(op.buf)
    builder = flatbuffers.Builder(0)
    lod = builder.CreateNumpyVector(np.array([]))
    dim = builder.CreateNumpyVector(np.array(op.dim))
    data = builder.CreateByteVector(op.buf)

    DenseTensorDesc.DenseTensorDescStart(builder=builder)
    DenseTensorDesc.DenseTensorDescAddDim(builder=builder, dim=dim)
    DenseTensorDesc.DenseTensorDescAddDataType(builder=builder, dataType=op.type)
    DenseTensorDesc.DenseTensorDescAddData(builder=builder, data=data)
    dense_tensor_desc = DenseTensorDesc.DenseTensorDescEnd(builder=builder)

    op_name = builder.CreateString(op.name)

    ParamDesc.ParamDescStart(builder=builder)
    ParamDesc.ParamDescAddName(builder=builder, name=op_name)
    ParamDesc.ParamDescAddVariableType(builder=builder, variableType=VariableDesc.VariableDesc.DenseTensorDesc)
    ParamDesc.ParamDescAddVariable(builder=builder, variable=dense_tensor_desc)
    param = ParamDesc.ParamDescEnd(builder=builder)
    builder.Finish(param)
    buf = builder.Output()

    return buf

# for read nb model test.
# if __name__  == "__main__":
#     with open("./test.nb") as m:
#         data = m.read()
#     nb = NbParser(data)
#     nb.parser()

# for add fa vela into nb model
# if __name__  == "__main__":
#     from .test.testdata import testdata
#     data = testdata.data
#     print(len(data))
#     with open("/data1/wangteng/work_project/arm_etos_paddlelite/executorch/abc/add_save/out_vela.bin", "rb+") as f:
#         vela_buf = f.read()
#     print(len(vela_buf), Type.INT8)

#     op = CustomOp()
#     op.buf = vela_buf
#     op.type = Type.INT8

#     # op.buf = data
#     # op.type += Type.FP32
#     op.dim = [1]
#     op.name = "vela"
#     op.buf = struct.pack(f"{len(op.buf)}f", *op.buf)

#     from readnb.parser.graph import ModelLoader, NBGraph
#     NBGraph()
#     param = convert_customop_to_nb_param(op)
#     read_nb_param_desc(param)
