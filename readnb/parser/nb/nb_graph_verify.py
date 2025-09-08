import math
import numpy as np
import copy
from paddle.lite.fbs.proto.VarType_.Type import Type
from parser.nb.fake_operators.fake_node_visitor import get_fake_node_visitors

from parser.nb.np_graph_temp_job import (
    temp_feed_job,
    temp_fetch_job
)
from parser.nb.nb_utils import get_param
from parser.nb.debug_utils import print_op_ids # only for test.
from parser.nb.nb_dag import NBDAG

def _go_through_fake_op(type, visitor):
    go_through_ops = [
        "calib",
        "softmax",
        "shuffle_channel",
        "scale",
        "hard_sigmoid",
        "relu",
        "hard_swish", # temp for same
        "multiclass_nms3", # temp for same
        "sqrt",
        "arg_max",
        "fusion_elementwise_add_activation",
        "shape", # temp for same
        "slice", # temp for same
        "cast",
        "meshgrid",

        #"matmul_v2", # temp for same
    ]

    go_element_ops = [
        "elementwise_add",
        "elementwise_mul",
        "elementwise_div",
    ]

    if type in go_through_ops:
        return visitor['identity']
    elif type in go_element_ops:
        return visitor['elementwise']
    else:
        try:
            return visitor[type]
        except:
            raise RuntimeWarning(f"Not supportted Op {type}")

def build_op_full_graph(program, isfull=True):
    return program

def build_op_graph(program, dump_dag=True, ):
    for block in program.blocks:
        dag = NBDAG(block.ops)
        block.dag = dag

        # dag.generate_plantuml("test_puml/", max_nodes=7000, is_op_only=False)
        # exit()
    return program

def graph_passes(program, custom_inputs=None, remove_op_id=-1):
    print("!!!!!!!!!!!!!!!! BF renew !!!!!!!!!!!!!!!!!!!")
    program__ = renew_tesnors(program)
    program__ = prune_graph(program__, "", remove_op_id)
    program__ = build_op_graph(program__)
    temp_g_2 = refill_program(program__, custom_inputs)
    print("!!!!!!!!!!!!!!!! Renew fin !!!!!!!!!!!!!!!!!!!")
    return temp_g_2

def prune_graph(program, output_tensor="", temp_use_output_id=-1):
    # Warning: only support 1 block.
    for block in program.blocks:
        ops = block.ops
        if temp_use_output_id > 0:
            fetch = ops[-1]
            if temp_use_output_id - 1 != 0:
                last_op = ops[temp_use_output_id-1]
            fetch["inputs"][0]['arguments'] = last_op["outputs"][0]['arguments']

            idx_to_remove = range(temp_use_output_id, fetch['id'])
            block.ops = [item for item in ops if item['id'] not in list(idx_to_remove)]
            fetch['id'] = temp_use_output_id
        else :
            print("Not prune graph")
    return program

def renew_tesnors(program, output_id=-1):
    '''output id '''
    pairing_op_list = {}

    for block in program.blocks:
        feed_op = block.ops[0]
        feed_ops = [op for op in block.ops if op['type'] == 'feed']
        for feed_op in feed_ops:
            pairing_op_list[feed_op['outputs'][0]['arguments'][0]['ori_name']] = feed_op

        # pairing_op_list[feed_op['outputs'][0]['arguments'][0]['ori_name']] = feed_op
        no_input_ops = [op for op in block.ops if op['inputs'] == []]
        for op in no_input_ops:
            if op['outputs'][0]['arguments'][0]['ori_name'] not in pairing_op_list.keys():
                pairing_op_list[op['outputs'][0]['arguments'][0]['ori_name']] = op


        for idx, op in enumerate(block.ops):
            if op['type'] == "split":
                print(idx)
            # Do pair
            for input_id, input in enumerate(op['inputs']):
                for input_argu_idx, op_arg in enumerate(input['arguments']):
                    try:
                        # for some no resion, input['arguments'] is [] like "ScaleTensor" of op scale
                        input_name = input['arguments'][input_argu_idx]['name']
                        pair_name = input['arguments'][input_argu_idx]['ori_name']
                    except:
                        continue
                    if pair_name in pairing_op_list:
                        outs = pairing_op_list[pair_name]['outputs']
                        # o = [out for out in outs if out['arguments'][0]['ori_name'] == input['arguments'][input_argu_idx]['ori_name']]
                        o_sames = [out for out in outs if any([o_argu for o_argu in out['arguments'] if o_argu['ori_name'] == input['arguments'][input_argu_idx]['ori_name']])]
                        if o_sames != []:
                            if len(o_sames) > 1:
                                raise ValueError(f"OP:{op['id']} {op['type']} find more than one output with same name out {o_sames}")

                            paired_out = [out_argu for out_argu in o_sames[0]['arguments'] if out_argu['ori_name'] == pair_name]
                            if len(paired_out) > 1:
                                raise ValueError(f"OP:{op['id']} {op['type']} find more than one paired_out {paired_out}")

                            op['inputs'][input_id]['arguments'][input_argu_idx] = block.get_variable(paired_out[0]['name'])
                            # op['inputs'][input_id]['arguments'][input_argu_idx] = block.get_variable(o_sames[0]['arguments'][0]['name'])
                            # print(f"Paired:{input}, {o_sames}")


            # Do renew
            for o_id, output in enumerate(op['outputs']):
                for input_argu_idx, op_arg in enumerate(output['arguments']):
                    # try:
                    #     op_arg = output['arguments'][input_argu_idx]
                    # except:
                    #     continue
                    output_name = op_arg['name']
                    pair_name = op_arg['ori_name']
                    if pair_name in pairing_op_list:
                        o = output

                        new_name = f"{output_name}_tmp_{idx}"
                        new_tensor_arg = copy.deepcopy(op_arg['type'])
                        block.add_variable(new_name, new_tensor_arg, op_arg['persistable'], op_arg['need_check_feed'])
                        block.get_variable(new_name)['ori_name'] = op_arg['ori_name']
                        op['outputs'][o_id]['arguments'][input_argu_idx] = block.get_variable(new_name)

                        pairing_op_list[pair_name] = op

                    elif pair_name not in pairing_op_list:
                        pairing_op_list[pair_name] = op
            print_op_ids(op)

    return program


def refill_program(program, custom_inputs=None):
    visitors = get_fake_node_visitors()

    #FIXME: for now, input/output of OPs are not connected between them.
    # which I have save a infer shape to all OPs one by one.
    # need to make all input & output to be same object.
    # inputs is "input 1,3,224,224 input2 1,3,224,224"
    input_shapes = {}
    if custom_inputs != None and isinstance(custom_inputs, str):
        ins = custom_inputs.strip("\"").split(" ")
        for i in range(0, len(ins), 2):
            name = ins[i]
            shape = [int(dim) for dim in ins[i+1].split(",")]
            input_shapes[name] = shape

    for block in program.blocks:
        for idx, op in enumerate(block.ops):
            #FIXME TEMP
            if op['type'] == "feed":
                input_shape = temp_feed_job(op, input_shapes)
                print(f"feed out shape {input_shape}")
                continue
            elif op['type'] == "fetch":
                # temp_fetch_job(op)
                continue

            attr_dict = {}
            for attr in op["attrs"]:
                attr_dict[attr["name"]] = attr['val']

            print_op_ids(op)
            fake_op = _go_through_fake_op(op['type'], visitor=visitors)

            print(f"Processing fake op type {op['type']}")
            if op['type'] == 'slice':
                print(op['type'])
            fake_op.infer(op=op, attrs=attr_dict, params=program.params)
            fake_op.expand(gram=block, op=op)

    # 1. Check fake op data type, all fake o data type should be INT8 (32)
    #    For conv2d, depthwise_conv2d, data type will be changed in tosa operators accroding to params
    for block in program.blocks:
        for idx, op in enumerate(block.ops):
            for sub_out in op['inputs']:
                if sub_out['arguments'][0]['type']['dense_tensor']['dt_type'] != Type.INT8 and \
                    op['type'] != "conv2d" and \
                    op['type'] != "depthwise_conv2d":
                    print(f"fake op [{op['id']}] inputs type {op['type']}, out type {sub_out['arguments'][0]['type']['dense_tensor']['dt_type']}")
            for sub_out in op['outputs']:
                if sub_out['arguments'][0]['type']['dense_tensor']['dt_type'] != Type.INT8  and \
                    op['type'] != "conv2d" and \
                    op['type'] != "depthwise_conv2d":
                    print(f"fake op [{op['id']}] outputs type {op['type']}, out type {sub_out['arguments'][0]['type']['dense_tensor']['dt_type']}")
    return program
