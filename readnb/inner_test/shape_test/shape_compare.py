import sys
import json
import re
import warnings
from parser.nb.nb_graph import ProgramStru
from parser.nb.nb_graph import Op

def shape_log_to_json(file_str:str):
    instructions = []
    with open(file_str, "r") as f:
        data = f.readlines()
        for shape_line in data:
            parts = shape_line.split()
            if "input tensor shapes" not in shape_line:
                instruction = {
                    "id": int(parts[2].strip(",")),
                    "op_type": parts[6].strip(","),
                    "output_tensor_shapes": [],
                    "input_tensor_shapes": [],
                }
            else:
                instruction = {
                    "id": int(parts[2].strip(",")),
                    "op_type": parts[6].strip(","),
                    "output_tensor_shapes": [[int(x) for x in parts[9].strip("shapes:{}").split(",")]],
                    "input_tensor_shapes": [[int(x) for x in parts[12].strip("shapes:{}").split(",")]],
                }
                if instruction["op_type"] == "elementwise_add" or instruction["op_type"] == "elementwise_mul" \
                    or instruction["op_type"] == "concat":
                    instruction["input_tensor_shapes"] = [
                        [int(x) for x in parts[12].strip("shapes:{}").split(",")],
                        [int(x) for x in parts[15].strip("shapes:{}").split(",")],
                    ]
            instructions.append(instruction)
    return instructions

def shape_log_to_json2(file_str:str):
    instructions = []
    with open(file_str, "r") as f:
        data = f.readlines()
    for idx, line in enumerate(data):
        parts = line.strip().split()

        if not parts:
            continue


        instruction_id = int(parts[1])
        op_type = parts[4]
        instruction = {
            "id": instruction_id,
            "op_type": op_type,
            "output_tensor_shapes": [],
            "input_tensor_shapes": []
        }
        if "input_tensor_shapes" not in line:
            instructions.append(instruction)
            continue

        i = 5
        new_case = 0
        while i < len(parts):
            part = parts[i]
            if part.startswith('case_'):
                new_case += 1
                i += 1
            elif part in ['output_tensor_shapes:', 'input_tensor_shapes:']:
                key = part.rstrip(':')
                if i + 1 >= len(parts):
                    i += 1
                    continue
                value_str = parts[i + 1]

                stripped = value_str[1:-1]
                shape_matches = re.findall(r'\{([^}]+)\}', stripped)
                shapes = []
                for match in shape_matches:
                    shape = [int(x.strip()) for x in match.split(',') if x.strip()]
                    shapes.append(shape)


                if key == 'output_tensor_shapes':
                    if new_case == 0:
                        instruction['output_tensor_shapes'].extend(shapes)
                    else:
                        instruction['output_tensor_shapes'] = shapes
                else:
                    if new_case == 0:
                        instruction['input_tensor_shapes'].extend(shapes)
                    else:
                        instruction['input_tensor_shapes'] = shapes

                i += 2
            else:
                i += 1

        instructions.append(instruction)

    return instructions


def infer(bg_file, op_file):
    print(f"DO SHAPE COMPARE {bg_file} {op_file}")

    program = ProgramStru.from_json(f"{op_file}")

    bgjson = shape_log_to_json2(bg_file)
    # opblock = program.blocks[0].ops
    opblock = program.blocks[0].ops_ex
    for id_idx, op in enumerate(opblock[1:-2]):
        op = op
        assert(bgjson[id_idx]['id'] == op.id)
        assert((bgjson[id_idx]['op_type'] == op.type) or (bgjson[id_idx]['op_type'] in op.type))

        # print(f"!!!!{bgjson[id_idx]['id']}, {bgjson[id_idx]['op_type']} | {op.id} {op.type}!!!!")

        bj_input_shapes = []
        for bjin in bgjson[id_idx]['input_tensor_shapes']:
            bj_input_shapes.append(bjin)
            
        bj_output_shapes = []
        for bjin in bgjson[id_idx]['output_tensor_shapes']:
            bj_output_shapes.append(bjin)

        if bj_input_shapes == [] and bj_output_shapes == []:
            # print(f"No need to compare {bgjson[id_idx]['id']}, {bgjson[id_idx]['op_type']} | {op.id} {op.type}")
            continue

        input_shapes = []
        for one in op.inputs:
            presist_tensor = [inp.dt_dims for inp in one['dense_tensors'] if inp.persistable == False]
            if presist_tensor != []:
                input_shapes.extend(presist_tensor)

        assert(len(bj_input_shapes) == len(input_shapes))
        for idx in range(len(bj_input_shapes)):
            if bj_input_shapes[idx] != input_shapes[idx]:
                print(f"[ERR]OP in :id {bgjson[id_idx]['id']}, tp {bgjson[id_idx]['op_type']} | inshape {bj_input_shapes[idx]} not same infer inshape{input_shapes[idx]}")
                # print(f"bg outshape {bj_output_shapes[idx]} not same infer outshape{output_shapes[idx]}")

        output_shapes = []
        for one in op.outputs:
            presist_tensor = [inp.dt_dims for inp in one['dense_tensors'] if inp.persistable == False]
            if presist_tensor != []:
                output_shapes.extend(presist_tensor)

        assert(len(bj_output_shapes) == len(output_shapes))
        for idx in range(len(bj_output_shapes)):
            if bj_output_shapes[idx] != output_shapes[idx]:
                # print(f"bg inshape {bj_input_shapes[idx]} not same infer inshape{input_shapes[idx]}")
                print(f"[ERR]OP out :id {bgjson[id_idx]['id']}, tp {bgjson[id_idx]['op_type']} | bg outshape {bj_output_shapes[idx]} not same infer outshape{output_shapes[idx]}")




