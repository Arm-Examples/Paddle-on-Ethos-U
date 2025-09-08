# from parser.nb.nb_graph import ProgramStru
import numpy as np
def print_op_ids(op):
    try:
        __input = [input for input in op['inputs'] if (input['arguments'] != [] and input['arguments'][0]['persistable'] == False)][0]
        print(f"!!!ID test OP {op['type']:<20} input  ",
                f"in  {id(__input)}, ",  \
                f"ar  {id(__input['arguments'][0])}, ", \
                f"ty  {id(__input['arguments'][0]['type'])}, "
                f"de  {id(__input['arguments'][0]['type']['dense_tensor'])}, "
                f"dim {id(__input['arguments'][0]['type']['dense_tensor']['dt_dims'])}. ")
    except:
        pass
    print(f"!!!ID test OP {op['type']:<20} output ",
            f"in  {id(op['outputs'][0])}, ",  \
            f"ar  {id(op['outputs'][0]['arguments'][0])}, ", \
            f"ty  {id(op['outputs'][0]['arguments'][0]['type'])}, "
            f"de  {id(op['outputs'][0]['arguments'][0]['type']['dense_tensor'])}, "
            f"dim {id(op['outputs'][0]['arguments'][0]['type']['dense_tensor']['dt_dims'])}. ")


def compare_programs(program1, program2):
    """Compare two ProgramStru objects in detail and print differences."""

    def compare_dicts(dict1, dict2, path=""):
        """Recursively compare two dictionaries and print differences."""
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        for key in all_keys:
            if key not in dict1:
                print(f"[ERROR] Key '{key}' is missing in the first program at path '{path}'")
            elif key not in dict2:
                print(f"[ERROR] Key '{key}' is missing in the second program at path '{path}'")
            else:
                value1 = dict1[key]
                value2 = dict2[key]
                if isinstance(value1, dict) and isinstance(value2, dict):
                    compare_dicts(value1, value2, path + f".{key}")
                elif isinstance(value1, list) and isinstance(value2, list):
                    compare_lists(value1, value2, path + f".{key}")
                elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                    if not np.array_equal(value1, value2):
                        print(f"[ERROR] Difference at path '{path}.{key}': {value1} != {value2}")
                elif type(value1) != type(value2):
                    print(f"[ERROR] Type mismatch at path '{path}.{key}': {type(value1)} != {type(value2)}")
                elif value1 != value2:
                    print(f"[ERROR] Value mismatch at path '{path}.{key}': value1 != value2")

    def compare_lists(list1, list2, path=""):
        """Compare two lists and print differences."""
        if len(list1) != len(list2):
            print(f"[ERROR] List length mismatch at path '{path}': {len(list1)} != {len(list2)}")
        for i, (item1, item2) in enumerate(zip(list1, list2)):
            if isinstance(item1, dict) and isinstance(item2, dict):
                compare_dicts(item1, item2, path + f"[{i}]")
            elif isinstance(item1, list) and isinstance(item2, list):
                compare_lists(item1, item2, path + f"[{i}]")
            elif isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
                if not np.array_equal(item1, item2):
                    print(f"[ERROR] Difference at path '{path}[{i}]': {item1} != {item2}")
            elif item1 != item2:
                print(f"[ERROR] Value mismatch at path '{path}[{i}]': item1 != item2")

    # Compare top-level attributes
    if program1.block_nums != program2.block_nums:
        print(f"[ERROR] Block nums mismatch: {program1.block_nums} != {program2.block_nums}")
    if program1.param_header_size != program2.param_header_size:
        print(f"[ERROR] Param header size mismatch: {program1.param_header_size} != {program2.param_header_size}")
    if program1.param_size != program2.param_size:
        print(f"[ERROR] Param size mismatch: {program1.param_size} != {program2.param_size}")
    if program1.param_max_tensor_size != program2.param_max_tensor_size:
        print(f"[ERROR] Param max tensor size mismatch: {program1.param_max_tensor_size} != {program2.param_max_tensor_size}")

    # Compare blocks
    compare_lists(program1.blocks, program2.blocks, "blocks")

    # Compare params
    compare_lists(program1.params, program2.params, "params")

    # Compare each block in detail
    for i, (block1, block2) in enumerate(zip(program1.blocks, program2.blocks)):
        print(f"\nComparing Block {i}:")
        if block1.id != block2.id:
            print(f"[ERROR] Block ID mismatch: {block1.id} != {block2.id}")
        if block1.parent != block2.parent:
            print(f"[ERROR] Block parent mismatch: {block1.parent} != {block2.parent}")
        if block1.forwardblockidx != block2.forwardblockidx:
            print(f"[ERROR] Block forward block index mismatch: {block1.forwardblockidx} != {block2.forwardblockidx}")

        # Compare variables in the block
        compare_lists(block1.vars, block2.vars, f"blocks[{i}].vars")

        # Compare operations in the block
        compare_lists(block1.ops, block2.ops, f"blocks[{i}].ops")

    # Compare each param in detail
    for i, (param1, param2) in enumerate(zip(program1.params, program2.params)):
        print(f"\nComparing Param {i}:")
        if param1.total_size != param2.total_size:
            print(f"[ERROR] Total size mismatch: {param1.total_size} != {param2.total_size}")
        if param1.offset != param2.offset:
            print(f"[ERROR] Offset mismatch: {param1.offset} != {param2.offset}")
        if param1.param_size != param2.param_size:
            print(f"[ERROR] Param size mismatch: {param1.param_size} != {param2.param_size}")
        compare_dicts(param1.tensor, param2.tensor, f"params[{i}].tensor")