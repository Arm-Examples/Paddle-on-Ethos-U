from typing import Dict, List

from parser.nb.nb_utils import get_param, attr_to_dict, update_attr


def find_outscale_from_next(op, op_attr, next_id, next_op):

    jump_op_type = [
        'pool2d'
    ]

    calcu_scale_op_type = [
        'sigmoid',
    ]

    calcu_scale_next_op_type = [
        'dropout'
    ]

    next_attrs = attr_to_dict(next_op["attrs"])

    if next_op['type'] in jump_op_type:
        print(f"find_outscale_from_next case: jump_op_type")
        if next_op['next_op'] != [] and next_op['next_op'][0]['type'] == 'calib':
            calib_op = next_op['next_op'][0]
            attr_dict = attr_to_dict(calib_op["attrs"])
            return attr_dict['scale']
        else:
            raise ValueError(f"OP:{op['id']} Not support Cases of {op['type']} in jump_next")

    else:
        if next_op['type'] in calcu_scale_op_type:
            print(f"find_outscale_from_next case: calcu_scale_op_type")
            next_attrs = attr_to_dict(next_op['attrs'])
            if 'out_threshold' not in op_attr.keys():
                return -1.0

            scale = op_attr['out_threshold'] / 127
            #set scale to next sigmoid
            update_attr(next_op['attrs'], "Input0_scale", [scale])
            return scale

        elif next_op['type'] in calcu_scale_next_op_type:
            print(f"find_outscale_from_next case: calcu_scale_next_op_type")
            next_attrs = attr_to_dict(next_op['attrs'])
            scale = next_attrs['out_threshold'] / 127
            # update_attr(next_op['attrs'], "Input0_scale", [scale])
            return scale
        
        # Scale operator has attribute 'bias_after_scale' : True
        elif "Input0_scale" not in next_attrs.keys() and any('_scale' in key and key != 'bias_after_scale' for key in next_attrs.keys()):
            print(f"find_outscale_from_next case: Input0_scale not in next_attrs.keys() _scale in next_attrs.keys()")
            output_name = op["outputs"][0]['arguments'][0]['name']

            next_op_input_para = ""
            for idx, input in enumerate(next_op['inputs']):
                for arug in input['arguments']:
                    if output_name == arug['name']:
                        next_op_input_para = input['parameter']
                        scale_idx = idx
            scale_p = []
            for key, val in next_attrs.items():
                if "_scale" in key:
                    scale_p.append(*val)

            return scale_p[scale_idx]
        elif "Input0_scale" in next_attrs.keys():
            print(f"find_outscale_from_next case: Input0_scale in next_attrs.keys()")
            next_inscale = next_attrs.get("Input0_scale", -1)
            return next_inscale

        elif next_attrs.get("scale", []):
            print(f"find_outscale_from_next case: scale in next_attrs.keys()")
            next_inscale = next_attrs.get("scale", -1)
            if isinstance(next_inscale, list) and len(next_inscale) != 0:
                return next_inscale[0]
            else:
                return next_inscale

        else:
            print(f"find_outscale_from_next case: final special")
            attr_dict = {}
            for attr in op["attrs"]:
                attr_dict[attr["name"]] = attr["val"]

            output_scale = attr_dict.get("Output0_scale", [])
            if len(output_scale) == 0 or \
                any(isinstance(x, list) and len(x) == 0 for x in output_scale if isinstance(x, list)):
                if "out_threshold" in attr_dict.keys():
                    if isinstance(attr_dict["out_threshold"], float):
                        output_scale = attr_dict["out_threshold"] / 127
                    else:
                        raise ValueError(f"OP:{op['id']} Not support Cases of {op['type']} not in jump_next")
                else:
                    #TODO: !!!!!!def to def will not find a useful scale!!!!!!
                    return -1.0
            else:
                return -1.0
                pass
                # raise ValueError(f"OP:{op['id']} Not support Cases of {op['type']} not in jump_next")
            return output_scale
