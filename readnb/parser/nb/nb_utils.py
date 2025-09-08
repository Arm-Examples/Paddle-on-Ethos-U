
def get_param(params, name):
    return [p for p in params if p.tensor['name'] == name][0]

def get_param_safe(params, name):
    try:
        return [p for p in params if p.tensor['name'] == name][0]
    except:
        return None

def attr_to_dict(attrs):
    attr_dict = {}
    for attr in attrs:
        attr_dict[attr["name"]] = attr['val']
    return attr_dict

def update_attr(attrs, name, value):
    idx = [idx for idx, attr in enumerate(attrs) if attr['name'] == name]
    if len(idx) > 0:
        idx = idx[0]
        attrs[idx]['val'] = value
    else:
        attrs.append({'name': name, 'val': value})

def find_my_prev_op(op, name):
    for op in op['prev_op']:
        for output in op['outputs']:
            for argu in output['arguments']:
                if argu['name'] == name:
                    return op