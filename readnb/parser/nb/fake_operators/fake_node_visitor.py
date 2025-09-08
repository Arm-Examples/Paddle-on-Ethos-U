from typing import Dict

class FakeNodesVisitor():
    def __init__(self):
        pass

    def shape_infer(self, **kwargs):
        raise NotImplementedError("Fake NodeVisitor must be extended.")

    def infer(self, **kwargs):
        raise NotImplementedError("Fake NodeVisitor must be extended.")

    def expand(self ,**kwargs):
        pass
        # raise NotImplementedError("Fake NodeVisitor must be extended.")

# container for all node visitors
_node_visitor_dicts = {}

def register_fake_nodes(visitor):
    _node_visitor_dicts[visitor.target] = visitor
    return visitor

def get_fake_node_visitors() -> Dict[str, FakeNodesVisitor]:
    node_visitors = {}
    for target, fake_op in _node_visitor_dicts.items():
        node_visitors[target] = fake_op()
    return node_visitors

if __name__ == "__main__":
    visitor = get_fake_node_visitors()
    print(visitor)
   
    conv2d_shape = visitor['conv2d'].shape_infer(
            input_shape=(1, 3, 224, 244),
            filter_shape=(32, 3, 3, 3),
            strides=(2, 2),
            padding=(1, 1),
            dilation=(0, 0)
    )
    print (f"conv2d_shape {conv2d_shape}")
    depthwise_shape = visitor['depthwise_conv2d'].shape_infer(
            input_shape=(1, 32, 112, 112),
            filter_shape=(32, 1, 3, 3),
            strides=(1, 1),
            padding=(1, 1)
    )
    print (f"depthwise_shape {depthwise_shape}")