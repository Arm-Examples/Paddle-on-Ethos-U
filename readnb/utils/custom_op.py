import numpy

class CustomOp:
    def __init__(self) :
        self.buf = b""
        self.offset = 4
        self.param_size = len(self.buf)
        self.total_size = self.param_size + self.offset
        self.type = 1
        self.name = ""
        self.dim = []
