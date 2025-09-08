

class ParamTable:
    def __init__(self):
        self.total_size = 0
        self.offset = 0
        self.param_data_size =0

        self.param_desc_buf = b""

    def  __str__(self):
        return f"Total Size: {self.total_size}\n \
                Offset: {self.offset}\n \
                Param Data Size: {self.param_data_size}\n \
                Param Desc Buf: {self.param_desc_buf}"

    def __repr__(self):
        return self.__str__()

class ParamBlockTable:
    def __init__(self):
        self.header_size = 0
        self.params_size = 0
        self.max_tensor_size =0

        self.params = []

    def  __str__(self):
        return f"Header Size: {self.header_size}\n \
                Params Size: {self.params_size}\n \
                Max Tensor Size: {self.max_tensor_size}\n \
                Params: {self.params}"

    def __repr__(self):
        return self.__str__()

 