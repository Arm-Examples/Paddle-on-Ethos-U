from enum import Enum

class DataLayoutType:
    kUnk = 0
    kNCHW = 1
    kNHWC = 3
    kImageDefault = 4
    kImageFolder = 5
    kImageNW = 6
    kAny = 2
    kMetalTexture2DArray = 7
    kMetalTexture2D = 8
    NUM = 9

    # create dict from a value to name
    _type_d = {value: name for name, value in vars().items() if isinstance(value, int)}
    _type_reverse_d = {name: value for name, value in vars().items() if isinstance(value, int)}

    @classmethod
    def get_type(cls, value):
        return cls._type_d.get(value)

    @classmethod
    def get_value(cls, type):
        return cls._type_reverse_d.get(type)

class TargetType:
    kUnk = 0
    kHost = 1
    kX86 = 2
    kCUDA = 3
    kARM = 4
    kOpenCL = 5
    kAny = 6
    kFPGA = 7
    kNPU = 8
    kXPU = 9
    kBM = 10
    kMLU = 11
    kRKNPU = 12
    kAPU = 13
    kHuaweiAscendNPU = 14
    kImaginationNNA = 15
    kIntelFPGA = 16
    kMetal = 17
    kNNAdapter = 18
    NUM = 19

    # create dict from a value to name
    _type_d = {value: name for name, value in vars().items() if isinstance(value, int)}
    _type_reverse_d = {name: value for name, value in vars().items() if isinstance(value, int)}

    @classmethod
    def get_type(cls, value):
        return cls._type_d.get(value)

    @classmethod
    def get_value(cls, type):
        return cls._type_reverse_d.get(type)


class PrecisionType:
    kUnk = 0
    kFloat = 1
    kInt8 = 2
    kInt32 = 3
    kAny = 4
    kFP16 = 5
    kBool = 6
    kInt64 = 7
    kInt16 = 8
    kUInt8 = 9
    kFP64 = 10
    NUM = 11

    # create dict from a value to name
    _type_d = {value: name for name, value in vars().items() if isinstance(value, int)}
    _type_reverse_d = {name: value for name, value in vars().items() if isinstance(value, int)}

    @classmethod
    def get_type(cls, value):
        return cls._type_d.get(value)

    @classmethod
    def get_value(cls, type):
        return cls._type_reverse_d.get(type)
