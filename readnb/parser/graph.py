import numpy as np


from utils.utils import read_buf, write_buf
from utils.custom_op import CustomOp

from parser.nb.nb_graph import NbParser
from parser.tosa.tosa_graph import TosaParser

class ModelLoader:

    @classmethod
    def parser(cls, buffer, model_type:str):
        model_type = model_type.lower()
        if model_type != "nb" and model_type != "tosa":
            raise NotImplementedError("Only support model type is NB or TOSA")
        
        if model_type == "nb":
            return NbParser(buffer)
        elif model_type == "tosa":
            return TosaParser(buffer)