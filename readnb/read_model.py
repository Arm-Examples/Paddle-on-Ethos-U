import os,sys

from parser.graph import ModelLoader

if len(sys.argv) < 2:
    raise FileNotFoundError("Need input paddle-lite nb model")

model_path = sys.argv[1]

with open(model_path, "rb") as f:
    data = f.read()

file_size = len(data)
type = os.path.abspath(model_path).split(".")[-1]
print(f"File {model_path}\n type:{type} size: {file_size} bytes")

try:
    print(model_path, type)
    # parser NB
    if type == "nb":
        parser_nb = ModelLoader.parser(data, "nb")
        parser_nb.parser()
        exit()

    # parser TOSA
    elif type == "tosa":
        parser_tosa = ModelLoader.parser(data, "tosa")
        parser_tosa.parser()
        parser_tosa.print_model()
    else:
        raise TypeError("Only support type [nb | tosa]")
except Exception as e:
    print(f"Error: {e}")
