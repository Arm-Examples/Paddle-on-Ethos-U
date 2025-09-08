import numpy as np

np.load("./pdlite_npy/output_tensor_1x37x100x76.npy").tofile("output/output_0.bin")
np.load("./pdlite_npy/output_tensor_1x37x50x38.npy").tofile("output/output_1.bin")
np.load("./pdlite_npy/output_tensor_1x37x25x19.npy").tofile("output/output_2.bin")
np.load("./pdlite_npy/output_tensor_1x37x13x10.npy").tofile("output/output_3.bin")
