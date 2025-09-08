import numpy as np
import os

dir = "./model_zoo/BlazeFace_infer_int8/output/"
l = os.listdir(dir)
output_dir = "./model_zoo/BlazeFace_infer_int8/out_float/"

for i in l:
    shape = i.split("_")[2]
    scale = i.split("_")[4].split(".bin")[0]
    scale = np.float32(scale)
    data = np.fromfile(dir + i, dtype=np.int8)
    print(shape, data.min(), data.max())
    # print(data)
    data = data.astype(np.float32)
    data = data * scale
    # print(shape, data.min(), data.max())
    data.tofile(output_dir + shape + ".bin")


# read 1x16x16x2.bin
with open('./model_zoo/BlazeFace_infer_int8/out_float/1x16x16x2.bin', 'rb') as f:
    data1 = np.fromfile(f, dtype=np.float32)

data1 = data1.reshape(1, 16, 16, 2)
data1 = data1.reshape(1, -1)

# read 1x8x8x6.bin
with open('./model_zoo/BlazeFace_infer_int8/out_float/1x8x8x6.bin', 'rb') as f:
    data2 = np.fromfile(f, dtype=np.float32)

data2 = data2.reshape(1, 8, 8, 6)
data2 = data2.reshape(1, -1)

concatenated_data = np.concatenate((data1, data2), axis=1)

concatenated_data.tofile('./model_zoo/BlazeFace_infer_int8/out_float/output_1.bin')
print("Saved output_1.bin")


with open('./model_zoo/BlazeFace_infer_int8/out_float/1x16x16x32.bin', 'rb') as f:
    data1 = np.fromfile(f, dtype=np.float32)
data1 = data1.reshape([1, 16, 16, 32])
data1 = data1.reshape([1, 512, 16])

with open('./model_zoo/BlazeFace_infer_int8/out_float/1x8x8x96.bin', 'rb') as f:
    data2 = np.fromfile(f, dtype=np.float32)
data2 = data2.reshape([1, 8, 8, 96])
data2 = data2.reshape([1, 384, 16])

concat_data = np.concatenate([data1, data2], axis=1)

concat_data.tofile('./model_zoo/BlazeFace_infer_int8/out_float/output_0.bin')
print("Saved  output_0.bin")
