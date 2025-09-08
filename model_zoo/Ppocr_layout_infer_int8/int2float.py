import numpy as np
import os

dir = "./model_zoo/Ppocr_layout_infer_int8/output/"
l = os.listdir(dir)
output_dir = "./model_zoo/Ppocr_layout_infer_int8/out_float/"

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


with open('./model_zoo/Ppocr_layout_infer_int8/out_float/1x37x100x76.bin', 'rb') as f:
    data1 = np.fromfile(f, dtype=np.float32)
data1 = data1.reshape(1, 100, 76, 37)
data1 = data1.transpose(0,3,1,2)
data1.tofile('model_zoo/Ppocr_layout_infer_int8/layout_postprocess/output/output_0.bin')

with open('./model_zoo/Ppocr_layout_infer_int8/out_float/1x37x50x38.bin', 'rb') as f:
    data2 = np.fromfile(f, dtype=np.float32)
data2 = data2.reshape(1, 50, 38, 37)
data2 = data2.transpose(0,3,1,2)
data2.tofile('model_zoo/Ppocr_layout_infer_int8/layout_postprocess/output/output_1.bin')

with open('./model_zoo/Ppocr_layout_infer_int8/out_float/1x37x25x19.bin', 'rb') as f:
    data3 = np.fromfile(f, dtype=np.float32)
data3 = data3.reshape(1, 25, 19, 37)
data3 = data3.transpose(0,3,1,2)
data3.tofile('model_zoo/Ppocr_layout_infer_int8/layout_postprocess/output/output_2.bin')

with open('./model_zoo/Ppocr_layout_infer_int8/out_float/1x37x13x10.bin', 'rb') as f:
    data4 = np.fromfile(f, dtype=np.float32)
data4 = data4.reshape(1, 13, 10, 37)
data4 = data4.transpose(0,3,1,2)
data4.tofile('model_zoo/Ppocr_layout_infer_int8/layout_postprocess/output/output_3.bin')

print("Saved  4xoutput_x.bin")
