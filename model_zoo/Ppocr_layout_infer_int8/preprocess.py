import torch
import numpy as np
import argparse
from PIL import Image


def preprocess(image_path):
    image = Image.open(image_path)
    image = image.resize((608, 800))
    image = np.array(image).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image -= 0.5
    image /= 0.5

    # x = np.fromfile("input.bin", dtype=np.float32).reshape(1, 3, 800, 608)
    # print(x.reshape(-1)[:10])
    # print(image.reshape(-1)[:10])
    # print(np.allclose(x, image, atol=1e-5, rtol=1))
    # x = torch.from_numpy(x)
    # print(x.min(), x.max())
    x = torch.from_numpy(image)

    conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    conv1.weight.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/w1.npy"))
    conv1.bias.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/b1.npy"))

    conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16)
    conv2.weight.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/w2.npy"))
    conv2.bias.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/b2.npy"))

    conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
    conv3.weight.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/w3.npy"))
    conv3.bias.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/b3.npy"))

    conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
    conv4.weight.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/w4.npy"))
    conv4.bias.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/b4.npy"))

    conv5 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
    conv5.weight.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/w5.npy"))
    conv5.bias.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/b5.npy"))

    conv6 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
    conv6.weight.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/w6.npy"))
    conv6.bias.data = torch.from_numpy(np.load("./model_zoo/Ppocr_layout_infer_int8/weights/b6.npy"))

    x = torch.nn.functional.hardswish(conv1(x))
    x = torch.nn.functional.hardswish(conv2(x))
    x = torch.nn.functional.hardswish(conv3(x))
    x = torch.nn.functional.hardswish(conv4(x))
    x = torch.nn.functional.hardswish(conv5(x))
    x = torch.nn.functional.hardswish(conv6(x))

    x /= 0.3498304486274719
    # print(x.min(), x.max())
    x = x.to(torch.int8)

    return x.detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    image = preprocess(args.image)
    np.save(args.output, image)
    # print("Preprocess done!")
