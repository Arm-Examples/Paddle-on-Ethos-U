# About Dataset

* The test image [000000423506.jpg](./000000423506.jpg) for this example comes from [COCO 2017](https://cocodataset.org/#detection-2017) dataset `2017 Val images`. It is used for verifying the inference results of BlazeFace model running on Corstone-320 FVP. 
* COCO is a large-scale object detection, segmentation, and captioning dataset. Please refer to [COCO dataset](https://cocodataset.org/#home) for more features. You can use relevant dataset after accepting their [Terms of Use](https://cocodataset.org/#termsofuse). You can also visit [Kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) page to obtain the images. It is licensed by [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


# About Model

>Paddle-lite Model Download from URL [BlazeFace_int8.nb](https://huggingface.co/Alisson-Ason/arm-paddle/tree/main/paddle_lite_models/BlazeFace_int8.nb)

Alternativly, you can convert by following steps

## Model Convert
1. Environment

```bash
conda create -n paddlepaddle python==3.10
pip install paddlepaddle==2.4.2
pip install paddleslim==2.4 
pip install paddledet==2.4
pip install numpy==1.23.5
```

## Model Quantize


1. Prepare ImageNet datasets

2. Prepare model
    * Clone [Blazeface](https://github.com/hollance/BlazeFace-PyTorch)
    ```bash
    git clone https://github.com/hollance/BlazeFace-PyTorch

    # Alternativly, you can use this as helper for exporting to paddlepaddle model
    git clone https://github.com/yytsweet/BlazeFace-PyTorch-Paddle.git
    cd BlazeFace-PyTorch-Paddle
    python export.py
    ```

3. Clone [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

4. Do Quantization
    * Under example of picodet.
        ```bash
        cd PaddleSlim/demo
        PYTHONPATH=./ python quant/quant_post/quant_post.py --model_path exported_model_fixed --save_path ./blazeface --use_gpu false --input_name x --algo='avg'
        ```

5. Conver PaddleModel to nb
    ```bash
    paddle_lite_opt --model_dir ./blazeface --optimize_out_type=naive_buffer --optimize_out ./BlazeFace
    ```

6. Conver NB
    ```bash
    cd readnb
    python write_model.py
    --model_path ./BlazeFace.nb
    --out_dir ./BlazeFace
    ```


>This will create middle model JSON. The difference between this JSON and final one, please according to readnb/test_asset/blazeface/blazeface.json
