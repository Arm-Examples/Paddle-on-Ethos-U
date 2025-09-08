# About Dataset

The test image [layout.jpg](./layout.jpg) for this example is manually generated using LaTeX, and the copyright and license is consistent with the current project Arm Limited. It is used for testing the inference results of PPOcr_layout Demo running on Corstone-320 FVP.


# About Model

>Paddle-lite Model Download from [PicoDet_layout_1x_infer_int8.nb](https://huggingface.co/Alisson-Ason/arm-paddle/tree/main/paddle_lite_models/PicoDet_layout_1x_infer_int8.nb)

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
1. Prepare COCO datasets(this default to use COCO dataset. If use other datasets, please infer to [PaddleDataset Prepare](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md))

2. Prepare [PaddleModel](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md)

3. Clone [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

4. Do Quantization
    * Under example of picodet.
        ```bash
        cd example/full_quantization/picodet
        ```

    * Modify `./configs/picodet_reader.yml`
        ```yaml
        eval_height: &eval_height 800
        eval_width: &eval_width 608
        ```

    * Modify `./configs/picodet_npu_with_postprocess.yaml`
        ```yaml
        model_dir: ./PicoDet_layout_1x_infer
        model_filename: inference.pdmodel
        params_filename: inference.pdiparams
        ```
    * Run Quantization
        ```bash
        python post_quant.py --config_path ./configs/picodet_npu_with_postprocess.yaml --save_dir ./output
        ```

5. Conver PaddleModel to nb
    ```bash
    paddle_lite_opt --model_dir ./model --optimize_out_type=naive_buffer --optimize_out ./PicoDet_layout_1x_infer_optimized
    ```

6. Conver NB
    ```bash
    cd readnb
    python write_model.py
    --model_path ./PicoDet_layout_1x_infer_optimized.nb
    --out_dir ./PicoDet_layout_1x_infer_optimized
    ```
>This will create middle model JSON. The difference between this JSON and final one, please according to readnb/test_asset/ppocr_layout/ppocr_layout.json
