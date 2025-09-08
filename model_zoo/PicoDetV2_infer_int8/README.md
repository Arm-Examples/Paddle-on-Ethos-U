# About Dataset

The test image [demo.jpg](./demo.jpg) for this example comes from detection demo image from [YOLOv4-Pytorch](https://github.com/gwinndr/YOLOv4-Pytorch). It is used for testing the inference results of PicoDetV2 Demo running on Corstone-320 FVP. Please refer to [YOLOv4-Pytorch](https://github.com/gwinndr/YOLOv4-Pytorch) for more features. License: [MIT license](https://github.com/gwinndr/YOLOv4-Pytorch?tab=MIT-1-ov-file#readme)


# About Model

>Paddle-lite Model Download from [picodetv2_relu6_coco_no_fuse.nb](https://huggingface.co/Alisson-Ason/arm-paddle/tree/main/paddle_lite_models/picodetv2_relu6_coco_no_fuse.nb
)

Alternativly, you can convert by following steps

1. Download PDModel [PicoDetV2](https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/models/picodetv2_relu6_coco_no_fuse.tar.gz)

2. Conver PaddleModel to nb
    ```bash
    paddle_lite_opt --model_dir ./picodetv2_relu6_coco_no_fuse --optimize_out ./picodetv2_relu6_coco_no_fuse
    ```

3. Conver NB
    ```bash
    cd readnb
    python write_model.py
    --model_path ./picodetv2_relu6_coco_no_fuse.nb
    --out_dir ./picodetv2_relu6_coco_no_fuse
    ```
>This will create middle model JSON. The difference between this JSON and final one, please according to readnb/test_asset/picodet/picodet.json
