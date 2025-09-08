# About Dataset

The test image [posedet_demo.jpg](./posedet_demo.jpg) for this example comes from pose_detection demo image of [posedet_demo.jpg in Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/develop/pose_detection/assets/images/posedet_demo.jpg). It is used for testing the inference results of PP_TinyPose Demo running on Corstone-320 FVP. Please refer to [Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/develop/pose_detection) for more features. License: [Apache-2.0 license](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop?tab=Apache-2.0-1-ov-file#readme)

# About Model

>Paddle-lite Model Download from [PP_TinyPose_128x96_qat_dis_nopact_opt.nb](https://huggingface.co/Alisson-Ason/arm-paddle/tree/main/paddle_lite_models/PP_TinyPose_128x96_qat_dis_nopact_opt.nb)

Alternativly, you can convert by following steps

1. Download PDModel [PP_TinyPose](https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/models/PP_TinyPose_128x96_qat_dis_nopact.tgz)

2. Conver PaddleModel to nb
    ```bash
    paddle_lite_opt --model_dir ./PP_TinyPose_128x96_qat_dis_nopact/ --optimize_out ./PP_TinyPose_128x96_qat_dis_nopact_opt
    ```

3. Conver NB
    ```bash
    cd readnb
    python write_model.py
    --model_path ./PP_TinyPose_128x96_qat_dis_nopact_opt.nb
    --out_dir ./PP_TinyPose_128x96_qat_dis_nopact_opt
    ```
>This will create middle model JSON. The difference between this JSON and final one, please according to readnb/test_asset/tinypose/
