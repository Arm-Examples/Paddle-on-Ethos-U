#!/usr/bin/env bash
set -e

usage() {
    echo "Usage $0 [options] [arguments]"
    echo "Options:"
    echo "  -h, --help            Show this help message and exit."
    echo "  -m, --model <name>    Specify the model type."
    echo "  -p, --pic <file>      Specify the path to the image."
    echo "  -c, --confidence <score>    Specify the threshold of confidences."
    echo "Model types:"
    echo "  mv1                   MobileNetV1"
    echo "  pplcnetv2             PPLCNetV2"
    echo "  tinypose              PP_TinyPose"
    echo "  picodetv2             PICODetV2"
    echo "  blazeface             BlazeFace"
    echo "  ppocr_det             PPOCR_det"
    echo "  humansegv2            Humansegv2"
    echo "  ppocr_rec             PPOCR_rec"
    echo "  ppocr_layout          PP_OCR_layout"
    echo "Example:"
    echo "  $0 --model=mv1 --pic=./data/1.jpg"
    exit 1
}

while getopts ":hm:p:c:-:" opt; do
    case $opt in
        h)
            usage
            ;;
        m)
            MODEL_TYPE="$OPTARG"
            ;;
        p)
            PIC_PATH="$OPTARG"
            ;;
        c)
            CONFIDENCE_THRESHOLD="$OPTARG"
            ;;
        -)
            LONG_OPTARG="${OPTARG#*=}"
            case "${OPTARG}" in
                help)
                    usage
                    ;;
                model)
                    if [[ -z "${!OPTIND}" || "${!OPTIND}" == -* ]]; then
                        echo "ERROR: --$OPTARG requires an argument" >&2
                        usage
                    fi
                    MODEL_TYPE="${!OPTIND}"; OPTIND=$((OPTIND + 1))
                    ;;
                model=*)
                    MODEL_TYPE="$LONG_OPTARG"
                    ;;
                pic)
                    if [[ -z "${!OPTIND}" || "${!OPTIND}" == -* ]]; then
                        echo "ERROR: --$OPTARG requires an argument" >&2
                        usage
                    fi
                    PIC_PATH="${!OPTIND}"; OPTIND=$((OPTIND + 1))
                    ;;
                pic=*)
                    PIC_PATH="$LONG_OPTARG"
                    ;;
                confidence)
                    if [[ -z "${!OPTIND}" || "${!OPTIND}" == -* ]]; then
                        echo "ERROR: --$OPTARG requires an argument" >&2
                        usage
                    fi
                    CONFIDENCE_THRESHOLD="${!OPTIND}"; OPTIND=$((OPTIND + 1))
                    ;;
                confidence=*)
                    CONFIDENCE_THRESHOLD="$LONG_OPTARG"
                    ;;
                *)
                    echo "ERROR: Invalid Option --${OPTARG}" >&2
                    usage
                    ;;
            esac
            ;;
        \?)
            echo "Invalid Option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

if [ -z "$MODEL_TYPE" ]; then
    echo "ERROR: Model Type Required (-m/--model)" >&2
    usage
fi

if [ -z "$PIC_PATH" ]; then
    echo "ERROR: Input Pic Required (-p/--pic)" >&2
    usage
fi

if [ ! -f "$PIC_PATH" ]; then
    echo "ERROR: Picture file does not exist: $PIC_PATH" >&2
    exit 1
fi
echo "Using model Type: $MODEL_TYPE"
echo "Using Input File: $PIC_PATH"

mv1_preprocess() {
    echo "Running preprocess steps for MobileNetV1..."
    # Convert model to generate vela.bin
    pushd readnb
    echo "Build vela"
    python write_model.py --model_path ../model_zoo/MobileNetV1_infer_int8/mobilenetv1_int8_opt.nb --out_dir ../model_zoo/MobileNetV1_infer_int8/ --remove_op_id 32 --do_vela
    popd
    cp -rf model_zoo/MobileNetV1_infer_int8/out_vela.bin verify/vela.bin
    # preprocess
    python model_zoo/MobileNetV1_infer_int8/mv1_preprocess_load.py $PIC_PATH
}

pplcnetv2_preprocess() {
    echo "Running preprocess steps for PPLCNetV2..."
    pushd readnb
    # Convert model to generate vela.bin
    echo "Build vela"
    python write_model.py --model_path ../model_zoo/PPLCNetV2_infer_int8/PPLCNetV2_base_infer_int8_opt.nb --out_dir ../model_zoo/PPLCNetV2_infer_int8/ --remove_op_id 85 --do_vela
    popd
    cp -rf model_zoo/PPLCNetV2_infer_int8/out_vela.bin verify/vela.bin
    # preprocess
    python model_zoo/PPLCNetV2_infer_int8/pplc_preprocess_load.py $PIC_PATH
}

pp_tinypose_preprocess() {
    echo "Running preprocess steps for PP-TinyPose..."
    # generate input_tensor.bin and copy the vela.bin from model zoo
    pushd readnb
    python write_model.py --model_path ../download_nb/PP_TinyPose_128x96_qat_dis_nopact_opt.nb --out_dir ../model_zoo/PP_TinyPose_128x96_qat_dis_nopact_opt/

    python write_model.py --model_path ./test_asset/tinypose/tinypose_part1_1.json --out_dir ../download_nb --do_vela
    cp ../download_nb/out_vela.bin ../download_nb/vela_part1_1.bin
    python write_model.py --model_path ./test_asset/tinypose/tinypose_part1_2.json --out_dir ../download_nb --do_vela
    cp ../download_nb/out_vela.bin ../download_nb/vela_part1_2.bin
    python write_model.py --model_path ./test_asset/tinypose/tinypose_part1_3.json --out_dir ../download_nb --do_vela
    cp ../download_nb/out_vela.bin ../download_nb/vela_part1_3.bin
    python write_model.py --model_path ./test_asset/tinypose/tinypose_part2.json --out_dir ../download_nb --do_vela
    cp ../download_nb/out_vela.bin ../download_nb/vela_part2.bin
    popd

    cp -rf download_nb/vela_part1_1.bin verify/vela.bin
    cp -rf download_nb/vela_part1_2.bin verify/vela2.bin
    cp -rf download_nb/vela_part1_3.bin verify/vela3.bin
    cp -rf download_nb/vela_part2.bin verify/vela4.bin
    python model_zoo/PP_TinyPose_128x96_qat_dis_nopact_opt/tinypose_preprocess_load.py $PIC_PATH
}

picodetv2_preprocess() {
    echo "Running preprocess steps for PICODetV2..."
    # Convert model to generate vela.bin
    pushd readnb
    echo "Build vela"
    if [ ! -e "../model_zoo/PicoDetV2_infer_int8/g_picodetv2_relu6_coco_no_fuse.json" ]; then
        echo "Please put generated model json file 'g_picodetv2_relu6_coco_no_fuse.json' into 'model_zoo/PicoDetV2_infer_int8' directory"
        echo "Then 'patch -p0 model_zoo/PicoDetV2_infer_int8/g_picodetv2_relu6_coco_no_fuse.json < readnb/test_asset/picodet/g_picodetv2.patch'"
        exit
    fi
    python write_model.py --model_path ../model_zoo/PicoDetV2_infer_int8/g_picodetv2_relu6_coco_no_fuse.json --out_dir ../model_zoo/PicoDetV2_infer_int8/ --do_vela
    popd
    cp -rf model_zoo/PicoDetV2_infer_int8/out_vela.bin verify/vela.bin
    python model_zoo/PicoDetV2_infer_int8/picodet_preprocess_load.py $PIC_PATH
}

blazeface_preprocess() {
    echo "Running preprocess steps for blazeface..."
    # generate input_tensor.bin and copy the vela.bin from model zoo
    mkdir -p model_vela

    if [ ! -e "./readnb/test_asset/blazeface/g_BlazeFace.json" ]; then
        echo "Please put generate model json file into ./readnb/test_asset/blazeface/g_BlazeFace.json"
        echo "Then 'patch -p0 g_BlazeFace.json < g_BlazeFace.patch'"
        exit
    fi

    pushd readnb
    python write_model.py --model_path ./test_asset/blazeface/g_BlazeFace.json --out_dir ../model_vela --do_vela
    popd

    cp -rf model_vela/out_vela.bin verify/vela.bin
    python model_zoo/BlazeFace_infer_int8/blazeface_preprocess_load.py $PIC_PATH
}

ppocr_det_preprocess() {
    echo "Running preprocess steps for ppocr_det..."
    # Convert model to generate vela.bin
    pushd readnb
    if [ ! -e "../model_zoo/PpocrDet_infer_int8/g_ch_ppocr_mobile_v2.0_det_slim_opt.json" ]; then
        echo "Please put generated model json file 'g_ch_ppocr_mobile_v2.0_det_slim_opt.json' into 'model_zoo/PpocrDet_infer_int8' directory"
        echo "Then 'patch -p0 model_zoo/PpocrDet_infer_int8/g_ch_ppocr_mobile_v2.0_det_slim_opt.json < readnb/test_asset/ppocr_det/g_ch_ppocr_det.patch'"
        exit
    fi
    python write_model.py --model_path ../model_zoo/PpocrDet_infer_int8/g_ch_ppocr_mobile_v2.0_det_slim_opt.json --out_dir ../model_zoo/PpocrDet_infer_int8 --do_vela
    popd
    cp -rf model_zoo/PpocrDet_infer_int8/out_vela.bin verify/vela.bin
    python model_zoo/PpocrDet_infer_int8/ppocr_det_preprocess_load.py $PIC_PATH
}

humanseg_preprocess() {
    echo "Running preprocess steps for humanseg..."
    # generate input_tensor.bin and copy the vela.bin from model zoo
    mkdir -p model_vela

    if [ ! -e "./readnb/test_asset/humanseg/g_humanseg_v2_int8.json" ]; then
        echo "Please put generate model json file into ./readnb/test_asset/humanseg/g_humanseg_v2_int8.json"
        echo "Then 'patch -p0 g_humanseg_v2_int8.json < g_humanseg_v2_int8.patch'"
        exit
    fi

    pushd readnb
    python write_model.py --model_path ./test_asset/humanseg/g_humanseg_v2_int8.json --out_dir ../model_vela --do_vela
    popd
    # cp -rf model_zoo/Human_pp_humansegv2_lite_avgmax_int8_opt/vela.bin verify/vela.bin
    cp -rf model_vela/out_vela.bin verify/vela.bin
    python model_zoo/Human_pp_humansegv2_lite_avgmax_int8_opt/humanseg_preprocess_load.py $PIC_PATH
}

ppocr_rec_preprocess() {
    echo "Running preprocess steps for ppocr_rec..."
    # Convert model to generate vela.bin
    pushd readnb
    if [ ! -e "../model_zoo/PpocrRec_infer_int8/g_ch_ppocr_mobile_v2.0_rec_slim_opt.json" ]; then
        echo "Please put generated model json file 'g_ch_ppocr_mobile_v2.0_rec_slim_opt.json' into 'model_zoo/PpocrRec_infer_int8' directory"
        echo "Then 'patch -p0 model_zoo/PpocrRec_infer_int8/g_ch_ppocr_mobile_v2.0_rec_slim_opt.json < readnb/test_asset/ppocr_rec/g_ch_ppocr_rec.patch'"
        exit
    fi
    python write_model.py --model_path ../model_zoo/PpocrRec_infer_int8/g_ch_ppocr_mobile_v2.0_rec_slim_opt.json --out_dir ../model_zoo/PpocrRec_infer_int8 --do_vela
    popd
    cp -rf model_zoo/PpocrRec_infer_int8/out_vela.bin verify/vela.bin
    python model_zoo/PpocrRec_infer_int8/ppocr_rec_preprocess_load.py $PIC_PATH
}

ppocr_layout_preprocess() {
    echo "Running preprocess steps for ppocr_layout..."
    # generate input_tensor.bin and copy the vela.bin from model zoo
    mkdir -p model_vela

    if [ ! -e "./readnb/test_asset/ppocr_layout/g_PicoDet_layout_1x_infer_optimized.json" ]; then
        echo "Please put generate model json file into ./readnb/test_asset/ppocr_layout/g_PicoDet_layout_1x_infer_optimized.json"
        echo "Then 'patch -p0 g_PicoDet_layout_1x_infer_optimized.json < g_PicoDet_layout_1x_infer_optimized.patch'"
        exit
    fi

    pushd readnb
    python write_model.py --model_path ./test_asset/ppocr_layout/g_PicoDet_layout_1x_infer_optimized.json --out_dir ../model_vela --do_vela
    popd

    # not pass
    cp -rf model_vela/out_vela.bin verify/vela.bin
    cp -rf model_zoo/Ppocr_layout_infer_int8/weights/*.bin verify/
    python model_zoo/Ppocr_layout_infer_int8/ppocr_layout_preprocess_load.py $PIC_PATH
}

mv1_postprocess() {
    echo "Running postprocess steps for MobileNetV1..."
    python model_zoo/MobileNetV1_infer_int8/mv1_postprocess_load.py verify/output_tensor.bin 1,1000
}

pplcnetv2_postprocess() {
    echo "Running postprocess steps for PPLCNetV2..."
    python model_zoo/PPLCNetV2_infer_int8/pplc_postprocess_load.py verify/output_tensor.bin 1,1000
}

pp_tinypose_postprocess() {
    echo "Running postprocess steps for PP-TinyPose..."

    if [ -z "$CONFIDENCE_THRESHOLD" ]; then
        echo "Confidence Threshold is Defult [0.001]" >&2
        CONFIDENCE_THRESHOLD=0.001
    else
        echo "Confidence Threshold is $CONFIDENCE_THRESHOLD" >&2
    fi
    python model_zoo/PP_TinyPose_128x96_qat_dis_nopact_opt/tinypose_postprocess_load.py --image $PIC_PATH --confidence $CONFIDENCE_THRESHOLD --output verify/output_tensor.bin
}

picodet_postprocess() {
    echo "Running postprocess steps for PicoDet..."

    if [ -z "$CONFIDENCE_THRESHOLD" ]; then
        echo "Confidence Threshold is Defult [0.5]" >&2
        CONFIDENCE_THRESHOLD=0.5
    else
        echo "Confidence Threshold is $CONFIDENCE_THRESHOLD" >&2
    fi
    python model_zoo/PicoDetV2_infer_int8/picodet_postprocess.py --output_dir model_zoo/PicoDetV2_infer_int8/output --image $PIC_PATH --confidence $CONFIDENCE_THRESHOLD --result "model_zoo/PicoDetV2_infer_int8/" --label_file ./model_zoo/PicoDetV2_infer_int8/coco_label_list.txt
}

blazeface_postprocess() {
    echo "Running postprocess steps for BlazeFace..."

    if [ -z "$CONFIDENCE_THRESHOLD" ]; then
        echo "Confidence Threshold is Defult [0.65]" >&2
        CONFIDENCE_THRESHOLD=0.65
    else
        echo "Confidence Threshold is $CONFIDENCE_THRESHOLD" >&2
    fi
    python model_zoo/BlazeFace_infer_int8/int2float.py
    python model_zoo/BlazeFace_infer_int8/blazeface_postprocess.py --image $PIC_PATH --confidence $CONFIDENCE_THRESHOLD
}

ppocr_det_postprocess() {
    echo "Running postprocess steps for Ppocr_det..."

    if [ -z "$CONFIDENCE_THRESHOLD" ]; then
        echo "Confidence Threshold is Defult [0.6]" >&2
        CONFIDENCE_THRESHOLD=0.6
    else
        echo "Confidence Threshold is $CONFIDENCE_THRESHOLD" >&2
    fi
    python model_zoo/PpocrDet_infer_int8/ppocr_det_postprocess.py --image $PIC_PATH --confidence $CONFIDENCE_THRESHOLD --output_path ./model_zoo/PpocrDet_infer_int8/detresult.jpg --model_output_path ./verify/output_tensor.bin
}

humanseg_postprocess() {
    echo "Running postprocess steps for Humanseg..."
    python model_zoo/Human_pp_humansegv2_lite_avgmax_int8_opt/humanseg_postprocess.py $PIC_PATH ./verify/output_tensor.bin
}

ppocr_rec_postprocess() {
    echo "Running postprocess steps for Ppocr_rec..."
    python model_zoo/PpocrRec_infer_int8/ppocr_rec_postprocess.py  --model_output_path ./verify/output_tensor.bin --dict_path model_zoo/PpocrRec_infer_int8/labels/ppocr_keys_v1.txt --weight_path model_zoo/PpocrRec_infer_int8/weights
}
ppocr_layout_postprocess() {
    echo "Running postprocess steps for Ppocr_layout..."

    if [ -z "$CONFIDENCE_THRESHOLD" ]; then
        echo "Confidence Threshold is Defult [0.42]" >&2
        CONFIDENCE_THRESHOLD=0.42
    else
        echo "Confidence Threshold is $CONFIDENCE_THRESHOLD" >&2
    fi
    python model_zoo/Ppocr_layout_infer_int8/int2float.py
    python model_zoo/Ppocr_layout_infer_int8/layout_postprocess/layout_postprocess.py --image $PIC_PATH --confidence $CONFIDENCE_THRESHOLD --output_dir model_zoo/Ppocr_layout_infer_int8/layout_postprocess/output
}

paddle_run() {
    WORK_ROOT=verify
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
    xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h

    cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

    if [[ $# -eq 2 ]];then
        echo "Vela run to dump file"
        ./lite/backends/arm/run_only.sh $1 $2 > verify/paddle_runner.txt 2>&1
    else
        echo "Vela build & run without dump file"
        echo "./lite/backends/arm/build_only.sh $MODEL_TYPE"
        ./lite/backends/arm/build_only.sh $MODEL_TYPE
        echo "The vela binary only represents the execution of operators of the compiled subgraph."
        echo "Build finished"
        ./lite/backends/arm/run_only.sh > verify/paddle_runner.txt 2>&1
        sync
        sleep 1
        # cat verify/paddle_runner.txt | grep -E "output_addr address|output shapes"
        o_address=$(cat verify/paddle_runner.txt | grep -E "output tensor output_addr address" | sed 's/.*output_addr address\s*\(0x[0-9a-fA-F]*\)/\1/' | sed 's/^\s*//;s/\s*$//')
        o_shapes=$(cat verify/paddle_runner.txt | grep -E "output shapes" | sed 's/.*output shapes //' | sed 's/^\s*//;s/\s*$//')
        ./lite/backends/arm/run_only.sh ${o_address} ${o_shapes} > /dev/null 2>&1
    fi
}

paddle_run_picodet() {
    WORK_ROOT=verify
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
    xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h

    cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

    if [[ $# -eq 2 ]];then
        echo "Vela run to dump file"
        ./lite/backends/arm/run_only.sh $1 $2 > verify/paddle_runner.txt 2>&1
    else
        echo "Vela build & run pico model dump file"
        ./lite/backends/arm/build_only.sh $MODEL_TYPE
        echo "The vela binary only represents the execution of operators of the compiled subgraph."
        echo "Build finished"
        sleep 1
        ./lite/backends/arm/run_only.sh $1 $2 > verify/paddle_runner.txt 2>&1
        ./lite/backends/arm/run_only.sh  0x7c4c9010 865280 0x7c59c418 216320 0x7c5d1120 54080 0x7c5de468 15680 0x7c5e21b0 346112 0x7c6369b8  86528 0x7c64bbc0 21632 0x7c651048 6272 > /dev/null 2>&1
    fi
}

paddle_run_blazeface() {
    WORK_ROOT=verify
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
    xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h

    cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

    echo "Vela build & run blazeface model dump file"
    ./lite/backends/arm/build_only.sh $MODEL_TYPE
    echo "The vela binary only represents the execution of operators of the compiled subgraph."
    echo "Build finished"
    sleep 1
    ./lite/backends/arm/run_only.sh 0x7c043cd0 8192 0x7c045cd0 512 0x7c045ed0 6144 0x7c047a30 384 > /dev/null 2>&1
}

paddle_run_ppocr_layout() {
    WORK_ROOT=verify
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
    xxd -i $WORK_ROOT/b1.bin > $WORK_ROOT/b1.h
    xxd -i $WORK_ROOT/b2.bin > $WORK_ROOT/b2.h
    xxd -i $WORK_ROOT/b3.bin > $WORK_ROOT/b3.h
    xxd -i $WORK_ROOT/b4.bin > $WORK_ROOT/b4.h
    xxd -i $WORK_ROOT/b5.bin > $WORK_ROOT/b5.h
    xxd -i $WORK_ROOT/b6.bin > $WORK_ROOT/b6.h
    xxd -i $WORK_ROOT/w1.bin > $WORK_ROOT/w1.h
    xxd -i $WORK_ROOT/w2.bin > $WORK_ROOT/w2.h
    xxd -i $WORK_ROOT/w3.bin > $WORK_ROOT/w3.h
    xxd -i $WORK_ROOT/w4.bin > $WORK_ROOT/w4.h
    xxd -i $WORK_ROOT/w5.bin > $WORK_ROOT/w5.h
    xxd -i $WORK_ROOT/w6.bin > $WORK_ROOT/w6.h
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h

    xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h

    cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/b*.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/w*.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

    echo "Vela build & run ppocr_layout model dump file"
    ./lite/backends/arm/build_only.sh $MODEL_TYPE
    echo "The vela binary represents the execution of operators of the compiled subgraph."
    echo "Build finished"
    sleep 1
    ./lite/backends/arm/run_only.sh $MODEL_TYPE 0x7cceba20 281200 0x7cd6ba90 70300 0x7cd30490 17575 0x7cd34940 4810 > verify/paddle_runner.txt 2>&1
}

paddle_run_ppocr_rec() {
    WORK_ROOT=verify
    MODEL_ROOT=$1
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
    xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h

    cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

    cp -r $MODEL_ROOT/weight lite/backends/arm/paddle_runner/
    cp -r $MODEL_ROOT/rec_postprocess.h lite/backends/arm/paddle_runner/

    echo "Vela build & run ppocr_rec model dump file"
    ./lite/backends/arm/build_only.sh $MODEL_TYPE
    echo "The vela binary represents the execution of operators of the compiled subgraph."
    echo "Build finished"
    sleep 1
    ./lite/backends/arm/run_only.sh $MODEL_TYPE 0x7c198b20 655360
}

paddle_run_tinypose() {
    WORK_ROOT=verify
    xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
    xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h
    xxd -i $WORK_ROOT/vela2.bin > $WORK_ROOT/vela2.h
    xxd -i $WORK_ROOT/vela3.bin > $WORK_ROOT/vela3.h
    xxd -i $WORK_ROOT/vela4.bin > $WORK_ROOT/vela4.h

    cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/vela2.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/vela3.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/vela4.h lite/backends/arm/paddle_runner/
    cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

    if [[ $# -eq 2 ]];then
        echo "Vela run to dump file"
        ./lite/backends/arm/run_only.sh $1 $2 > verify/paddle_runner.txt 2>&1
    else
        echo "Vela build & run tinypose model dump file"
        echo "./lite/backends/arm/build_only.sh $MODEL_TYPE"
        ./lite/backends/arm/build_only.sh $MODEL_TYPE
        echo "The vela binary represents the execution of operators of the compiled subgraph."
        echo "Build finished"
        ./lite/backends/arm/run_only.sh > verify/paddle_runner.txt 2>&1
        sync
        sleep 1
        # cat verify/paddle_runner.txt | grep -E "output_addr address|output shapes"
        o_address=$(cat verify/paddle_runner.txt | grep -E "output tensor output_addr address" | sed 's/.*output_addr address\s*\(0x[0-9a-fA-F]*\)/\1/' | sed 's/^\s*//;s/\s*$//')
        o_shapes=$(cat verify/paddle_runner.txt | grep -E "output shapes" | sed 's/.*output shapes //' | sed 's/^\s*//;s/\s*$//')
        ./lite/backends/arm/run_only.sh ${o_address} ${o_shapes} > /dev/null 2>&1
    fi
}


case $MODEL_TYPE in
    mv1)
        echo "You have selected MobileNetV1."
        mv1_preprocess
        paddle_run
        mv1_postprocess
        ;;
    pplcnetv2)
        echo "You have selected PP-LCNetV2."
        pplcnetv2_preprocess
        paddle_run
        pplcnetv2_postprocess
        ;;
    tinypose)
        echo "You have selected PP_TinyPose_128x96_qat_dis_nopact_opt"
        pp_tinypose_preprocess
        paddle_run_tinypose
        pp_tinypose_postprocess
        ;;
    picodetv2)
        echo "You have selected PICODetV2."
        picodetv2_preprocess
        paddle_run_picodet
        picodet_postprocess
        ;;
    blazeface)
        echo "You have selected blazeface"
        blazeface_preprocess
        paddle_run_blazeface
        blazeface_postprocess
        ;;
    ppocr_det)
        echo "You have selected ppocr_det"
        ppocr_det_preprocess
        paddle_run
        ppocr_det_postprocess
        ;;
    humansegv2)
        echo "You have selected humansegv2"
        humanseg_preprocess
        paddle_run
        humanseg_postprocess
        ;;
    ppocr_rec)
        echo "You have selected ppocr_rec"
        ppocr_rec_preprocess
        paddle_run_ppocr_rec model_zoo/PpocrRec_infer_int8/post_process
        ;;
    ppocr_layout)
        echo "You have selected ppocr_layout"
        ppocr_layout_preprocess
        paddle_run_ppocr_layout
        ppocr_layout_postprocess
        ;;
    *)
        echo "Unknown model type. Defaulting to MobileNetV1."
        ;;
esac
