#!/usr/bin/env bash

WORK_ROOT=$1
ADDRESS=$2
OFFSET=$3
echo "WORK_ROOT: ${WORK_ROOT}"
xxd -i $WORK_ROOT/input_tensor.bin > $WORK_ROOT/input_tensor.h
xxd -i $WORK_ROOT/vela.bin > $WORK_ROOT/vela.h

cp $WORK_ROOT/vela.h lite/backends/arm/paddle_runner/
cp $WORK_ROOT/input_tensor.h lite/backends/arm/paddle_runner/

if [[ $# -eq 3 ]];then
    echo "run to dump file"
    ./lite/backends/arm/run_only.sh $2 $3 > $WORK_ROOT/paddle_runner.txt 2>&1
else
    echo "build & run wo dump file"
    ./lite/backends/arm/build_only.sh testmodel
    echo "build finished"
    ./lite/backends/arm/run_only.sh > $WORK_ROOT/paddle_runner.txt 2>&1
    cat $WORK_ROOT/paddle_runner.txt | grep -E "output_addr address|output shapes"
    exit
fi

cat $WORK_ROOT/paddle_runner.txt | grep "paddle lite arm output:" > $WORK_ROOT/output.txt

