# rtmdet

## adeploy
```shell
cd mmdeploy
python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime_static.py \
../configs/rtmdet/rotated_rtmdet_tiny-6x-lpst.py \
../work_dirs/rotated_rtmdet_tiny-6x-lpst/epoch_72.pth \
lpst.jpg \
--work-dir mmdeploy_models/mmrotate/lpst \
--device cuda \
--show \
--dump-info


python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
../configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
../work_dirs/rotated_rtmdet_tiny-6x-hituav/epoch_72.pth \
hit-uav.jpg \
--work-dir mmdeploy_models/mmrotate/ort \
--device cpu \
--show \
--dump-info

python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
../configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
../work_dirs/rotated_rtmdet_tiny-6x-hituav/epoch_72.pth \
hit-uav.jpg \
--work-dir mmdeploy_models/mmrotate/ort \
--device cuda \
--show \
--dump-info

python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime-fp16_dynamic.py \
../configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
../work_dirs/rotated_rtmdet_tiny-6x-hituav/epoch_72.pth \
hit-uav.jpg \
--work-dir mmdeploy_models/mmrotate/ort \
--device cuda \
--show \
--dump-info


python tools/deploy.py \
configs/mmrotate/rotated-detection_sdk_dynamic.py \
../configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
../work_dirs/rotated_rtmdet_tiny-6x-hituav/epoch_72.pth \
hit-uav.jpg \
--work-dir mmdeploy_models/mmrotate/ort \
--device cuda \
--show \
--dump-info


cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATHpython3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
wget https://github.com/open-mmlab/mmrotate/raw/main/demo/dota_demo.jpg

# 3.2.2 onnxruntime-gpu
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime-gpu==1.14.0
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-gpu-1.12.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
../configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py \
../work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota/epoch_12.pth \
dota_demo.jpg \
--work-dir mmdeploy_models/mmrotate/ort \
--device cpu \
--show \
--dump-info
```

## arun
```shell
使用 SyncBN
orcnn  删除空数据 使用map12 使用trainval评估
修改评估标准，保留小数


----------------------------------------*****|RTMDet|*****---------------------------------------------------
bash script/rtmdet.sh
bash script/testfps.sh
CUDA_VISIBLE_DEVICES=5 tools/dist_train.sh configs/rtmdet/rotated_rtmdet_tiny-6x-lpst.py 1


CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/rtmdet/rotated_rtmdet_tiny-6x-lpst.py
CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/rtmdet/rotated_rtmdet_l-6x-lpst.py

CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/rtmdet/rotated_rtmdet_tiny-6x-hit_uav.py --resume
CUDA_VISIBLE_DEVICES=6 python tools/train.py configs/rtmdet/rotated_rtmdet_l-6x-hit_uav.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/rtmdet/rotated_rtmdet_tiny-100e-aug-hit_uav.py
---------------------------------------------------------------------------------------------------------------


------------------*****|可视化数据集|*****-----------------
python tools/analysis_tools/browse_dataset.py \
configs/rtmdet/rotated_rtmdet_l-6x-lpst.py \
--output-dir work_dirs/lpst --show-interval 0.001

python tools/analysis_tools/browse_dataset.py \
configs/rtmdet/rotated_rtmdet_tiny-100e-aug-hit_uav.py \
--output-dir work_dirs/aug-hit_uav --show-interval 0.001

python tools/analysis_tools/browse_dataset.py \
configs/rtmdet/rotated_rtmdet_l-3x-hit_uav.py \
--output-dir work_dirs/hit_uav --show-interval 0.001
---------------------------------------------------------

--------------------*****|Flops|*****--------------------

CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/get_flops.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hit_uav.py --shape 256 256

CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/get_flops.py \
configs/rtmdet/rotated_rtmdet_l-6x-hit_uav.py --shape 1280 1024

CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/get_flops.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py --shape 640 512

CUDA_VISIBLE_DEVICES=5 python tools/analysis_tools/get_flops.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py --shape 1280 1024
---------------------------------------------------------


------------------*****|可视化结果|*****---------------
CUDA_VISIBLE_DEVICES=6 python tools/test.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
work_dirs/rotated_rtmdet_tiny-6x-hituav/epoch_72.pth \
--out work_dirs/rotated_rtmdet_tiny-6x-hituav/main.pkl

CUDA_VISIBLE_DEVICES=6 python tools/test.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
work_dirs/rotated_rtmdet_tiny-6x-hituav/epoch_72.pth \
--show --show-dir ../show

python tools/analysis_tools/confusion_matrix.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
work_dirs/rotated_rtmdet_tiny-6x-hituav/main.pkl \
work_dirs/rotated_rtmdet_tiny-6x-hituav


CUDA_VISIBLE_DEVICES=5 python tools/test.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-lpst.py \
work_dirs/rotated_rtmdet_tiny-6x-lpst/epoch_72.pth \
--show --show-dir ./show

python tools/analysis_tools/confusion_matrix.py \
configs/rtmdet/rotated_rtmdet_tiny-6x-hituav.py \
work_dirs/rotated_rtmdet_tiny-6x-hituav/main.pkl \
work_dirs/rotated_rtmdet_tiny-6x-hituav
------------------------------------------------------
```