# MMrotate-lcs

## 1 Install

### Eenvironment
```shell
conda create --name lcs_mmrotate --no-default-packages python=3.8
# conda remove --name lcs_mmrotate --all
# rm -r /data1/anaconda3/envs/lcs_mmrotate

conda activate lcs_mmrotate
```

### Requirements
```shell
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
pip install mmengine==0.10.4
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmpretrain==1.2.0
pip install triton==2.0.0
pip install transformers==4.42.4
pip install timm
pip install future tensorboard
# github down
pip install ../../pretrain/causal_conv1d-1.1.3+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install ../../pretrain/mamba_ssm-1.1.3+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## 2 Run

### Train On DOTA
```shell
CUDA_VISIBLE_DEVICES=1,2 /
tools/dist_train.sh configs/comp/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_drfnet.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_rfla.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla_head.py 2
```

### Train On HRSC
```shell
CUDA_VISIBLE_DEVICES=3,4 /
tools/dist_train.sh configs/comp/oriented_rcnn/oriented-rcnn-le90_r50_fpn_3x_hrsc.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_3x_hrsc_drfnet.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_mamba.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_rfla.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_mamba_rfla.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_mamba_rfla_head.py 2
```