# python tools/test.py \
# configs/comp/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py \
# work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth \
# --show --show-dir ../show

# python tools/test.py \
# configs/comp/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py \
# work_dirs/oriented-rcnn-le90_r50_fpn_1x_dota/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
# --show --show-dir ../show

python tools/test.py \
configs/comp/lsknet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s.py \
work_dirs/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s/epoch_12.pth \
--show --show-dir ../show

python tools/test.py \
configs/comp/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla_head.py \
work_dirs/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla_head/epoch_12.pth \
--show --show-dir ../show

# python tools/test.py \
# configs/comp/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla_head.py \
# work_dirs/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla_head2/epoch_12.pth \
# --show --show-dir ../show