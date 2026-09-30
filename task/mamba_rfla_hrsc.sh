#!/bin/bash
config_dir="configs/drfnet"
configs=(
    "oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_mamba_rfla"
)

device_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for config in "${configs[@]}"; do
    config_path="${config_dir}/${config}.py"
    if [[ -f "tools/dist_train.sh" && -f "tools/dist_test.sh" && -f "$config_path" ]]; then
        tools/dist_train.sh "$config_path" $device_num
        # tools/dist_test.sh "$config_path" "/output/work_dirs/work_dirs/${config}/epoch_12.pth" $device_num
    else
        echo -e "\e[31mRequired files for ${config} are missing.\e[0m"
    fi
done


# rm -rf mmrotate-lcs
# git clone https://gitee.com/luchaoshi/mmrotate-lcs.git
# cd mmrotate-lcs
# chmod -R 777 *
# pip install -v -e .
# bash task.sh