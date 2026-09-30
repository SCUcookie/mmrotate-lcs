#!/bin/bash
config_dir="configs/dflan"
configs=(
    "orcnn_lsk_fsipn_rfla_dhead_18e_hrsc"
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