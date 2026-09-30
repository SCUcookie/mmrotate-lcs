rm -rf mmrotate-lcs;
git clone https://gitee.com/luchaoshi/mmrotate-lcs.git;
cd mmrotate-lcs; chmod -R 777 *; pip install -v -e .;



# for file in configs/dflan/task/*.sh; do
#   echo "Running $file";
#   cp "$file" "task.sh"
#   bash task.sh;
# done



# cp configs/dflan/task/base_1x.sh task.sh;
# bash task.sh;

# cp configs/dflan/task/base_2x.sh task.sh;
# bash task.sh;

# cp configs/dflan/task/base_3x.sh task.sh;
# bash task.sh;

# cp configs/dflan/task/base_4x.sh task.sh;
# bash task.sh;

cp configs/dflan/task/base_18e.sh task.sh;
bash task.sh;


# cp configs/dflan/task/fsipn_rfla_dhead_1x.sh task.sh;
# bash task.sh;

# cp configs/dflan/task/fsipn_rfla_dhead_2x.sh task.sh;
# bash task.sh;

# cp configs/dflan/task/fsipn_rfla_dhead_3x.sh task.sh;
# bash task.sh

# cp configs/dflan/task/fsipn_rfla_dhead_4x.sh task.sh;
# bash task.sh

cp configs/dflan/task/fsipn_rfla_dhead_18e.sh task.sh;
bash task.sh