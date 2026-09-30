import os
import time
import shutil

class Clear:
    def __init__(self) -> None:
        # 目标路径
        self.target_dir = 'work_dirs'  # 替换为你的目录路径
        # start_with = ['oriented-rcnn', 'rotated_rtmdet']  # 以这些前缀开头的文件夹
        self.start_with = ['oriented-rcnn']  # 以这些前缀开头的文件夹
        self.time_diff = 3600 # 1小时
        # 初始化计数器
        self.log_num = 0
        self.deleted_num = 0
        # 获取一级子目录，并筛选出以指定前缀开头的目录
        self.subdirs = [os.path.join(self.target_dir, name) for name in os.listdir(self.target_dir)
                if os.path.isdir(os.path.join(self.target_dir, name)) and 
                any(name.startswith(prefix) for prefix in self.start_with)]
    def run(self):
        for subdir in self.subdirs:
            self.clear_subdir(subdir)
        
        print(f"\033[1;31m共处理 {self.log_num} 个log文件，删除 {self.deleted_num} 个文件夹\033[0m")

    def clear_subdir(self, subdir):
        logdirs = [os.path.join(subdir, name)  for name in os.listdir(subdir) if 
                os.path.isdir(os.path.join(subdir, name))]
        for logdir in logdirs:
            self.clear_logdir(logdir)

    def clear_logdir(self, logdir):
        for root, dirs, files in os.walk(logdir):
            for file in files:
                if file.endswith('.log'):
                    self.ana_log(os.path.join(root, file))
                    
    def ana_log(self, log_file_path):
        self.log_num += 1
        # 打开并逐行读取log文件
        
        if "3x_hrsc" in log_file_path:
            keyword = "Epoch(train) [36]"
            self.log_line(log_file_path, keyword)
            pass
        elif "1x_dota" in log_file_path:
            keyword = "Epoch(train) [12]"
            self.log_line(log_file_path, keyword)
    
    def log_line(self, log_file_path, keyword):
        with open(log_file_path, 'r') as f:
            for line in f:
                if keyword in line:
                    print(f"\033[92m【保留文件】 {log_file_path}\033[0m")
                    return # 匹配到目标行，不删除文件
        
        # 获取当前时间
        current_time = time.time()
        # 获取文件的最后修改时间
        file_mod_time = os.path.getmtime(log_file_path)
        # 计算时间差（秒）
        time_diff = current_time - file_mod_time

        # 如果文件的最后修改时间在1小时（3600秒）以内，则不删除
        if time_diff <= self.time_diff:
            print(f"\033[94m【保留最近修改的文件】 {log_file_path}\033[0m")
            return
         
        self.deleted_num += 1
        parent_dir = os.path.dirname(log_file_path)
        print(f"\033[91m【删除文件夹】 {parent_dir}\033[0m")
        shutil.rmtree(parent_dir)

class ShowWorkdir:
    def __init__(self):
        self.workdir = "work_dirs"
        self.start_with = ['oriented-rcnn-le90_r50_fpn_1x_dota']  # 以这些前缀开头的文件夹
        self.show_dir = "work_dirs_show"
        if not os.path.exists(self.show_dir):
            os.makedirs(self.show_dir)
            
    def run(self):
        for root, dirs, files in os.walk(self.workdir):
            for file in files:
                if file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    # Check if the file path starts with any of the specified prefixes
                    if any(file_path.startswith(os.path.join(self.workdir, prefix)) 
                           for prefix in self.start_with):
                        new_name = root.split("/")[1] + "_" + file
                        new_file_path = os.path.join(self.show_dir, new_name)
                        shutil.copy(file_path, new_file_path)
                        print(f"\033[92m【复制并重命名文件】 {file_path} \033[0m")
  
if __name__ == "__main__":
    clear = Clear()
    clear.run()
    show_workdir = ShowWorkdir()
    show_workdir.run()
