import os
import subprocess
import re
import time

def get_gpu_utilization():
    output = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True)
    output = output.decode("utf-8").strip()
    utilizations = [int(x) for x in re.findall(r"\d+", output)]
    return sum(utilizations) / len(utilizations)


def get_gpu_utilization_one_minute():
    gpu_util_list = []
    for i in range(10):
        time.sleep(1)
        gpu_util = get_gpu_utilization()
        gpu_util_list.append(gpu_util)
    avg_gpu_util = sum(gpu_util_list) / len(gpu_util_list)
    return avg_gpu_util


def run_script_b():
    # 在此处替换“Script_B.py”为你的脚本B的实际路径
    subprocess.call(["/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/conda/envs/unicr1/bin/python", "/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/train_null.py"])


def main():
    low_usage_count = 0
    while True:
        # gpu_utilization = get_gpu_utilization()
        gpu_utilization = get_gpu_utilization_one_minute()
        print("Current GPU utilization: {:.2f}%".format(gpu_utilization))
        if gpu_utilization < 40:
            low_usage_count += 1
        else:
            low_usage_count = 0

        if low_usage_count >= 2:
            print("GPU utilization is low for two consecutive checks, running script B.")
            run_script_b()
            low_usage_count = 0

        time.sleep(300)  # 等待15分钟

if __name__ == "__main__":
    main()