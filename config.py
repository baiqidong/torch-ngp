from __future__ import print_function
import platform
import cpuinfo
import os
import subprocess
from rich.console import Console
from tabulate import tabulate
import time


class Config(object):
    def __init__(self):
        #hardware
        # self.cpu_info = ""
        # self.gpu_info = ""

        # software
        # self.os_version = ""
        # self.sdk_version = ""

        #dataset
        self.imagesize = ""
        self.image_mode = ""
        # blender/colmap/llff/
        self.datatype = ""
        self.dataset_name = ""
        self.train_data_size = 0
        self.test_data_size = 0
        self.val_data_size = 0

        #train parameter
        self.epoch = 0
        self.batch_size = 0

        self.start_time = ""
        self.end_time = ""
        self.load_traindata_time = ""
        self.load_testdata_time = ""
        self.load_valdata_time = ""
        # self.load_data_time = ""
        self.train_time = ""
        self.test_time = ""
        self.val_time = ""
        self.total_time = ""
        self.save_mesh_time = ""


# 获取操作系统名称
def get_processor_name():
    if platform.system() == "Windows":
        return platform.platform()
    # elif platform.system() == "Darwin":
    #     os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
    #     command ="sysctl -n machdep.cpu.brand_string"
    #     return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        if os.path.exists("/etc/redhat-release"):
            command = "cat /etc/redhat-release"
        else:
            command = "cat /etc/issue"
        os_version = subprocess.check_output(command, shell=True).decode().strip()
        return os_version


#gpu信息
def get_gpu_info():
    os_version = get_processor_name()
    table = [['OS-Version', os_version]]

    command = "ixsmi"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    s = all_info.split("\n")
    ss = s[2].split()
    ss4 = s[4].split()
    ss5 = s[5].split()
    ss7 = s[7].split()
    ss8 = s[8].split()
    table.append([ss[1][:-1], ss[2]])  # IX-ML
    table.append([ss[3] + " " + ss[4][:-1], ss[5]])  # Driver Version
    table.append([ss[6] + " " + ss[7][:-1], ss[8]])  # CUDA Version

    table.append([ss4[1], ss7[1]])  # GPU
    table.append([ss4[2], ss7[2] + " " + ss7[3]])  # Name
    table.append([ss4[4], ss7[5]])  # Bus-Id

    table.append([ss5[8] + " " + ss5[9], ss8[13]])  # Compute M.
    table.append([ss5[3], ss8[3]])  # Perf

    return table


#cpu信息
def get_cpu_infos():
    out = cpuinfo.get_cpu_info()
    cpu_info = [
             ['brand_raw', str(out['brand_raw'])],
             ['arch', str(out['arch'])],
             ['count', str(out['count'])]]
    return cpu_info


def log(log_ptr, *args, **kwargs):
    Console().print(*args, **kwargs)
    print(*args, file=log_ptr)
    log_ptr.flush() # write immediately to file


def prn_obj(obj, log_ptr):
    # print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
    # log(log_ptr, '\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

    log(log_ptr, "== == == == == == == == CPU Info == == == == == == == ==")
    log(log_ptr, tabulate(get_cpu_infos(), tablefmt='grid'))

    log(log_ptr, "== == == == == == == == GPU Info == == == == == == == ==")
    log(log_ptr, tabulate(get_gpu_info(), tablefmt='grid'))

    table = []
    for item in vars(obj).items():
        table.append(list(item))
    log(log_ptr, tabulate(table, tablefmt='grid'))


#根据开始时间戳和结束时间戳来求耗时
def take_up_time_format(start_time, end_time):
    stime_stamp = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    etime_stamp = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    return time.strftime("%H:%M:%S", time.gmtime(etime_stamp - stime_stamp))


def take_up_time(start_time, end_time):
    stime_stamp = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    etime_stamp = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    return etime_stamp - stime_stamp


if __name__ == '__main__':
    stime_stamp = time.mktime(time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    time.sleep(10)
    etime_stamp = time.mktime(time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    print(time.strftime("%H:%M:%S", time.gmtime(etime_stamp - stime_stamp)))


    stime_stamp = time.mktime(time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    time.sleep(10)
    etime_stamp = time.mktime(time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    print(etime_stamp - stime_stamp)


    os.makedirs("workspace", exist_ok=True)
    report_path = os.path.join("workspace", "report.txt")
    log_ptr = open(report_path, "a+")
    config = Config()
    prn_obj(config, log_ptr)





