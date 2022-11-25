from __future__ import print_function
import platform
import cpuinfo
import os
import subprocess
from rich.console import Console
from tabulate import tabulate
import time


class NerfReport:
    __instance = None
    __flag = False

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if not NerfReport.__flag:
            NerfReport.__flag = True
            self.loss_dict = dict()
            self.psnr_dict = dict()
            self.ssim_dict = dict()
            self.lpips_dict = dict()

            # dataset
            self.imagesize = ""
            self.image_mode = ""
            # blender/colmap/llff/
            self.datatype = ""
            self.dataset_name = ""
            self.train_data_size = 0
            self.test_data_size = 0
            self.val_data_size = 0

            # train parameter
            self.epoch = 0
            self.batch_size = 0

            self.start_epoch = 0
            self.load_traindata_time = ""
            self.load_testdata_time = ""
            self.load_valdata_time = ""
            self.train_time = ""
            self.test_time = ""
            self.val_time = ""
            self.total_time = ""
            self.process_name = ""

    def set_process_name(self, process_name):
        self.process_name = process_name

    def get_process_name(self):
        return self.process_name

    def set_start_epoch(self, start_epoch):
        self.start_epoch = start_epoch

    def get_start_epoch(self):
        return self.start_epoch

    def set_loss_dict(self, loss_dict):
        self.loss_dict = loss_dict

    def get_loss_dict(self):
        return self.loss_dict

    def set_psnr_dict(self, psnr_dict):
        self.psnr_dict = psnr_dict

    def get_psnr_dict(self):
        return self.psnr_dict

    def set_ssim_dict(self, ssim_dict):
        self.ssim_dict = ssim_dict

    def get_ssim_dict(self):
        return self.ssim_dict

    def set_lpips_dict(self, lpips_dict):
        self.lpips_dict = lpips_dict

    def get_lpips_dict(self):
        return self.lpips_dict

    def set_imagesize(self, imagesize):
        self.imagesize = imagesize

    def get_imagesize(self):
        return self.imagesize

    def set_image_mode(self, image_mode):
        self.image_mode = image_mode

    def get_image_mode(self):
        return self.image_mode

    def set_datatype(self, datatype):
        self.datatype = datatype

    def get_datatype(self):
        return self.datatype

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.dataset_name

    def set_train_data_size(self, train_data_size):
        self.train_data_size = train_data_size

    def get_train_data_size(self):
        return self.train_data_size

    def set_test_data_size(self, test_data_size):
        self.test_data_size = test_data_size

    def get_test_data_size(self):
        return self.test_data_size

    def set_val_data_size(self, val_data_size):
        self.val_data_size = val_data_size

    def get_val_data_size(self):
        return self.val_data_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def set_load_traindata_time(self, load_traindata_time):
        self.load_traindata_time = load_traindata_time

    def get_load_traindata_time(self):
        return self.load_traindata_time

    def set_load_testdata_time(self, load_testdata_time):
        self.load_testdata_time = load_testdata_time

    def get_load_testdata_time(self):
        return self.load_testdata_time

    def set_load_valdata_time(self, load_valdata_time):
        self.load_valdata_time = load_valdata_time

    def get_load_valdata_time(self):
        return self.load_valdata_time

    def set_train_time(self, train_time):
        self.train_time = train_time

    def get_train_time(self):
        return self.train_time

    def set_test_time(self, test_time):
        self.test_time = test_time

    def get_test_time(self):
        return self.test_time

    def set_val_time(self, val_time):
        self.val_time = val_time

    def get_val_time(self):
        return self.val_time

    def set_total_time(self, total_time):
        self.total_time = total_time

    def get_total_time(self):
        return self.total_time


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


# gpus
def get_gpus(process_name):
    command = "ixsmi -q"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    s = all_info.split("GPU 00000000")

    gpus = ""
    for i in range(len(s)):
        ss = s[i].split(":")
        for j in range(len(ss)):
            if "python3" in ss[j]:
                temp = ss[j].split("\n")[0]
                if process_name.strip() == temp.strip():
                    gpus += str(i) + " "

    #there is no hostpid in job
    if gpus == "":
        ss = all_info.split("\n")
        for i in range(len(ss)):
            if "Minor" in ss[i]:
                gpus += ss[i].split(':')[1]
    return gpus


# gpu信息
def get_gpu_info(obj):
    table = []
    command = "ixsmi"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    s = all_info.split("\n")
    for i in range(len(s)):
        if "Timestamp" in s[i]:
            s = s[i:]
            break

    ss = s[2].split()
    ss4 = s[4].split()
    ss5 = s[5].split()
    ss7 = s[7].split()
    ss8 = s[8].split()
    table.append([ss[1][:-1], ss[2]])  # IX-ML
    table.append([ss[3] + " " + ss[4][:-1], ss[5]])  # Driver Version
    table.append([ss[6] + " " + ss[7][:-1], ss[8]])  # CUDA Version
    table.append([ss4[2], ss7[2] + " " + ss7[3]])  # Name

    # table.append([ss4[1], ss7[1]])  # GPU
    table.append([ss4[1], get_gpus(obj.process_name)])  # multiple GPU

    return table


# cpu信息
def get_cpu_infos():
    out = cpuinfo.get_cpu_info()
    cpu_info = [
        ['brand_raw', str(out['brand_raw'])],
        ['OS-Version', get_processor_name()],
        ['arch', str(out['arch'])],
        ['count', str(out['count'])]]

    return cpu_info


def log(log_ptr, *args, **kwargs):
    Console().print(*args, **kwargs)
    print(*args, file=log_ptr)
    log_ptr.flush()  # write immediately to file


def print_parameter(obj, log_ptr, train_parameter, *params):
    train_parameter_table = []
    for item in vars(obj).items():
        if item[0] in train_parameter:
            train_parameter_table.append(list(item))
    if len(params) > 0:
        for item in vars(params[0]).items():
            train_parameter_table.append(list(item))
    log(log_ptr, tabulate(train_parameter_table, tablefmt='grid'))
    #log(log_ptr, tabulate(train_parameter_table, tablefmt='grid', maxcolwidths=[None, 100]))


def prn_obj(obj, log_ptr, params):
    log(log_ptr, "CPU Info: ")
    log(log_ptr, tabulate(get_cpu_infos(), tablefmt='grid'))
    log(log_ptr, "\n")

    log(log_ptr, "GPU Info: ")
    log(log_ptr, tabulate(get_gpu_info(obj), tablefmt='grid'))
    log(log_ptr, "\n")

    log(log_ptr, "Dataset: ")
    data = ['imagesize', 'image_mode', 'datatype', 'dataset_name', 'train_data_size', 'test_data_size', 'val_data_size']
    print_parameter(obj, log_ptr, data)
    log(log_ptr, "\n")

    log(log_ptr, "Train Parameters: ")
    train_parameter = ['epoch', 'batch_size']
    print_parameter(obj, log_ptr, train_parameter, params)
    log(log_ptr, "\n")

    log(log_ptr, "Performance: ")
    performance = ['start_epoch', 'load_traindata_time', 'load_testdata_time', 'load_valdata_time',
                   'train_time', 'val_time', 'test_time', 'total_time']
    print_parameter(obj, log_ptr, performance)
    log(log_ptr, "\n")

    if len(obj.loss_dict.items()) > 0:
        log(log_ptr, "Loss: ")
        log(log_ptr, tabulate(list(obj.loss_dict.items()), tablefmt='grid'))
        log(log_ptr, "\n")

    if len(obj.psnr_dict.items()) > 0:
        log(log_ptr, "PSNR: ")
        log(log_ptr, tabulate(list(obj.psnr_dict.items()), tablefmt='grid'))
        log(log_ptr, "\n")

    if len(obj.ssim_dict.items()) > 0:
        log(log_ptr, "SSIM: ")
        log(log_ptr, tabulate(list(obj.ssim_dict.items()), tablefmt='grid'))
        log(log_ptr, "\n")

    if len(obj.lpips_dict.items()) > 0:
        log(log_ptr, "LPIPS: ")
        log(log_ptr, tabulate(list(obj.lpips_dict.items()), tablefmt='grid'))


# 根据开始时间戳和结束时间戳来求耗时
def take_up_time_format(start_time, end_time):
    stime_stamp = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    etime_stamp = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    return time.strftime("%H:%M:%S", time.gmtime(etime_stamp - stime_stamp))


def take_up_time(start_time, end_time):
    stime_stamp = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    etime_stamp = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    return etime_stamp - stime_stamp


if __name__ == '__main__':
    stime_stamp = time.mktime(
        time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    time.sleep(10)
    etime_stamp = time.mktime(
        time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    print(time.strftime("%H:%M:%S", time.gmtime(etime_stamp - stime_stamp)))

    stime_stamp = time.mktime(
        time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳
    time.sleep(10)
    etime_stamp = time.mktime(
        time.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S"))  # 格式化后的时间转换成时间戳

    print(etime_stamp - stime_stamp)

    os.makedirs("workspace", exist_ok=True)
    report_path = os.path.join("workspace", "report.txt")
    log_ptr = open(report_path, "a+")
    # config = NerfReport()
    # prn_obj(config, log_ptr)
