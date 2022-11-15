import time
import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *
from NerfReport import *
import sys

from functools import partial
from loss import huber_loss

#torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    report = NerfReport()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('[TIMESTAMP] start time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--epochs', type=int, default=-1, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### test options
    parser.add_argument('--fps', type=int, default=15, help="test video fps")
    parser.add_argument('--duration', type=int, default=10, help="video duration second")
    parser.add_argument('--view', type=str, default='yaw', help="view direction:random or yaw")

    ### evaluate options
    parser.add_argument('--eval_interval', type=int, default=50, help="eval_interval")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")

    ### hash parameter
    parser.add_argument('--num_levels', type=int, default=16, help="hash number of levels")
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help="hash map size of log2")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    process_name = "python3 "
    for i in sys.argv:
        process_name += i + " "
    report.set_process_name(process_name)

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        num_levels=opt.num_levels,
        log2_hashmap_size=opt.log2_hashmap_size,
    )

    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    # criterion = partial(huber_loss, reduction='none')
    # criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            print('[TIMESTAMP] start load test data:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            s_load_testdata_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            e_load_testdata_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[TIMESTAMP] load test data end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            report.set_load_testdata_time(take_up_time_format(s_load_testdata_time, e_load_testdata_time))
            report.set_test_data_size(len(test_loader))

            if test_loader.has_gt:
                print('[TIMESTAMP] start evaluate:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

                e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('[TIMESTAMP] evaluate end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                report.set_val_time(take_up_time_format(s_time, e_time))

            print('[TIMESTAMP] start test:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            trainer.test(test_loader, write_video=True)  # test and save video

            e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[TIMESTAMP] test end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            report.set_test_time(take_up_time_format(s_time, e_time))

            print('[TIMESTAMP] start save mesh:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            trainer.save_mesh(resolution=256, threshold=10)
            print('[TIMESTAMP] save mesh end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        print('[TIMESTAMP] start load train data:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('[TIMESTAMP] load train data end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        report.set_load_traindata_time(take_up_time_format(s_time, e_time))

        report.set_dataset_name(opt.path.split('/')[1])
        report.set_train_data_size(len(train_loader))
        report.set_datatype(opt.datatype)
        report.set_imagesize(opt.imagesize)
        report.set_image_mode(opt.image_mode)

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                  lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                          eval_interval=opt.eval_interval)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            print('[TIMESTAMP] start load val data:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[TIMESTAMP] load val data end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            report.set_load_valdata_time(take_up_time_format(s_time, e_time))

            report.set_val_data_size(len(valid_loader))

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            if opt.epochs != -1:
                max_epoch = opt.epochs

            report.set_epoch(max_epoch)
            report.set_batch_size(train_loader.batch_size)

            print('[TIMESTAMP] start trainning:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            trainer.train(train_loader, valid_loader, max_epoch)

            e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[TIMESTAMP] trainning end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            training_time = take_up_time(s_time, e_time)
            report.set_loss_dict(trainer.loss_dict)
            report.set_psnr_dict(trainer.psnr_dict)
            report.set_ssim_dict(trainer.ssim_dict)
            report.set_lpips_dict(trainer.lpips_dict)
            report.set_start_epoch(trainer.start_epoch)

            evl_sum = 0
            for i in trainer.evl_time:
                evl_sum = evl_sum + i
            print('Val time:'+str(evl_sum))
            report.set_val_time(time.strftime("%H:%M:%S", time.gmtime(evl_sum)))
            report.set_train_time(time.strftime("%H:%M:%S", time.gmtime(training_time-evl_sum)))

            # also test
            print('[TIMESTAMP] start load test data:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            s_load_testdata_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            e_load_testdata_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[TIMESTAMP] load test data end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            report.set_load_testdata_time(take_up_time_format(s_load_testdata_time, e_load_testdata_time))
            report.set_test_data_size(len(test_loader))

            if test_loader.has_gt:
                print('[TIMESTAMP] start evaluate:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

                e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('[TIMESTAMP] evaluate end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                report.set_val_time(time.strftime("%H:%M:%S", time.gmtime(evl_sum + take_up_time(s_time, e_time))))
                print(report.get_val_time())

            print('[TIMESTAMP] start test:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            s_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            trainer.test(test_loader, write_video=True)  # test and save video

            e_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[TIMESTAMP] test end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            report.set_test_time(take_up_time_format(s_time, e_time))

            print('[TIMESTAMP] start save mesh:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            trainer.save_mesh(resolution=256, threshold=10)
            print('[TIMESTAMP] save mesh end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('[TIMESTAMP] end time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    report.set_total_time(take_up_time_format(start_time, end_time))

    os.makedirs(opt.workspace, exist_ok=True)
    report_path = os.path.join(opt.workspace, "torch-ngp_report.txt")
    log_ptr = open(report_path, "a+")
    prn_obj(report, log_ptr, opt)


