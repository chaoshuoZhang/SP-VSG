import argparse
import os
import cv2
import torch
import torch as th
import numpy as np
from alexnet_eval.dataload import draw_three
from sketch_diffusion import dist_util, logger
from sketch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_2,
    # different modes
    create_model_and_diffusion,
    # create_model_and_diffusion_acc
    # create_model_and_diffusion_noise,
    add_dict_to_argparser,
    args_to_dict,
)
import sketch_diffusion.dpmsolver_pro as DPMsolver


def bin_pen(x, pen_break=0.1):
    result = x[:, :, :3]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j][2] >= pen_break:
                result[i][j][2] = 0
            else:
                result[i][j][2] = 1
    return result


def main():
    args = create_argparser().parse_args()
    if not os.path.exists(args.log_dir + '/test'):
        os.makedirs(args.log_dir + '/test')
    args.log_dir = args.log_dir + '/test'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dist_util.setup_dist()
    logger.configure(args.log_dir)

    logger.log("creating model and diffusion...")
    # different modes, if noise or acc method, please specify 'data', 'raster', and 'loss'.
    model_1, diffusion = create_model_and_diffusion(

        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # args.image_size = int(args.image_size / 3)
    # args.num_channels = int(args.num_channels / 3)
    model_2, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model_1.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model_2.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model_1.to(dist_util.dev())
    model_1.eval()

    model_2.to(dist_util.dev())
    model_2.eval()

    ns = DPMsolver.NoiseScheduleVP('discrete', betas=torch.tensor(diffusion.betas, dtype=torch.float32))
    model_fn_1 = DPMsolver.model_wrapper(model_1, ns)
    model_fn_2 = DPMsolver.model_wrapper(model_2, ns)
    logger.log("sampling...")
    all_images = []
    break_list = args.pen_break
    dpm_solver = DPMsolver.DPM_Solver(model_fn_1, model_fn_2, ns, slerp_step=15)

    while len(all_images) < args.num_samples:
        xt = torch.randn((args.batch_size, args.image_size, 2)).to(dist_util.dev())
        xt_2 = torch.randn((args.batch_size, int(args.image_size), 2)).to(dist_util.dev())
        print(f"{len(all_images) / args.num_samples}%")
        x, pen_state = dpm_solver.sample(xt, xt_2, steps=30, order=3)
        # softmax
        pen_state = torch.softmax(pen_state, dim=2)

        sample_all = (th.cat((x, pen_state), 2).detach().cpu())
        all_images.extend(sample_all.cpu().numpy())

    for penbreak in break_list:
        name = f'mutli_fish_{penbreak}'
        if not os.path.exists(f"{args.save_path}/{name}"):
            os.mkdir(f"{args.save_path}/{name}")
        npz = []
        epoch_images = bin_pen(np.array(all_images), penbreak)
        for sample in epoch_images:
            sketch_cv = draw_three(np.array(sample), img_size=40)
            cv2.imwrite(f"{args.save_path}/{name}/{len(npz)}.jpg", sketch_cv)
            npz.append(sketch_cv)
        npz = np.array(npz)
        np.savez(os.path.join(args.save_path, f'{name}.npz'), npz)
        print(f'{name}.npz')


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=2000,
        batch_size=1000,
        use_ddim=False,
        model_path="../pre-model/192/1/model035000_fish.pt",  # ./save_model/
        model_path_2="../pre-model/192/1/model035000_fish.pt",
        save_path='../dataset/xr/dpm_scale',  #
        log_dir='../logs',
        pen_break=[0.8],

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
# python evaluation.py D:\2023_9\SketchKnitter-nyx\dataset\xr\dpm\xr_dpm_0.8.npz D:\2023_9\SketchKnitter-pro\dataset\real_data\sketchrnn_apple_40.npz
