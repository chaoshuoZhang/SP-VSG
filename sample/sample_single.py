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
    # different modes
    create_model_and_diffusion,
    # create_model_and_diffusion_acc
    # create_model_and_diffusion_noise,
    add_dict_to_argparser,
    args_to_dict,
)
import sketch_diffusion.signle_dpmsolver as DPMsolver


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
    model, diffusion = create_model_and_diffusion(
        # model, diffusion = create_model_and_diffusion_acc(
        # model, diffusion = create_model_and_diffusion_noise(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    model.eval()
    ns = DPMsolver.NoiseScheduleVP('discrete', betas=torch.tensor(diffusion.betas, dtype=torch.float32))
    model_fn = DPMsolver.model_wrapper(model, ns)
    logger.log("sampling...")
    all_images = []
    break_list = args.pen_break
    dpm_solver = DPMsolver.DPM_Solver(model_fn, ns, algorithm_type="dpmsolver++")
    xt = torch.randn((args.batch_size, args.image_size, 2)).to(dist_util.dev())
    while len(all_images) < args.num_samples:
        print(len(all_images) / args.num_samples)
        x, pen_state = dpm_solver.sample(xt, steps=15, order=3)
        pen_state = torch.softmax(pen_state, dim=2)
        sample_all = (th.cat((x, pen_state), 2).detach().cpu())
        all_images.extend(sample_all.cpu().numpy())


    for penbreak in break_list:
        name = f''
        if not os.path.exists(f"{args.save_path}/{name}"):
            os.mkdir(f"{args.save_path}/{name}")
        npz = []
        epoch_images = bin_pen(np.array(all_images), penbreak)
        for sample in epoch_images:
            sketch_cv = draw_three(np.array(sample), img_size=256)
            cv2.imwrite(f"{args.save_path}/{name}/{len(npz)}.jpg", sketch_cv)
            npz.append(sketch_cv)
            # npz.append(sample[:, :2])
        npz = np.array(npz)  # .transpose((0, 3, 2, 1))
        print(npz.shape)
        np.savez(os.path.join(args.save_path, f'{name}.npz'), npz)
        # print(f'{name}.npz')
        # np.savez(os.path.join(args.save_path, f'seq.npz'), npz)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=2000,
        batch_size=1000,
        use_ddim=False,
        model_path="../pre-model/",  # ./save_model/
        save_path='../dataset/',  #
        log_dir='',
        pen_break=[0.8],
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
