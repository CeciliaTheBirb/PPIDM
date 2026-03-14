"""
Sample video reconstructions from a diffusion model and save:
1) generated samples (npz)
2) ground truth videos (npz)

No metric logging. No start_j_list. Uses a single --start_j.
"""

import argparse
import logging
import os
import random
from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import torch as th
import torchvision.transforms as transforms

from diffusion_openai.video_datasets_nc_sample import load_data
from diffusion_openai.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def setup_logger() -> logging.Logger:
    lg = logging.getLogger("ppidm_sample")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        lg.addHandler(h)
    return lg


def set_seed(seed: int) -> None:
    if seed is None or seed == 0:
        return
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_int_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def to_uint8_video(x: th.Tensor) -> th.Tensor:
    """
    x: (B, C, T, H, W), in [-1, 1]
    returns: (B, T, H, W, C) uint8
    """
    x = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return x.permute(0, 2, 3, 4, 1).contiguous()


def save_npz(out_dir: str, name: str, arr: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, name), arr)


def build_data_files(args) -> Tuple[List[str], List[str]]:
    if not args.chl_dir or not args.u_dir or not args.v_dir:
        raise ValueError("Missing paths. Provide --chl_file, --u_file, --v_file.")
    return [args.chl_dir, args.u_dir, args.v_dir], ["BLGCHL", "UVEL", "VVEL"]


def prepare_condition_kwargs(
    chl: th.Tensor,
    u: th.Tensor,
    v: th.Tensor,
    device: th.device,
    cond_generation: bool,
    cond_frames: List[int],
    seq_len: int,
) -> Tuple[Dict, List[int]]:
    unknown_frames = [i for i in range(seq_len) if i not in cond_frames] if cond_generation else list(range(seq_len))
    cond_kwargs = {
        "cond_img": [],
        "unknown_img": [],
        "img": [],
        "u_img": [],
        "v_img": [],
        "cond_frames": cond_frames,
        "unknown_frames": unknown_frames,
    }

    for t in range(chl.shape[1]):
        frame = chl[:, t, :, :, :]
        cond_kwargs["img"].append(frame)
        if cond_generation:
            if t in cond_frames:
                cond_kwargs["cond_img"].append(frame)
            elif t in unknown_frames:
                cond_kwargs["unknown_img"].append(frame)

    for t in range(u.shape[1]):
        cond_kwargs["u_img"].append(u[:, t, :, :, :])

    for t in range(v.shape[1]):
        cond_kwargs["v_img"].append(v[:, t, :, :, :])

    cond_kwargs["img"] = th.stack(cond_kwargs["img"], dim=2).to(device)
    cond_kwargs["u_img"] = th.stack(cond_kwargs["u_img"], dim=2).to(device)
    cond_kwargs["v_img"] = th.stack(cond_kwargs["v_img"], dim=2).to(device)

    if cond_generation:
        if len(cond_kwargs["cond_img"]) == 0:
            raise ValueError("cond_generation=True but no cond frames were given. Set --cond_frames.")
        cond_kwargs["cond_img"] = th.stack(cond_kwargs["cond_img"], dim=2).to(device)
        cond_kwargs["unknown_img"] = th.stack(cond_kwargs["unknown_img"], dim=2).to(device)
    else:
        cond_kwargs["unknown_img"] = cond_kwargs["img"]

    return cond_kwargs



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        use_ddim=False,
        batch_size=1,
        num_samples=1,
        seed=0,
        rgb=False,

        chl_dir="",
        u_dir="",
        v_dir="",

        seq_len=20,
        image_size=64,
        class_cond=False,

        cond_generation=True,
        cond_frames="",
        resample_steps=1,

        save_gt=True,
        out_dir="results/samples",
        gt_out_dir="results/gt",
        deterministic=True,

        model_paths="",

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    logger = setup_logger()
    set_seed(args.seed)

    if not args.model_paths:
        raise ValueError("Provide --model_paths as comma-separated checkpoint paths.")
    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip() != ""]

    cond_frames = parse_int_list(args.cond_frames)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_files, var_names = build_data_files(args)

    logger.info("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(device)
    model.eval()

    logger.info(f"loading data...")
    data_iter = load_data(
        data_file=data_files,
        variable_name=var_names,
        batch_size=1,
        image_size=args.image_size,
        class_cond=args.class_cond,
        transforms=transforms.Compose([transforms.Normalize((0.5,), (0.5,))]),
        deterministic=args.deterministic,
        seq_len=args.seq_len,
    )
    chl_array, u_array, v_array = next(iter(data_iter))
    logger.info("data loaded")

    for ckpt in model_paths:
        logger.info(f"loading checkpoint: {ckpt}")
        state = th.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state, strict=True)
        except TypeError:
            model.load_state_dict(state)
        model.to(device)
        model.eval()

        model_name = os.path.basename(os.path.dirname(ckpt)) or os.path.basename(ckpt)

        for trial in range(args.num_samples):
            cond_kwargs= prepare_condition_kwargs(
                chl=chl_array,
                u=u_array,
                v=v_array,
                device=device,
                cond_generation=args.cond_generation,
                cond_frames=cond_frames,
                seq_len=args.seq_len,
            )

            mask = th.zeros_like(cond_kwargs["unknown_img"], device=device)
            cond_kwargs["unknown_img"] = cond_kwargs["unknown_img"] * mask

            sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop


            logger.info(f"sampling (model={model_name}, trial={trial})...")
            sample = sample_fn(
                model,
                (args.batch_size, 3 if args.rgb else 1, args.seq_len, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                progress=True,
                cond_kwargs=cond_kwargs,
            )

            sample_u8 = to_uint8_video(sample).cpu().numpy()
            save_npz(args.out_dir, f"{model_name}_trial{trial}.npz", sample_u8)

            if args.save_gt:
                gt_u8 = to_uint8_video(cond_kwargs["img"]).cpu().numpy()
                save_npz(args.gt_out_dir, f"{args.start_j}.npz", gt_u8)

    logger.info("done")


if __name__ == "__main__":
    main()