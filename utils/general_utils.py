import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import random
import yaml

from pathlib import Path


def tolist(x):
    if x == "[]":
        return []
    try:
        return [int(itm) for itm in x[1:-1].split(",")]
    except:
        try:
            return [float(itm) for itm in x[1:-1].split(",")]
        except:
            return [itm for itm in x[1:-1].split(",")]


def tobool(x):
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False
    else:
        return None


def dict_to_args(dic, args=None):
    if args is None:
        args = argparse.Namespace()
    for k, v in dic.items():
        if isinstance(v, dict):
            if k in args:
                setattr(args, k, dict_to_args(v, args=getattr(args, k)))
            else:
                setattr(args, k, dict_to_args(v))
        else:
            setattr(args, k, v)
    return args


def args_to_parser(args, parser, prefix=""):
    args = vars(args)
    for key in args:
        if isinstance(args[key], argparse.Namespace):
            parser = args_to_parser(args[key], parser, prefix=prefix + key + ".")
        else:
            if isinstance(args[key], list):
                parser.add_argument(
                    "--{}{}".format(prefix, key), type=tolist, default=args[key]
                )
            elif isinstance(args[key], bool):
                parser.add_argument(
                    "--{}{}".format(prefix, key), type=tobool, default=args[key]
                )
            elif args[key] is None:
                parser.add_argument(
                    "--{}{}".format(prefix, key), type=float, default=args[key]
                )
            else:
                parser.add_argument(
                    "--{}{}".format(prefix, key),
                    type=type(args[key]),
                    default=args[key],
                )
    return parser


def sync_args(new_args, old_args):
    new_args = vars(new_args)
    for keys in new_args:
        curr_arg = old_args
        for key in keys.split(".")[:-1]:
            curr_arg = getattr(curr_arg, key)
        curr_arg.__setattr__(keys.split(".")[-1], new_args[keys])
    return old_args


def load_args(config_file, args=None):
    config_path = Path(os.path.abspath(config_file))

    config_files = [config_path]
    while config_path is not None:
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        if config_dict is not None and "base_config" in config_dict:
            config_path = Path(
                os.path.abspath(
                    os.path.join(config_path.parent, config_dict["base_config"])
                )
            )
            config_files = [config_path] + config_files
        else:
            config_path = None

    for config in config_files:
        with open(config) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            if config_dict is not None:
                args = dict_to_args(config_dict, args)

    args.__delattr__("base_config")

    parser = argparse.ArgumentParser()
    parser = args_to_parser(args, parser)
    new_args = parser.parse_args()
    args = sync_args(new_args, args)

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_args(args, fname):
    with open(fname, "w") as f:
        args_list = [("", args)]
        while len(args_list) != 0:
            prefix, curr_args = args_list.pop(0)
            curr_args = vars(curr_args)
            for key in curr_args:
                if isinstance(curr_args[key], argparse.Namespace):
                    args_list.append(("{}{}.".format(prefix, key), curr_args[key]))
                else:
                    f.write("{}{}: {}\n".format(prefix, key, curr_args[key]))


def config_output(expname):
    output_path = os.path.join("./output/", expname)
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    tb_writer = SummaryWriter(output_path)
    return output_path, tb_writer


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [batch, c, h-1, w]
    second_difference = (
        first_difference[..., 1:, :] - first_difference[..., : h - 2, :]
    )  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rgb_to_luma(img):
    if len(img.shape) == 3:
        return 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    elif len(img.shape) == 4:
        return (
            0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        )

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5