import torch
import torch.nn.functional as F
import numpy as np

import math

from PIL import Image

from .general_utils import rgb_to_luma


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    img.index_put_((pys, pxs), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
    img.index_put_((pys, pxs + 1), weights * dxs * (1.0 - dys), accumulate=True)
    img.index_put_((pys + 1, pxs), weights * (1.0 - dxs) * dys, accumulate=True)
    img.index_put_((pys + 1, pxs + 1), weights * dxs * dys, accumulate=True)
    return img


def events_to_image_torch(
    xs, ys, ps, h, w, clip_out_of_range=True, interpolation=None, padding=True
):
    device = xs.device
    if interpolation == "bilinear" and padding:
        img_size = (h + 1, w + 1)
    else:
        img_size = (h, w)

    mask = torch.ones(xs.size(), device=device)

    if clip_out_of_range:
        zero_v = torch.tensor([0.0], device=device)
        ones_v = torch.tensor([1.0], device=device)
        if interpolation is None and not padding:
            clipx = img_size[1]
            clipy = img_size[0]
        else:
            clipx = img_size[1] - 1
            clipy = img_size[0] - 1
        mask = torch.where(xs >= clipx, zero_v, other=ones_v) * torch.where(
            ys >= clipy, zero_v, ones_v
        )

    img = torch.zeros(img_size).to(device)
    if (
        interpolation == "bilinear"
        and xs.dtype is not torch.long
        and ys.dtype is not torch.long
    ):
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs - pxs).float()
        dys = (ys - pys).float()
        pxs = (pxs * mask).long()
        pys = (pys * mask).long()
        masked_ps = ps.squeeze() * mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        num_events = xs.shape[0]

        img.index_put_((ys, xs), ps, accumulate=True)
    return img


def events_to_voxel(xs, ys, ts, ps, combine_channel, h, w):
    zero_v = torch.tensor([0.0])
    ones_v = torch.tensor([1.0])
    if not combine_channel:
        pos_weights = torch.where(ps > 0, ones_v, zero_v)
        neg_weights = torch.where(ps <= 0, ones_v, zero_v)
        voxel_pos = events_to_image_torch(
            xs, ys, pos_weights, h, w, clip_out_of_range=False
        )
        voxel_neg = events_to_image_torch(
            xs, ys, neg_weights, h, w, clip_out_of_range=False
        )
        return torch.cat([voxel_pos.unsqueeze(0), voxel_neg.unsqueeze(0)], dim=0)
    else:
        return events_to_image_torch(xs, ys, ps, h, w, clip_out_of_range=False)


def event_to_vis(events, max=None):
    events = events.detach().cpu()
    if max is None:
        max = torch.abs(events).max()

    img = torch.zeros_like(events).unsqueeze(0).repeat(3, 1, 1)
    mag = torch.clip(torch.abs(events), max=max)

    img[0, :, :] = (events > 0).float()
    img[2, :, :] = (events < 0).float()

    pos_add = (events > 0).float() * (1 - mag / max)
    img[1, :, :] += pos_add
    img[2, :, :] += pos_add

    neg_add = (events < 0).float() * (1 - mag / max)
    img[0, :, :] += neg_add
    img[1, :, :] += neg_add

    img = torch.where(events == 0, 1, img)
    img = Image.fromarray((img * 255).permute(1, 2, 0).numpy().astype(np.uint8))

    return img


def linlog(x, thr=20):
    mask = (x > thr / 255.0).float()
    return torch.log(x * 255) * mask + (1 - mask) * x * 255 / thr * math.log(thr)


def inv_linlog(x, thr=20):
    return torch.clamp(
        torch.where(
            x > math.log(thr), torch.exp(x) / 255, x / math.log(thr) * thr / 255
        ),
        min=0.0,
        max=1.0,
    )

def event_from_image(img, next_img, thr=0.2): 
    img = rgb_to_luma(img)
    next_img = rgb_to_luma(next_img)

    return (linlog(next_img) - linlog(img)) / thr