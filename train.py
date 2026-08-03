import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import random
import warnings

from argparse import ArgumentParser
from tqdm import tqdm

from data import get_data
from models import get_model
from gaussian_renderer import render
from utils.event_utils import event_from_image
from utils.general_utils import load_args, setup_seed, save_args, config_output
from utils.loss_utils import l1, ssim, tv, eloss
from utils.image_utils import psnr, render_training_image
from utils.timer import Timer


def training_report(
    gaussians,
    writer,
    iteration,
    losses,
    elapsed,
    train_dataset,
    test_dataset,
    render,
    background,
    stage,
    args,
):
    for key in losses:
        writer.add_scalar(f"{stage}/train_loss/{key}", losses[key], iteration)
    writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in args.testing_iterations:
        num_test = 10

        train_idx = [int((idx + 0.5) * (len(train_dataset) // num_test)) for idx in range(num_test)]
        train_cams = []
        for idx in train_idx:
            curr_idx = idx
            while train_dataset[curr_idx].image is None:
                curr_idx += 1
                if curr_idx >= len(train_dataset):
                    curr_idx = 0
            train_cams.append(train_dataset[curr_idx])

        validation_configs = (
            {
                "name": "test",
                "cameras": [
                    test_dataset[int((idx + 0.5) * (len(test_dataset) // num_test))]
                    for idx in range(num_test)
                ],
            },
            {
                "name": "train",
                "cameras": train_cams,
            },
        )

        for config in validation_configs:
            l1_test = 0.0
            ssim_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config["cameras"]):
                image = torch.clamp(
                    render(viewpoint, gaussians, background, stage, args)["render"],
                    0.0,
                    1.0,
                )
                gt_image = torch.clamp(viewpoint.image.to("cuda"), 0.0, 1.0)
                writer.add_images(
                    stage
                    + "/"
                    + config["name"]
                    + "_view_{}/render".format(viewpoint.image_name),
                    image[None],
                    global_step=iteration,
                )
                if iteration == args.testing_iterations[0]:
                    writer.add_images(
                        stage
                        + "/"
                        + config["name"]
                        + "_view_{}/ground_truth".format(viewpoint.image_name),
                        gt_image[None],
                        global_step=iteration,
                    )

                l1_test += l1(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image, mask=None).mean().double()

            psnr_test /= len(config["cameras"])
            ssim_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            print(
                "\n[ITER {}] Evaluating {}: L1 {} SSIM {} PSNR {}".format(
                    iteration, config["name"], l1_test, ssim_test, psnr_test
                )
            )
            writer.add_scalar(
                stage + "/" + config["name"] + "/l1_loss", l1_test, iteration
            )
            writer.add_scalar(
                stage + "/" + config["name"] + "/ssim", ssim_test, iteration
            )
            writer.add_scalar(
                stage + "/" + config["name"] + "/psnr", psnr_test, iteration
            )

        writer.add_scalar(
            f"{stage}/total_points", gaussians.get_xyz.shape[0], iteration
        )


def train(
    stage, args, gaussians, train_dataset, test_dataset, output_dir, writer, timer
):
    gaussians.build_opt(args.optimize)

    first_iter = 0
    if args.checkpoint_file is not None:
        if stage == "coarse" and stage not in args.checkpoint_file:
            print("Start from fine stage, skip coarse stage.")
            return
        if stage in args.checkpoint_file:
            first_iter = gaussians.load(args.checkpoint_file, args.optimize)

    bg_color = [1, 1, 1] if args.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_eloss_for_log = 0.0

    if stage == "coarse":
        final_iter = args.optimize.coarse_iterations
    else:
        final_iter = args.optimize.fine_iterations

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=list,
    )
    train_iter = iter(train_loader)

    psnr_ = None

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        gaussians.update(iteration)

        try:
            viewpoint_cams = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            viewpoint_cams = next(train_iter)

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        events = []
        gt_events = []
        reg_losses = []
        color_reg_losses = []

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, background, stage, args)
            image, viewspace_point_tensor, visibility_filter, radii, depth = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
            )

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)


            if viewpoint_cam.image is not None:
                images.append(image.unsqueeze(0))
                gt_image = viewpoint_cam.image.cuda()
                gt_images.append(gt_image.unsqueeze(0))

            if viewpoint_cam.event_images is not None:
                event_idx = random.randint(0, viewpoint_cam.event_images.size(1) - 1)
                pkg_next = render(
                    viewpoint_cam, gaussians, background, stage, args, event_idx=event_idx,
                )
                image_next = pkg_next['render']
                events.append(event_from_image(image, image_next).unsqueeze(0))

                event_mask = F.avg_pool2d(torch.sum(viewpoint_cam.event_images[:,:event_idx+1,:,:].cuda(), dim=0, keepdim=True), kernel_size=3, stride=1, padding=1) > 0.2

                gt_event = ((
                    viewpoint_cam.event_images[0, :event_idx+1, :, :]
                    - viewpoint_cam.event_images[1, :event_idx+1, :, :]
                ).cuda() * event_mask.squeeze(0)).sum(dim=0)

                gt_events.append(gt_event.unsqueeze(0))
                if pkg_next['reg_loss'] is not None:
                    reg_losses.append(pkg_next['reg_loss'])
                
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        if len(images) != 0:
            image_tensor = torch.cat(images, 0)
        else:
            image_tensor = None
        if len(gt_images) != 0:
            gt_image_tensor = torch.cat(gt_images, 0)
        else:
            gt_image_tensor = None
        if len(events) != 0:
            event_tensor = torch.cat(events, 0)
        else:
            event_tensor = None
        if len(gt_events) != 0:
            gt_event_tensor = torch.cat(gt_events, 0)
        else:
            gt_event_tensor = None
        if len(color_reg_losses) != 0:
            color_reg_loss = sum(color_reg_losses) / len(color_reg_losses)
        else:
            color_reg_loss = torch.tensor(0.).cuda()

        if len(reg_losses) != 0:
            reg_loss = sum(reg_losses) / len(reg_losses)
        else:
            reg_loss = torch.tensor(0.).cuda()

        if gt_image_tensor is not None:
            l1_loss = l1(image_tensor, gt_image_tensor)
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            psnr_ = psnr(image_tensor, img2=gt_image_tensor).mean().double()

            loss = l1_loss
            if args.optimize.lambda_dssim != 0:
                loss += args.optimize.lambda_dssim * (1 - ssim_loss)
        else:
            l1_loss = torch.tensor(0.0).cuda()
            ssim_loss = torch.tensor(0.0).cuda()
            psnr_ = torch.tensor(0.0).cuda()
            loss = l1_loss

        if stage == "fine" and args.optimize.time_smoothness_weight != 0:
            tv_loss = tv(
                gaussians.get_grids(),
                args.optimize.time_smoothness_weight,
                args.optimize.l1_time_planes,
                args.optimize.plane_tv_weight,
            )
            loss += tv_loss
        if stage == "fine" and args.optimize.lambda_event != 0 and event_tensor is not None:
            event_loss = eloss(
                event_tensor,
                gt_event_tensor,
            )
        else:
            event_loss = torch.tensor(0.0).cuda()
        loss += args.optimize.lambda_event * event_loss

        if event_tensor is not None:
            loss += args.optimize.lambda_preg * torch.clamp(reg_loss, max=1.) + args.optimize.lambda_creg * color_reg_loss

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            print(gaussians._xyz.isnan().any(), gaussians._xyz.isinf().any())
            exit()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        try:
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = (
                    viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
                )
        except:
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if psnr_ is not None:
                ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            ema_eloss_for_log = 0.4 * event_loss.item() + 0.6 * ema_eloss_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "PSNR": f"{psnr_:.{2}f}",
                        "ELoss": f"{ema_eloss_for_log:.{4}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            losses = {
                "total_loss": loss.item(),
                "L1": l1_loss.item(),
                "SSIM": ssim_loss.item(),
                "PSNR": psnr_.item(),
                "Event": event_loss.item(),
            }
            training_report(
                gaussians,
                writer,
                iteration,
                losses,
                iter_start.elapsed_time(iter_end),
                train_dataset,
                test_dataset,
                render,
                background,
                stage,
                args,
            )
            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                gaussians.save(output_dir, iteration, stage)
            if args.render_process:
                if iteration % 50 == 49:
                    render_training_image(
                        output_dir,
                        gaussians,
                        [test_dataset[iteration % len(test_dataset)]],
                        render,
                        background,
                        stage + "test",
                        iteration,
                        timer.get_elapsed_time(),
                        args,
                    )
                    render_training_image(
                        output_dir,
                        gaussians,
                        [train_dataset[iteration % len(train_dataset)]],
                        render,
                        background,
                        stage + "train",
                        iteration,
                        timer.get_elapsed_time(),
                        args,
                    )
            timer.start()

            # Densification
            gaussians.densify(
                viewspace_point_tensor_grad,
                visibility_filter,
                radii,
                iteration,
                stage,
                args.densify,
            )

            # Event-based Densification
            viewset = train_dataset.group_by_timestamp(viewpoint_cam.time)
            event_idx = random.randint(0, train_dataset.get_event_size(viewpoint_cam.time) - 1)
            num_added = gaussians.event_densify(
                viewset,
                event_idx,
                viewpoint_cam.event_times[event_idx],
                viewpoint_cam.event_times[event_idx+1],
                depth,
                iteration,
                stage,
                args.ev_densify,
            )
            if num_added != 0:
                print(
                    "Added {} event-based Gaussians on iteration {}, total {}.".format(
                        num_added, iteration, gaussians._xyz.size(0)
                    )
                )

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument(
        "--testing_iterations",
        nargs="+",
        type=int,
        default=[
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            11000,
            12000,
            13000,
            14000,
            15000,
            16000,
            17000,
            18000,
            19000,
            20000,
            21000,
            22000,
            23000,
            24000,
            25000,
            26000,
            27000,
            28000,
            29000,
            30000
        ],
    )
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=list(range(1000, 40000, 500)),
    )
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_known_args()[0]
    args = load_args(args.config_file, args)

    output_dir, writer = config_output(args.expname)
    save_args(args, os.path.join(output_dir, "args.txt"))

    setup_seed(args.seed)
    scene_info, train_dataset, test_dataset, video_dataset = get_data(
        args.data_name, args.data
    )

    gaussians = get_model(scene_info.pcd, scene_info.radius, args.model, args.optimize)

    timer = Timer()
    timer.start()
    train(
        "coarse",
        args,
        gaussians,
        train_dataset,
        test_dataset,
        output_dir,
        writer,
        timer,
    )
    train(
        "fine", args, gaussians, train_dataset, test_dataset, output_dir, writer, timer
    )
    # All done
    print("\nTraining complete.")
