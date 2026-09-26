import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Type, Dict, Tuple, Optional
from collections import defaultdict
import os
import math
import argparse
import time
import logging
from datetime import datetime

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from clip.clip import _transform
from timm.utils import accuracy

import sot_glp.lib as lib
import sot_glp.models.tools as vlp_tools
import sot_glp.datasets.tools as dts_tools
from sot_glp.datasets import return_train_val_datasets, return_ood_loaders, return_domains_loaders
from sot_glp.models import SOTGLP
from sot_glp.models.tools import GLSotLoss
#torch.autograd.set_detect_anomaly(True)

NoneType = Type[None]


# ============================================================
# Progress logger: writes to a dedicated txt file, flushes each line
# ============================================================
def setup_progress_logger(save_dir: str, exp_name: str = None) -> logging.Logger:
    """
    Creates a dedicated logger that writes training progress to
    <save_dir>/training_progress.log. Always flushed, so you can `tail -f` it live.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_progress.log")

    logger = logging.getLogger("progress")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't duplicate into the root logger

    # remove old handlers (in case setup is called twice)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info(f"Progress log started. exp_name={exp_name}, path={log_path}")
    logger.info("=" * 80)
    return logger


def train_one_epoch(
    model: SOTGLP,
    train_loader: DataLoader,
    loss_fn: GLSotLoss,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    fp16_scaler: GradScaler,
    args: argparse.Namespace,
    progress_logger: logging.Logger = None,
) -> lib.DictAverage:
    meter = lib.DictAverage()

    class_names = train_loader.dataset.all_names
    if not args.learn_global_prompt and not args.learn_local_prompts:
        with torch.no_grad(), autocast("cuda", torch.float16):
            text_features, local_text_features = model.encode_text(class_names)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = local_text_features = None

    accum_steps = getattr(args, "accum_steps", 1)
    num_batches = len(train_loader)
    num_optim_steps = math.ceil(num_batches / accum_steps)

    model.train()
    optimizer.zero_grad()

    pbar = tqdm(
        enumerate(train_loader),
        total=num_batches,
        desc=f"Epoch {epoch}/{args.max_epoch - 1}",
        dynamic_ncols=True,
        leave=True,
    )

    epoch_start = time.time()

    # log epoch start
    if progress_logger is not None:
        progress_logger.info(
            f"[EPOCH {epoch}] START | num_batches={num_batches} | "
            f"accum_steps={accum_steps} | num_optim_steps={num_optim_steps}"
        )

    for i, batch in pbar:
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        with autocast("cuda", torch.float16):
            global_logits, local_logits = model(images, class_names, text_features, local_text_features)
            loss, global_loss, local_loss = loss_fn(global_logits, local_logits, targets, model.logit_scale.exp())

        # 🔑 scale the loss for gradient accumulation
        loss_for_backward = loss / accum_steps
        fp16_scaler.scale(loss_for_backward).backward()

        # 🔑 step only every accum_steps batches (or on the last batch)
        is_update_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
        if is_update_step:
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

        gl_probs, global_probs, local_probs = model.create_prediction_scores_last(global_logits, local_logits)
        topk = accuracy(gl_probs, targets, topk=(1,))
        global_topk = accuracy(global_probs, targets, topk=(1,))

        meter.update(
            {
                "loss": loss.detach().item(),
                "global_loss": global_loss.detach().item(),
                "local_loss": local_loss.detach().item(),
                "top1": topk[0],
                "top1_global": global_topk[0],
            },
            images.size(0),
        )

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            meter.update({"top1_local": local_topk[0]}, images.size(0))

        # 🔑 live postfix on the progress bar
        postfix = {
            "loss": f"{loss.detach().item():.3f}",
            "top1": f"{topk[0].item():.1f}",
            "gpu": f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
        }
        if local_probs is not None:
            postfix["top1_loc"] = f"{local_topk[0].item():.1f}"
        pbar.set_postfix(postfix)

        # 🔑 log every optimizer step (accum_steps batches)
        if progress_logger is not None and is_update_step:
            step_idx = (i + 1) // accum_steps
            lr_now = optimizer.param_groups[0]["lr"]
            progress_logger.info(
                f"[EPOCH {epoch}] STEP {step_idx}/{num_optim_steps} "
                f"(batch {i+1}/{num_batches}) | "
                f"loss={loss.detach().item():.4f} "
                f"global={global_loss.detach().item():.4f} "
                f"local={local_loss.detach().item():.4f} | "
                f"top1={topk[0].item():.2f} "
                f"top1_global={global_topk[0].item():.2f}"
                + (f" top1_local={local_topk[0].item():.2f}" if local_probs is not None else "")
                + f" | lr={lr_now:.6f} "
                f"| gpu={torch.cuda.max_memory_allocated() / 1024**3:.2f}G"
            )

    pbar.close()

    epoch_time = time.time() - epoch_start
    speed = num_batches / epoch_time if epoch_time > 0 else 0.0

    summary = meter.summary()
    summary_line = f"Epoch {epoch} done in {epoch_time:.1f}s ({speed:.2f} it/s) | " + " ".join(summary)
    print(summary_line, flush=True)

    if progress_logger is not None:
        progress_logger.info(
            f"[EPOCH {epoch}] DONE in {epoch_time:.1f}s "
            f"({speed:.2f} it/s, {epoch_time/60:.2f} min) | " + " ".join(summary)
        )

    lr_scheduler.step()
    return meter


@torch.no_grad()
def evaluate(
    model: SOTGLP,
    val_loader: DataLoader,
    class_names,
    args: argparse.Namespace,
    return_scores: bool = False,
    progress_logger: logging.Logger = None,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    class_names_original = val_loader.dataset.all_names

    with autocast("cuda", torch.float16):
        text_features, local_text_features = model.encode_text(class_names)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
        same_order = (class_names_original == class_names)
        if not same_order:
            name2idx = {name: i for i, name in enumerate(class_names)}
            idx = np.fromiter((name2idx[name] for name in class_names_original), dtype=np.int64)
            local_text_features = local_text_features[idx, ...]
            text_features = text_features[idx, ...]

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]

    pbar = tqdm(
        val_loader,
        total=len(val_loader),
        desc=f"Evaluating on {dataset_name}",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in pbar:
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)

        with autocast("cuda", torch.float16):
            global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
            if return_scores:
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        gl_probs, global_probs, local_probs = model.create_prediction_scores_last(global_logits, local_logits)
        global_topk = accuracy(global_probs, targets, topk=(1,))

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            topk = accuracy(gl_probs, targets, topk=(1,))
            logs = {"top1": topk[0], "top1_global": global_topk[0], "top1_local": local_topk[0]}
        else:
            logs = {"top1": global_topk[0], "top1_global": global_topk[0]}

        meter.update(logs, images.size(0))
        pbar.set_postfix({"top1": f"{logs['top1'].item():.1f}"})

    pbar.close()
    model.train(mode)

    if progress_logger is not None:
        progress_logger.info(f"[EVAL {dataset_name}] " + " ".join(meter.summary()))

    return meter, test_scores


@torch.no_grad()
def evaluate_ood(
    model: SOTGLP,
    val_loader: DataLoader,
    ood_loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    test_scores: Optional[np.ndarray] = None,
    progress_logger: logging.Logger = None,
) -> lib.DictAverage:
    metrics = defaultdict(dict)

    class_names = val_loader.dataset.all_names

    with autocast("cuda", torch.float16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    if test_scores is None:
        test_scores = np.zeros(len(val_loader.dataset))
        pbar = tqdm(val_loader, total=len(val_loader), desc="Computing ood scores for Test", dynamic_ncols=True, leave=False)
        for batch in pbar:
            images = batch["image"].cuda(non_blocking=True)
            with autocast("cuda", torch.float16):
                global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)
        pbar.close()

    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = np.zeros(len(ood_loader.dataset))
        pbar = tqdm(ood_loader, total=len(ood_loader), desc=f"Computing ood scores for {ood_name}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            images = batch["image"].cuda(non_blocking=True)
            with autocast("cuda", torch.float16):
                global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                ood_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)
        pbar.close()

        metrics[ood_name]["fpr95"] = lib.get_fpr(test_scores, ood_scores)
        metrics[ood_name]["auroc"] = lib.get_auroc(test_scores, ood_scores)

    model.train(mode)
    return metrics


if __name__ == "__main__":
    clip_model_names = [
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_resnet50",
        "clip_resnet101",
    ]

    parser = argparse.ArgumentParser("Learning prompts for CLIP with local and global features")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_dir", default="/share/DEEPLEARNING/datasets", type=str)
    parser.add_argument("--save_dir", default="./results/", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--eval_only", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_ood", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_domains", default=False, type=lib.boolean_flags)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_shots", default=16, type=int, help="Number of shots by class. -1 means the whole dataset")
    parser.add_argument("--use_local_features", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_global_loss", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_local_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--topk", default=[5, 10, 15, 20], type=int, nargs="+")
    parser.add_argument("--learn_local_proj", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_global_prompt", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--n_global_prompts", default=1, type=int)
    parser.add_argument("--n_local_prompts", default=1, type=int)
    parser.add_argument("--global_dropout_p", default=0.75, type=lib.float_range(0.0, 1.0))

    parser.add_argument("--prompts_batch_size", default=math.inf, type=int)

    parser.add_argument("--parallel_text_encoder", default=False, type=lib.boolean_flags)
    parser.add_argument("--parallel_vision_encoder", default=False, type=lib.boolean_flags)

    parser.add_argument("--ood_method", default="GL-MCM", type=str)
    parser.add_argument("--init_method", default="random", type=str)
    parser.add_argument("--ood_temp_scale", default=1000.0, type=float)

    parser.add_argument("--clip_name", required=True, choices=clip_model_names, type=str)

    # ---- Gradient accumulation related args ----
    parser.add_argument("--accum_batch_size", default=32, type=int,
                        help="Physical batch size pushed to the GPU per forward/backward.")
    parser.add_argument("--original_batch_size", default=None, type=int,
                        help="Logical/effective batch size. accum_steps = original_batch_size // accum_batch_size. "
                             "If None or <= accum_batch_size, accum_steps = 1.")

    parser.add_argument("--inference_batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr_init", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--cons_lr", default=1e-5, type=float)

    parser.add_argument("--use_fp16", default=True, type=lib.boolean_flags)
    parser.add_argument("--persistent_workers", default=False, type=lib.boolean_flags)
    parser.add_argument("--checkpointing_segments", default=4, type=int, help="Number of segments used for gradient checkpointing for the text encoder.")

    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    args = parser.parse_args()

    # ---- Gradient accumulation setup ----
    if args.original_batch_size is None or args.original_batch_size <= args.accum_batch_size:
        args.accum_steps = 1
        args.original_batch_size = args.accum_batch_size
    else:
        args.accum_steps = args.original_batch_size // args.accum_batch_size
        realized = args.accum_batch_size * args.accum_steps
        if realized != args.original_batch_size:
            print(f"[warn] original_batch_size={args.original_batch_size} is not a multiple of "
                  f"accum_batch_size={args.accum_batch_size}. Using {realized} instead "
                  f"(accum_steps={args.accum_steps}).")
            args.original_batch_size = realized

    print(f"[info] accum_batch_size={args.accum_batch_size}, "
          f"accum_steps={args.accum_steps}, "
          f"original_batch_size={args.original_batch_size}", flush=True)

    lib.setup_logger(args.exp_name)
    lib.random_seed(args.seed)

    if args.exp_name is not None:
        lib.LOGGER.info(f"Running experiment {args.exp_name}")
        args.save_dir = os.path.join(args.save_dir, args.exp_name)

    # ---- Progress logger (dedicated txt file) ----
    progress_logger = setup_progress_logger(args.save_dir, args.exp_name)
    progress_logger.info(
        f"Config: dataset={args.dataset_name} | clip={args.clip_name} | "
        f"num_shots={args.num_shots} | max_epoch={args.max_epoch} | "
        f"accum_batch_size={args.accum_batch_size} | "
        f"accum_steps={args.accum_steps} | "
        f"original_batch_size={args.original_batch_size} | "
        f"lr_init={args.lr_init} | use_fp16={args.use_fp16}"
    )

    args.eval_domains = args.eval_domains and (args.dataset_name == "imagenet")
    args.eval_ood = args.eval_ood and (args.dataset_name == "imagenet")

    # seting-up transforms
    train_transform = dts_tools.get_train_transform()
    val_transform = _transform(224)

    # Setting-up Imagenet dataset train
    train_dataset, val_dataset, template = return_train_val_datasets(args.dataset_name, args.data_dir, train_transform, val_transform)
    template = "A photo of a {}" if (args.learn_global_prompt or args.learn_local_prompts) else template

    train_dataset = dts_tools.create_few_shots_dataset(train_dataset, args.num_shots, seed=args.seed)
    lib.LOGGER.info("Using template: " + template.format("<class_name>"))

    # Setting-up dataloaders
    train_loader = dts_tools.get_train_loader(
        train_dataset,
        batch_size=args.accum_batch_size,
        num_workers=10,
        persistent_workers=args.persistent_workers,
    )
    val_loader = dts_tools.get_eval_loader(val_dataset, batch_size=args.inference_batch_size)

    progress_logger.info(
        f"DataLoader ready: train_samples={len(train_dataset)} | "
        f"train_batches={len(train_loader)} | val_samples={len(val_dataset)}"
    )

    if args.eval_ood:
        ood_loaders = return_ood_loaders(args.data_dir, val_transform)

    if args.eval_domains:
        domains_loaders = return_domains_loaders(args.data_dir, val_transform)

    # Setting-up model
    model = SOTGLP(
        clip_name=args.clip_name,
        use_local_features=args.use_local_features,
        checkpointing_segments=args.checkpointing_segments,
        template=template,
        learn_local_proj=args.learn_local_proj,
        learn_local_prompts=args.learn_local_prompts,
        learn_global_prompt=args.learn_global_prompt,
        class_names=train_dataset.all_names,
        n_global_prompts=args.n_global_prompts,
        n_local_prompts=args.n_local_prompts,
        prompts_batch_size=args.prompts_batch_size,
        ood_method=args.ood_method,
        ood_temp_scale=args.ood_temp_scale,
        topk=args.topk,
        parallel_text_encoder=args.parallel_text_encoder,
        parallel_vision_encoder=args.parallel_vision_encoder,
        init_method=args.init_method,
    )

    model.initialize_prompt()

    lib.load_checkpoint(model, args.checkpoint_path)
    print("Freezed Clip")
    model.v2v_use()
    model.freeze_clip()
    model = model.cuda()

    loss_fn = GLSotLoss(
        use_global_loss=args.use_global_loss,
        use_local_loss=args.use_local_loss,
        topk=args.topk,
        global_dropout_p=args.global_dropout_p,
    )

    optimizer = vlp_tools.get_optimizer(args.optimizer, model, args.lr_init, args.weight_decay, args.momentum)

    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    if args.warmup_epoch > 0:
        lr_scheduler = vlp_tools.ConstantWarmupScheduler(optimizer, lr_scheduler, args.warmup_epoch, args.cons_lr)

    fp16_scaler = GradScaler("cuda", enabled=args.use_fp16)

    progress_logger.info(f"Model+Optimizer ready. Starting training for {args.max_epoch} epochs.")

    # Training loop
    for epoch in range(args.max_epoch):
        if not args.eval_only:
            assert args.use_local_loss or args.use_global_loss or args.learn_local_prompts or args.learn_global_prompt, "At least one of use_local_loss or use_global_loss or learn_local_prompts or learn_global_prompt must be True"
            train_meter = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                fp16_scaler=fp16_scaler,
                args=args,
                progress_logger=progress_logger,
            )

            lib.save_checkpoint(args.save_dir, epoch, model, optimizer, lr_scheduler, fp16_scaler, train_meter, args)

        if ((epoch % args.eval_freq == 0) and (epoch > 0)) or (epoch + 1 == args.max_epoch) or args.eval_only:
            lib.LOGGER.info("Evaluation")
            val_meter, test_scores = evaluate(
                model, val_loader, train_loader.dataset.all_names, args,
                return_scores=args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)),
                progress_logger=progress_logger,
            )
            lib.LOGGER.info("Evaluation metrics: " + " ".join([" *"] + val_meter.summary()))
            progress_logger.info(f"[EPOCH {epoch}] EVAL top1={val_meter.avg.get('top1', -1):.2f}")

            if args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)):
                ood_metrics = evaluate_ood(model, val_loader, ood_loaders, args, test_scores=test_scores, progress_logger=progress_logger)
                lib.LOGGER.info(f"OOD Evaluation metrics with temperature scale {args.ood_temp_scale} (FPR95 / AUROC): ")
                lib.log_ood_metrics(ood_metrics)

            if args.eval_domains and (args.eval_only or (epoch + 1 == args.max_epoch)):
                metrics = {}
                for domain_name, domain_loader in domains_loaders.items():
                    metrics[domain_name], _ = evaluate(model, domain_loader, args, progress_logger=progress_logger)
                    lib.LOGGER.info(f"Evaluation metrics for {domain_name}: " + " ".join([" *"] + metrics[domain_name].summary()))
                avg_top1 = np.mean([metrics[domain_name].avg["top1"] for domain_name in domains_loaders.keys()])
                lib.LOGGER.info(f"Average evaluation metrics for domains: * top1: {avg_top1: .3f}")

            if args.eval_only:
                break

    progress_logger.info("Training finished.")