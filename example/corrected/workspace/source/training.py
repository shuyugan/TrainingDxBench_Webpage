from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from collator import PackedBatchCollator
from common import write_json
from data import ChatDocumentDataset
from model import (
    adapter_parameters,
    load_model,
    prepare_base_model,
    save_adapter,
)
from settings import (
    DOCUMENTS_PER_PACK,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    OPTIMIZER_BETAS,
    OPTIMIZER_EPSILON,
    REFERENCE_WORLD_SIZE,
    SEED,
    TRAIN_EXAMPLES,
    UPDATES,
    WARMUP_UPDATES,
    WEIGHT_DECAY,
    WORLD_SIZE,
)


class ReferencePackBatchSampler:
    def __init__(
        self,
        *,
        examples: int,
        world_size: int,
        rank: int,
        seed: int,
    ) -> None:
        if REFERENCE_WORLD_SIZE % world_size:
            raise RuntimeError(
                f"{world_size} ranks cannot distribute "
                f"{REFERENCE_WORLD_SIZE} reference pack lanes"
            )
        if not 0 <= rank < world_size:
            raise RuntimeError(f"invalid rank {rank} for {world_size} ranks")
        generator = torch.Generator()
        generator.manual_seed(seed)
        order = torch.randperm(examples, generator=generator).tolist()
        lanes = [
            order[lane::REFERENCE_WORLD_SIZE]
            for lane in range(REFERENCE_WORLD_SIZE)
        ]
        expected_lane_examples = UPDATES * DOCUMENTS_PER_PACK
        if any(
            len(indices) != expected_lane_examples
            for indices in lanes
        ):
            raise RuntimeError(
                "reference pack lanes do not match the update schedule"
            )
        self.batches = []
        for update in range(UPDATES):
            start = update * DOCUMENTS_PER_PACK
            stop = start + DOCUMENTS_PER_PACK
            for lane in range(rank, REFERENCE_WORLD_SIZE, world_size):
                self.batches.append(lanes[lane][start:stop])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def set_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def learning_rate_factor(step: int) -> float:
    if step < WARMUP_UPDATES:
        return (step + 1) / WARMUP_UPDATES
    progress = (step - WARMUP_UPDATES) / (UPDATES - WARMUP_UPDATES)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_training_state(
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    adapter_output_dir: Path,
    rank: int,
    local_rank: int,
) -> None:
    if rank == 0:
        save_adapter(model.module, adapter_output_dir)
        checkpoint = output_dir / "checkpoint"
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        checkpoint.mkdir(parents=True)
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        write_json(
            checkpoint / "trainer_state.json",
            {"epoch": 1, "update": UPDATES},
            exclusive=True,
        )
    dist.barrier(device_ids=[local_rank])
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(local_rank),
        },
        output_dir / "checkpoint" / f"rng_rank_{rank}.pt",
    )
    dist.barrier(device_ids=[local_rank])


def run_training(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"training requires {WORLD_SIZE} ranks")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            f"training requires exactly {WORLD_SIZE} visible GPUs"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl", device_id=device, timeout=timedelta(hours=3)
    )
    base_model_dir = args.base_model_dir.resolve()
    output_dir = args.output_dir.resolve()
    adapter_output_dir = base_model_dir.parent / "adapter"
    try:
        if rank == 0:
            prepare_base_model(base_model_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "training_trace.jsonl").unlink(missing_ok=True)
        dist.barrier(device_ids=[local_rank])
        set_determinism()
        model, _ = load_model(
            base_model_dir,
            adapter_dir=None,
            device=device,
        )
        trainable = adapter_parameters(model)
        optimizer = torch.optim.AdamW(
            list(trainable.values()),
            lr=LEARNING_RATE,
            betas=OPTIMIZER_BETAS,
            eps=OPTIMIZER_EPSILON,
            weight_decay=WEIGHT_DECAY,
            fused=False,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=learning_rate_factor
        )
        ddp = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        dataset = ChatDocumentDataset(args.train_records.resolve())
        if len(dataset) != TRAIN_EXAMPLES:
            raise RuntimeError("training document count differs")
        batch_sampler = ReferencePackBatchSampler(
            examples=len(dataset),
            world_size=world_size,
            rank=rank,
            seed=SEED,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=PackedBatchCollator(),
            num_workers=0,
        )
        expected_microbatches = UPDATES * GRADIENT_ACCUMULATION_STEPS
        if len(loader) != expected_microbatches:
            raise RuntimeError("packed microbatch count differs")
        trace = (
            (output_dir / "training_trace.jsonl").open(
                "x", encoding="utf-8"
            )
            if rank == 0
            else None
        )
        try:
            loader_iterator = iter(loader)
            for update in range(1, UPDATES + 1):
                microbatches = []
                local_targets = 0
                for _ in range(GRADIENT_ACCUMULATION_STEPS):
                    batch = next(loader_iterator)
                    inputs = {
                        name: batch[name].to(device)
                        for name in (
                            "input_ids",
                            "position_ids",
                            "attention_mask",
                        )
                        if name in batch
                    }
                    labels = batch["labels"][:, 1:].to(device)
                    local_targets += int((labels != -100).sum())
                    microbatches.append((inputs, labels))
                global_targets = torch.tensor(
                    local_targets, dtype=torch.float64, device=device
                )
                dist.all_reduce(global_targets)
                optimizer.zero_grad(set_to_none=True)
                local_loss = torch.zeros(
                    (), dtype=torch.float64, device=device
                )
                for microbatch_index, (inputs, labels) in enumerate(
                    microbatches
                ):
                    sync_gradients = (
                        microbatch_index
                        == GRADIENT_ACCUMULATION_STEPS - 1
                    )
                    sync_context = (
                        contextlib.nullcontext()
                        if sync_gradients
                        else ddp.no_sync()
                    )
                    with sync_context:
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            logits = ddp(
                                **inputs, use_cache=False
                            ).logits
                        loss_sum = functional.cross_entropy(
                            logits[:, :-1].float().reshape(
                                -1, logits.shape[-1]
                            ),
                            labels.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        (
                            loss_sum
                            * world_size
                            / global_targets.float()
                        ).backward()
                    local_loss += loss_sum.detach().double()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), GRADIENT_CLIP
                )
                if not bool(torch.isfinite(gradient_norm).item()):
                    raise RuntimeError("gradient norm is non-finite")
                learning_rate = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                global_loss = local_loss
                dist.all_reduce(global_loss)
                if rank == 0:
                    trace.write(
                        json.dumps(
                            {
                                "update": update,
                                "epoch": 1,
                                "token_mean_loss": float(
                                    global_loss.item()
                                    / global_targets.item()
                                ),
                                "gradient_norm": float(
                                    gradient_norm.detach().cpu()
                                ),
                                "learning_rate": learning_rate,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    trace.flush()
            try:
                next(loader_iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("packed microbatches remain after training")
        finally:
            if trace is not None:
                trace.close()
        save_training_state(
            model=ddp,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            adapter_output_dir=adapter_output_dir,
            rank=rank,
            local_rank=local_rank,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())
