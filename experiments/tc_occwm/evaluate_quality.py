from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.tc_occwm.occupancy_cost import sample_trajectory_occupancy_cost
from rwc.config import bev_spec_from_config, load_config
from rwc.data.nuscenes_dataset import build_dataset
from rwc.training.trainer import move_batch_to_device
from rwc.utils.seed import seed_everything
from scripts.evaluate_val_table import (
    build_candidates,
    build_proposal,
    build_settings,
    build_tc_occwm,
    candidate_sources_from_arg,
    perturb_bev,
    perturb_candidates,
    perturb_token,
    target_score,
)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = int(labels.sum())
    neg = int(labels.shape[0] - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = np.concatenate([[0.0], tps / max(pos, 1), [1.0]])
    fpr = np.concatenate([[0.0], fps / max(neg, 1), [1.0]])
    return float(np.trapz(tpr, fpr))


def binary_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels)
    precision = tps / np.arange(1, sorted_labels.shape[0] + 1)
    recall = tps / max(pos, 1)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def binary_ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    labels = labels.astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(scores.shape[0], 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.sum()) / total * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return ece


def safe_corr(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size < 2 or float(scores.std()) < 1e-8 or float(labels.std()) < 1e-8:
        return float("nan")
    return float(np.corrcoef(scores, labels)[0, 1])


def occupancy_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    """Evaluate expert-conditioned future occupancy maps."""
    b, n, t, c, h, w = logits.shape
    if target.shape[-2:] != (h, w):
        old_shape = target.shape
        target = F.interpolate(target.flatten(0, 1), size=(h, w), mode="nearest").view(old_shape[0], old_shape[1], old_shape[2], h, w)
    target = target[:, None].expand_as(logits)
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target).detach()
    pred = prob > threshold
    gt = target > 0.5
    inter = (pred & gt).float().sum()
    union = (pred | gt).float().sum()
    tp = inter
    fp = (pred & ~gt).float().sum()
    fn = (~pred & gt).float().sum()
    iou = inter / union.clamp_min(1.0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)

    pred_agent = pred[..., 0, :, :]
    gt_agent = gt[..., 0, :, :]
    inter_agent = (pred_agent & gt_agent).float().sum()
    union_agent = (pred_agent | gt_agent).float().sum()
    tp_agent = inter_agent
    fp_agent = (pred_agent & ~gt_agent).float().sum()
    fn_agent = (~pred_agent & gt_agent).float().sum()
    return {
        "occ_bce": float(bce.cpu()),
        "occ_miou": float(iou.cpu()),
        "occ_f1": float(f1.cpu()),
        "agent_iou": float((inter_agent / union_agent.clamp_min(1.0)).cpu()),
        "agent_f1": float((2.0 * tp_agent / (2.0 * tp_agent + fp_agent + fn_agent).clamp_min(1.0)).cpu()),
    }


class RunningMean:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, float], n: int) -> None:
        for key, value in values.items():
            if np.isfinite(value):
                self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        return {key: value / max(self.count, 1) for key, value in self.sums.items()}


def format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# TC-OccWM Quality Evaluation", ""]
    for source in sorted({row["candidate_source"] for row in rows}):
        lines += [f"## {source}", ""]
        lines.append("| setting | occ_bce | occ_miou | agent_f1 | cost_AUROC | cost_AUPRC | cost_ECE | cost_collision_corr |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in [r for r in rows if r["candidate_source"] == source]:
            lines.append(
                "| {setting} | {occ_bce:.4f} | {occ_miou:.4f} | {agent_f1:.4f} | {cost_AUROC:.4f} | {cost_AUPRC:.4f} | {cost_ECE:.4f} | {cost_collision_corr:.4f} |".format(
                    **row
                )
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TC-OccWM occupancy quality and cost-collision alignment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--proposal-ckpt", default="outputs/proposal/latest.pt")
    parser.add_argument("--tc-occwm-ckpt", default="outputs/tc_occwm/best.pt")
    parser.add_argument("--candidate-source", choices=["proposal", "lattice", "counterfactual", "ego_hold", "all", "combined"], default="proposal")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/tc_occwm_quality")
    parser.add_argument("--prefix", default="tc_occwm_quality")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--w-collision", type=float, default=5.0)
    parser.add_argument("--w-offroad", type=float, default=3.0)
    parser.add_argument("--w-ade", type=float, default=0.3)
    parser.add_argument("--w-fde", type=float, default=0.2)
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-comfort", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else cfg.seed))
    spec = bev_spec_from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(args.batch_size or cfg.training.batch_size)
    num_candidates = int(args.num_candidates or cfg.planning.num_candidates)
    weights = {
        "collision": float(args.w_collision),
        "offroad": float(args.w_offroad),
        "ade": float(args.w_ade),
        "fde": float(args.w_fde),
        "progress": float(args.w_progress),
        "comfort": float(args.w_comfort),
    }
    dataset = build_dataset(cfg, spec, args.split, args.synthetic)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0 if args.synthetic else int(cfg.data.num_workers))
    candidate_sources = candidate_sources_from_arg(args.candidate_source)
    proposal_encoder = proposal = None
    if any(source in {"proposal", "combined"} for source in candidate_sources):
        proposal_encoder, proposal = build_proposal(cfg, spec, device, args.proposal_ckpt, num_candidates)
    encoder, model = build_tc_occwm(cfg, spec, device, args.tc_occwm_ckpt)

    rows: list[dict[str, Any]] = []
    for source in candidate_sources:
        for setting in build_settings():
            occ_meter = RunningMean()
            all_costs: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"tc quality {source}/{setting.name}")):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                b = batch["bev"].shape[0]
                with torch.no_grad():
                    noisy_bev = perturb_bev(batch["bev"], cfg, setting)
                    token = perturb_token(b, device, setting)

                    expert_candidates = batch["future_ego"][:, None].clone()
                    expert_outputs = model(encoder(noisy_bev), expert_candidates, token)
                    occ_meter.update(occupancy_metrics(expert_outputs["future_occupancy"], batch["future_occupancy"], args.threshold), b)

                    candidates = build_candidates(source, batch, cfg, num_candidates, proposal_encoder, proposal)
                    candidates = perturb_candidates(candidates, setting)
                    _, components = target_score(candidates, batch, cfg, weights)
                    outputs = model(encoder(noisy_bev), candidates, token)
                    costs = sample_trajectory_occupancy_cost(outputs["future_occupancy"], candidates, cfg, outputs["world_uncertainty"])
                    all_costs.append(costs["collision_cost"].detach().float().cpu().reshape(-1).numpy())
                    all_labels.append(components["collision"].detach().float().cpu().reshape(-1).numpy())

            occ = occ_meter.compute()
            cost_arr = np.concatenate(all_costs) if all_costs else np.array([], dtype=np.float32)
            label_arr = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
            row = {
                "candidate_source": source,
                "setting": setting.name,
                **occ,
                "cost_AUROC": binary_auc(cost_arr, label_arr) if cost_arr.size else float("nan"),
                "cost_AUPRC": binary_auprc(cost_arr, label_arr) if cost_arr.size else float("nan"),
                "cost_ECE": binary_ece(cost_arr.clip(0.0, 1.0), label_arr) if cost_arr.size else float("nan"),
                "cost_collision_corr": safe_corr(cost_arr, label_arr) if cost_arr.size else float("nan"),
                "collision_positive_rate": float(label_arr.mean()) if label_arr.size else float("nan"),
            }
            rows.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.prefix}.json"
    csv_path = out_dir / f"{args.prefix}.csv"
    md_path = out_dir / f"{args.prefix}.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(format_markdown(rows), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
