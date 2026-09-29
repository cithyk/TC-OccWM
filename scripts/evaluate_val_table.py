from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.hybrid_world_scorer.model import HybridWorldScorer
from experiments.tc_occwm.model import CANDIDATE_SOURCE_IDS, TCOccWM
from experiments.tc_occwm.occupancy_cost import build_occupancy_selector_features, sample_trajectory_occupancy_cost
from rwc.config import bev_spec_from_config, load_config
from rwc.data.nuscenes_dataset import build_dataset
from rwc.data.perturbations import clean_perturb_tokens
from rwc.data.trajectory import comfort_cost, make_counterfactual_trajectories, trajectory_box_collision
from rwc.models.encoder import BEVEncoder
from rwc.models.no_rollout_scorer import NoRolloutTrajectoryScorer
from rwc.models.proposal import TrajectoryProposalCVAE
from rwc.models.world_model import LatentWorldModel
from rwc.planning.candidate_filter import FilterSpec, candidate_valid_mask
from rwc.training.trainer import move_batch_to_device
from rwc.utils.seed import seed_everything


@dataclass(frozen=True)
class EvalSetting:
    name: str
    bev_drop: float
    agent_drop: float
    xy_noise: float
    speed_noise: float
    delay: int


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, torch.Tensor | float], n: int) -> None:
        for key, value in metrics.items():
            scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            self.sums[key] = self.sums.get(key, 0.0) + scalar * n
            self.counts[key] = self.counts.get(key, 0) + n

    def compute(self) -> dict[str, float]:
        return {key: self.sums[key] / max(self.counts[key], 1) for key in sorted(self.sums)}


def parse_ckpt_arg(raw: str) -> tuple[str, str]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), path.strip()
    path = raw.strip()
    return Path(path).parent.name or Path(path).stem, path


def default_ckpts() -> list[tuple[str, str]]:
    def first_existing(*paths: str) -> str:
        return next((path for path in paths if Path(path).exists()), paths[0])

    return [
        ("world", "outputs/world_model/latest.pt"),
        ("proposal_aware", "outputs/world_model_proposal_aware/best.pt"),
        ("robust", "outputs/world_model_robust/latest.pt"),
        ("balanced_robust", first_existing("outputs/world_model_balanced_robust/best.pt", "outputs/world_model_balanced_robust/latest.pt")),
        ("robust_clean_preserve", first_existing("outputs/world_model_robust_clean_preserve/best.pt", "outputs/world_model_robust_clean_preserve/latest.pt")),
    ]


def default_no_rollout_ckpts() -> list[tuple[str, str]]:
    def first_existing(*paths: str) -> str:
        return next((path for path in paths if Path(path).exists()), paths[0])

    return [
        ("no_rollout_scorer", first_existing("outputs/no_rollout_scorer/best.pt", "outputs/no_rollout_scorer/latest.pt")),
    ]


def default_hybrid_ckpts() -> list[tuple[str, str]]:
    def first_existing(*paths: str) -> str:
        return next((path for path in paths if Path(path).exists()), paths[0])

    return [
        ("hybrid_world_scorer", first_existing("outputs/hybrid_world_scorer/best.pt", "outputs/hybrid_world_scorer/latest.pt")),
    ]


def default_tc_occwm_ckpts() -> list[tuple[str, str]]:
    def first_existing(*paths: str) -> str:
        return next((path for path in paths if Path(path).exists()), paths[0])

    return [
        ("tc_occwm", first_existing("outputs/tc_occwm/best.pt", "outputs/tc_occwm/latest.pt")),
    ]


def build_settings() -> list[EvalSetting]:
    return [
        EvalSetting("clean", 0.0, 0.0, 0.0, 0.0, 0),
        EvalSetting("mild", 0.05, 0.15, 0.2, 0.2, 1),
        EvalSetting("severe", 0.10, 0.30, 0.5, 0.5, 2),
    ]


def perturb_bev(bev: torch.Tensor, cfg: Any, setting: EvalSetting) -> torch.Tensor:
    out = bev.clone()
    if setting.bev_drop > 0:
        keep = torch.rand(out.shape[0], 1, out.shape[2], out.shape[3], device=out.device) > setting.bev_drop
        out = out * keep.float()
    if setting.agent_drop > 0 and "agents" in list(cfg.bev.channels):
        agent_idx = list(cfg.bev.channels).index("agents")
        keep = torch.rand(out.shape[0], 1, out.shape[2], out.shape[3], device=out.device) > setting.agent_drop
        out[:, agent_idx : agent_idx + 1] *= keep.float()
    return out


def perturb_candidates(candidates: torch.Tensor, setting: EvalSetting) -> torch.Tensor:
    out = candidates.clone()
    if setting.xy_noise > 0:
        out[..., :2] += torch.randn(out.shape[0], out.shape[1], 1, 2, device=out.device) * setting.xy_noise
    if setting.speed_noise > 0:
        out[..., 3:4] += torch.randn(out.shape[0], out.shape[1], 1, 1, device=out.device) * setting.speed_noise
    if setting.delay > 0:
        tail = out[:, :, -1:].expand(-1, -1, setting.delay, -1)
        out = torch.cat([out[:, :, setting.delay :], tail], dim=2)
    return out


def perturb_token(batch_size: int, device: torch.device, setting: EvalSetting) -> torch.Tensor:
    token = clean_perturb_tokens(batch_size, device)
    token[:, 0] = float(setting.delay)
    token[:, 1] = float(setting.bev_drop)
    token[:, 2] = float(setting.agent_drop)
    token[:, 3] = float(setting.xy_noise)
    token[:, 4] = float(setting.xy_noise)
    token[:, 6] = float(setting.speed_noise)
    token[:, 7] = 1.0 if setting.name != "clean" else 0.0
    return token


def ade_fde(candidates: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    err = torch.linalg.norm(candidates[..., :2] - target[:, None, :, :2], dim=-1)
    return err.mean(dim=-1), err[..., -1]


def gt_collision(candidates: torch.Tensor, agents: torch.Tensor) -> torch.Tensor:
    return trajectory_box_collision(candidates, agents)


def gt_offroad(candidates: torch.Tensor, bev: torch.Tensor, cfg: Any) -> torch.Tensor:
    b, n, t, _ = candidates.shape
    outs = []
    for bi in range(b):
        rr = ((candidates[bi, ..., 0] - float(cfg.bev.x_min)) / float(cfg.bev.resolution)).long()
        cc = ((candidates[bi, ..., 1] - float(cfg.bev.y_min)) / float(cfg.bev.resolution)).long()
        valid = (rr >= 0) & (rr < bev.shape[-2]) & (cc >= 0) & (cc < bev.shape[-1])
        sampled = torch.zeros(n, t, device=bev.device)
        sampled[valid] = bev[bi, 0, rr[valid], cc[valid]]
        outs.append(((~valid) | (sampled < 0.5)).any(dim=1).float())
    return torch.stack(outs, dim=0)


def progress(candidates: torch.Tensor, cfg: Any) -> torch.Tensor:
    return torch.clamp(candidates[..., -1, 0] / max(float(cfg.bev.x_max), 1.0), 0.0, 1.0)


def target_score(
    candidates: torch.Tensor,
    batch: dict[str, Any],
    cfg: Any,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ade, fde = ade_fde(candidates, batch["future_ego"])
    collision = gt_collision(candidates, batch["future_agents"])
    offroad = gt_offroad(candidates, batch["bev"], cfg)
    prog = progress(candidates, cfg)
    comfort = comfort_cost(candidates, float(cfg.planning.dt))
    score = (
        -weights["collision"] * collision
        -weights["offroad"] * offroad
        -weights["ade"] * ade
        -weights["fde"] * fde
        + weights["progress"] * prog
        -weights["comfort"] * comfort
    )
    return score, {
        "ade": ade,
        "fde": fde,
        "collision": collision,
        "offroad": offroad,
        "progress": prog,
        "comfort": comfort,
    }


def valid_mask(candidates: torch.Tensor, cfg: Any) -> torch.Tensor:
    spec = FilterSpec(
        max_speed_mps=float(cfg.planning.max_speed_mps),
        max_abs_y=float(cfg.planning.max_abs_y),
        min_x=float(cfg.planning.min_x),
        max_x=float(cfg.planning.max_x),
    )
    return candidate_valid_mask(candidates, spec)


def select_from_scores(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~valid, -1e6)
    return masked.argmax(dim=1)


def oracle_gt_score(components: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    return (
        -weights["collision"] * components["collision"]
        -weights["offroad"] * components["offroad"]
        -weights["ade"] * components["ade"]
        -weights["fde"] * components["fde"]
        + weights["progress"] * components["progress"]
        -weights["comfort"] * components["comfort"]
    )


def ego_hold_candidates(batch: dict[str, Any], cfg: Any, num_candidates: int) -> torch.Tensor:
    b = batch["future_ego"].shape[0]
    t = int(cfg.planning.horizon_steps)
    device = batch["future_ego"].device
    dtype = batch["future_ego"].dtype
    cand = torch.zeros(b, num_candidates, t, 4, device=device, dtype=dtype)
    speed = batch["ego"][:, 3].clamp(min=0.0)[:, None]
    steps = torch.arange(1, t + 1, device=device, dtype=dtype)[None, :] * float(cfg.planning.dt)
    base_x = speed * steps
    lateral = torch.linspace(-2.0, 2.0, num_candidates, device=device, dtype=dtype)
    cand[..., 0] = base_x[:, None, :]
    cand[..., 1] = lateral[None, :, None]
    cand[..., 3] = speed[:, None, :]
    if num_candidates > 0:
        cand[:, 0, :, 1] = 0.0
    return cand


def lattice_candidates(batch: dict[str, Any], cfg: Any, num_candidates: int) -> torch.Tensor:
    """Kinematic lattice around constant-velocity ego motion.

    This provides a non-learned candidate generator for plug-in reranker tests.
    It does not use future ego ground truth.
    """
    b = batch["future_ego"].shape[0]
    t = int(cfg.planning.horizon_steps)
    device = batch["future_ego"].device
    dtype = batch["future_ego"].dtype
    dt = float(cfg.planning.dt)
    base_speed = batch["ego"][:, 3].clamp(min=0.0)
    steps = torch.arange(1, t + 1, device=device, dtype=dtype)[None, :]

    lateral_offsets = torch.tensor([0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0], device=device, dtype=dtype)
    speed_scales = torch.tensor([1.0, 0.8, 1.2, 0.6, 1.4, 0.4, 1.6], device=device, dtype=dtype)
    pairs = [(float(lat.item()), float(scale.item())) for scale in speed_scales for lat in lateral_offsets]
    pairs = pairs[:num_candidates]
    if len(pairs) < num_candidates:
        pairs.extend([(0.0, 1.0)] * (num_candidates - len(pairs)))

    cand = torch.zeros(b, num_candidates, t, 4, device=device, dtype=dtype)
    for idx, (lat, scale) in enumerate(pairs):
        speed = (base_speed * scale).clamp(min=0.0, max=float(cfg.planning.max_speed_mps))
        final_lat = cand.new_full((b, 1), lat)
        s = (steps / max(float(t), 1.0)).clamp(0.0, 1.0)
        smooth = 3.0 * s.pow(2) - 2.0 * s.pow(3)
        cand[:, idx, :, 0] = speed[:, None] * steps * dt
        cand[:, idx, :, 1] = final_lat * smooth
        dx = torch.diff(cand[:, idx, :, 0], prepend=torch.zeros(b, 1, device=device, dtype=dtype), dim=1)
        dy = torch.diff(cand[:, idx, :, 1], prepend=torch.zeros(b, 1, device=device, dtype=dtype), dim=1)
        cand[:, idx, :, 2] = torch.atan2(dy, dx.clamp_min(1e-3))
        cand[:, idx, :, 3] = speed[:, None]
    return cand


def candidate_sources_from_arg(raw: str) -> list[str]:
    if raw == "all":
        return ["proposal", "lattice", "counterfactual", "ego_hold"]
    return [raw]


def counterfactual_candidates(batch: dict[str, Any], num_candidates: int) -> torch.Tensor:
    return torch.stack(
        [make_counterfactual_trajectories(batch["future_ego"][i], num_candidates) for i in range(batch["future_ego"].shape[0])],
        dim=0,
    )


def build_candidates(
    candidate_source: str,
    batch: dict[str, Any],
    cfg: Any,
    num_candidates: int,
    proposal_encoder: BEVEncoder | None,
    proposal: TrajectoryProposalCVAE | None,
) -> torch.Tensor:
    if candidate_source == "proposal":
        assert proposal_encoder is not None and proposal is not None
        z_prop = proposal_encoder(batch["bev"])
        return proposal.sample(z_prop, num_candidates)
    if candidate_source == "counterfactual":
        return counterfactual_candidates(batch, num_candidates)
    if candidate_source == "ego_hold":
        return ego_hold_candidates(batch, cfg, num_candidates)
    if candidate_source == "lattice":
        return lattice_candidates(batch, cfg, num_candidates)
    if candidate_source == "combined":
        assert proposal_encoder is not None and proposal is not None
        z_prop = proposal_encoder(batch["bev"])
        return torch.cat(
            [
                proposal.sample(z_prop, num_candidates),
                lattice_candidates(batch, cfg, num_candidates),
                counterfactual_candidates(batch, num_candidates),
                ego_hold_candidates(batch, cfg, num_candidates),
            ],
            dim=1,
        )
    raise ValueError(f"Unknown candidate source: {candidate_source}")


def build_proposal(cfg: Any, spec: Any, device: torch.device, ckpt_path: str, num_candidates: int):
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    proposal = TrajectoryProposalCVAE(
        int(cfg.model.latent_dim),
        int(cfg.planning.horizon_steps),
        int(cfg.model.trajectory_dim),
        int(cfg.model.proposal_latent_dim),
        num_candidates=num_candidates,
    ).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    proposal.load_state_dict(ckpt["proposal"])
    return encoder, proposal


def build_world(cfg: Any, spec: Any, device: torch.device, ckpt_path: str):
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    world = LatentWorldModel(
        int(cfg.model.latent_dim),
        int(cfg.model.hidden_dim),
        int(cfg.model.trajectory_dim),
        int(cfg.model.perturb_dim),
        int(cfg.planning.horizon_steps),
        int(cfg.model.occupancy_channels),
        len(cfg.model.risk_names),
    ).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    world.load_state_dict(ckpt["world_model"])
    return encoder, world


def build_no_rollout(cfg: Any, spec: Any, device: torch.device, ckpt_path: str):
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    scorer = NoRolloutTrajectoryScorer(
        int(cfg.model.latent_dim),
        int(cfg.model.hidden_dim),
        int(cfg.model.trajectory_dim),
        int(cfg.model.perturb_dim),
        int(cfg.planning.horizon_steps),
        len(cfg.model.risk_names),
    ).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["scorer"])
    return encoder, scorer


def build_hybrid(cfg: Any, spec: Any, device: torch.device, ckpt_path: str):
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    scorer = HybridWorldScorer(
        int(cfg.model.latent_dim),
        int(cfg.model.hidden_dim),
        int(cfg.model.trajectory_dim),
        int(cfg.model.perturb_dim),
        int(cfg.planning.horizon_steps),
        int(cfg.model.occupancy_channels),
        len(cfg.model.risk_names),
    ).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["hybrid_scorer"])
    return encoder, scorer


def build_tc_occwm(
    cfg: Any,
    spec: Any,
    device: torch.device,
    ckpt_path: str,
    disable_uncertainty_features: bool | None = None,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    scorer = TCOccWM(
        int(cfg.model.latent_dim),
        int(cfg.model.hidden_dim),
        int(cfg.model.trajectory_dim),
        int(cfg.model.perturb_dim),
        int(cfg.planning.horizon_steps),
        int(cfg.model.occupancy_channels),
        len(cfg.model.risk_names),
        num_candidate_sources=len(CANDIDATE_SOURCE_IDS) if bool(ckpt_args.get("use_source_embedding", False)) else 0,
        source_embed_dim=int(ckpt_args.get("source_embed_dim", 4)),
    ).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["tc_occwm"])
    scorer.disable_occupancy_features = bool(ckpt_args.get("disable_occupancy_features", False))
    scorer.disable_uncertainty_features = (
        bool(ckpt_args.get("disable_uncertainty_features", False))
        if disable_uncertainty_features is None
        else disable_uncertainty_features
    )
    scorer.normalize_occupancy_features = bool(ckpt_args.get("normalize_occupancy_features", False))
    scorer.use_source_embedding = bool(ckpt_args.get("use_source_embedding", False))
    return encoder, scorer


def try_build_world(cfg: Any, spec: Any, device: torch.device, name: str, path: str):
    try:
        encoder, world = build_world(cfg, spec, device, path)
    except RuntimeError as exc:
        print(f"skip incompatible checkpoint {name}: {path}")
        print(f"  {exc}")
        return None
    return name, encoder, world


def try_build_no_rollout(cfg: Any, spec: Any, device: torch.device, name: str, path: str):
    try:
        encoder, scorer = build_no_rollout(cfg, spec, device, path)
    except RuntimeError as exc:
        print(f"skip incompatible no-rollout checkpoint {name}: {path}")
        print(f"  {exc}")
        return None
    return name, encoder, scorer


def try_build_hybrid(cfg: Any, spec: Any, device: torch.device, name: str, path: str):
    try:
        encoder, scorer = build_hybrid(cfg, spec, device, path)
    except RuntimeError as exc:
        print(f"skip incompatible hybrid checkpoint {name}: {path}")
        print(f"  {exc}")
        return None
    return name, encoder, scorer


def try_build_tc_occwm(
    cfg: Any,
    spec: Any,
    device: torch.device,
    name: str,
    path: str,
    disable_uncertainty_features: bool | None = None,
):
    try:
        encoder, scorer = build_tc_occwm(cfg, spec, device, path, disable_uncertainty_features)
    except RuntimeError as exc:
        print(f"skip incompatible TC-OccWM checkpoint {name}: {path}")
        print(f"  {exc}")
        return None
    return name, encoder, scorer


def world_score(world: LatentWorldModel, z: torch.Tensor, candidates: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    raw = world(z, candidates, token)["risk"]
    risk = torch.sigmoid(raw[..., :3]).amax(dim=2)
    prog = raw[..., 3].mean(dim=2)
    uncertainty = F.softplus(raw[..., 4]).mean(dim=2)
    return prog - risk.sum(dim=-1) - 0.1 * uncertainty


def no_rollout_score(
    scorer: NoRolloutTrajectoryScorer,
    z: torch.Tensor,
    candidates: torch.Tensor,
    token: torch.Tensor,
) -> torch.Tensor:
    raw = scorer(z, candidates, token)["risk"]
    risk = torch.sigmoid(raw[..., :3]).amax(dim=2)
    prog = raw[..., 3].mean(dim=2)
    uncertainty = F.softplus(raw[..., 4]).mean(dim=2)
    return prog - risk.sum(dim=-1) - 0.1 * uncertainty


def hybrid_score(
    scorer: HybridWorldScorer,
    z: torch.Tensor,
    candidates: torch.Tensor,
    token: torch.Tensor,
) -> torch.Tensor:
    raw = scorer(z, candidates, token, return_world_outputs=False)["risk"]
    risk = torch.sigmoid(raw[..., : scorer.num_risks]).amax(dim=2)
    prog = raw[..., scorer.num_risks].mean(dim=2)
    uncertainty = F.softplus(raw[..., scorer.num_risks + 1]).mean(dim=2)
    return prog - risk.sum(dim=-1) - 0.1 * uncertainty


def tc_occwm_score(
    scorer: TCOccWM,
    z: torch.Tensor,
    candidates: torch.Tensor,
    token: torch.Tensor,
    cfg: Any,
    components: dict[str, torch.Tensor],
    candidate_source: str = "proposal",
) -> torch.Tensor:
    outputs = scorer(z, candidates, token)
    costs = sample_trajectory_occupancy_cost(
        outputs["future_occupancy"],
        candidates,
        cfg,
        uncertainty_logits=outputs["world_uncertainty"],
    )
    selector_features = build_occupancy_selector_features(
        costs,
        components["progress"],
        components["comfort"],
        normalize=bool(getattr(scorer, "normalize_occupancy_features", False)),
        disable_uncertainty=bool(getattr(scorer, "disable_uncertainty_features", False)),
    )
    if bool(getattr(scorer, "disable_occupancy_features", False)):
        selector_features = selector_features.clone()
        selector_features[..., :5] = 0.0
    source_ids = None
    if bool(getattr(scorer, "use_source_embedding", False)):
        source_value = CANDIDATE_SOURCE_IDS.get(candidate_source, CANDIDATE_SOURCE_IDS["combined"])
        source_ids = torch.full(candidates.shape[:2], source_value, device=candidates.device, dtype=torch.long)
    return scorer.score(outputs, selector_features, source_ids)


def rule_based_score(
    candidates: torch.Tensor,
    batch: dict[str, Any],
    cfg: Any,
    components: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Privileged hand-crafted reranker used as a non-learned safety baseline.

    It avoids expert future ego labels, but it uses cached future-agent tracks
    for collision and proximity costs. Treat it as a strong safety reference,
    not a fully deployable planner unless future-agent predictions replace
    those labels.
    """
    progress_score = components["progress"]
    comfort = components["comfort"]
    collision = components["collision"]
    offroad = components["offroad"]
    lateral = candidates[..., -1, 1].abs() / max(float(cfg.planning.max_abs_y), 1.0)
    final_x = candidates[..., -1, 0]
    reverse = (final_x < 0.0).float()

    ego_xy = candidates[..., :2].unsqueeze(3)
    agent_xy = batch["future_agents"][:, None, :, :, :2]
    valid_agent = (batch["future_agents"][:, None, :, :, 2] > 0) & (batch["future_agents"][:, None, :, :, 3] > 0)
    dist = torch.linalg.norm(ego_xy - agent_xy, dim=-1).masked_fill(~valid_agent, 1e6)
    min_dist = dist.amin(dim=(2, 3))
    proximity = torch.exp(-min_dist / 3.0)

    return (
        2.0 * progress_score
        - 8.0 * collision
        - 5.0 * offroad
        - 1.5 * proximity
        - 0.4 * lateral
        - 0.05 * comfort
        - 2.0 * reverse
    )


def _sample_bev_channel_along_candidates(channel: torch.Tensor, candidates: torch.Tensor, cfg: Any) -> torch.Tensor:
    """Sample one BEV channel along candidate trajectory points.

    Args:
        channel: [B,H,W]
        candidates: [B,N,T,D] in ego metric coordinates.
    """
    b, n, t, _ = candidates.shape
    rr = ((candidates[..., 0] - float(cfg.bev.x_min)) / float(cfg.bev.resolution)).long()
    cc = ((candidates[..., 1] - float(cfg.bev.y_min)) / float(cfg.bev.resolution)).long()
    valid = (rr >= 0) & (rr < channel.shape[-2]) & (cc >= 0) & (cc < channel.shape[-1])
    sampled = torch.zeros(b, n, t, device=candidates.device, dtype=channel.dtype)
    batch_idx = torch.arange(b, device=candidates.device)[:, None, None].expand(b, n, t)
    sampled[valid] = channel[batch_idx[valid], rr[valid], cc[valid]]
    return sampled


def rule_current_score(
    candidates: torch.Tensor,
    batch: dict[str, Any],
    cfg: Any,
    components: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Deployable hand-crafted reranker using only the current BEV observation.

    Unlike rule_based_score, this baseline does not use future-agent tracks.
    It scores candidates from current drivable area, current agent occupancy,
    progress, comfort, lateral offset, and reverse motion penalties.
    """
    progress_score = components["progress"]
    comfort = components["comfort"]
    offroad = components["offroad"]
    lateral = candidates[..., -1, 1].abs() / max(float(cfg.planning.max_abs_y), 1.0)
    reverse = (candidates[..., -1, 0] < 0.0).float()

    channels = list(cfg.bev.channels)
    if "agents" in channels:
        agent_idx = channels.index("agents")
        agent_map = batch["bev"][:, agent_idx]
        agent_hit = _sample_bev_channel_along_candidates(agent_map, candidates, cfg).amax(dim=2)
        near_map = F.max_pool2d(agent_map[:, None], kernel_size=7, stride=1, padding=3)[:, 0]
        far_map = F.max_pool2d(agent_map[:, None], kernel_size=15, stride=1, padding=7)[:, 0]
        near = _sample_bev_channel_along_candidates(near_map, candidates, cfg).amax(dim=2)
        far = _sample_bev_channel_along_candidates(far_map, candidates, cfg).amax(dim=2)
        proximity = 1.0 * near + 0.35 * far
    else:
        agent_hit = torch.zeros_like(progress_score)
        proximity = torch.zeros_like(progress_score)

    return (
        2.0 * progress_score
        - 6.0 * agent_hit
        - 5.0 * offroad
        - 1.2 * proximity
        - 0.4 * lateral
        - 0.05 * comfort
        - 2.0 * reverse
    )


def selected_metrics(
    components: dict[str, torch.Tensor],
    target_scores: torch.Tensor,
    selected: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ar = torch.arange(selected.shape[0], device=selected.device)
    oracle_idx = select_from_scores(target_scores, valid)
    return {
        "selected_ADE": components["ade"][ar, selected].mean(),
        "selected_FDE": components["fde"][ar, selected].mean(),
        "selected_collision": components["collision"][ar, selected].mean(),
        "selected_offroad": components["offroad"][ar, selected].mean(),
        "selected_progress": components["progress"][ar, selected].mean(),
        "selected_comfort": components["comfort"][ar, selected].mean(),
        "selected_target_score": target_scores[ar, selected].mean(),
        "oracle_target_score": target_scores[ar, oracle_idx].mean(),
        "score_regret": (target_scores[ar, oracle_idx] - target_scores[ar, selected]).mean(),
        "valid_rate": valid.float().mean(),
    }


def add_row(rows: list[dict[str, Any]], method: str, setting: str, candidate_source: str, metrics: dict[str, float]) -> None:
    rows.append(
        {
            "method": method,
            "setting": setting,
            "candidate_source": candidate_source,
            **{key: round(value, 6) for key, value in metrics.items()},
        }
    )


def format_markdown(rows: list[dict[str, Any]]) -> str:
    preferred = [
        "method",
        "setting",
        "candidate_source",
        "selected_ADE",
        "selected_FDE",
        "selected_collision",
        "selected_offroad",
        "selected_progress",
        "score_regret",
        "valid_rate",
    ]
    cols = [col for col in preferred if any(col in row for row in rows)]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(format_markdown(rows), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean/mild/severe validation tables for RWC baselines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--proposal-ckpt", default="outputs/proposal/latest.pt")
    parser.add_argument(
        "--ckpts",
        nargs="*",
        default=None,
        help="World checkpoints as name=path. Defaults to known outputs if present.",
    )
    parser.add_argument(
        "--no-rollout-ckpts",
        nargs="*",
        default=None,
        help="No-rollout scorer checkpoints as name=path. Defaults to outputs/no_rollout_scorer if present.",
    )
    parser.add_argument(
        "--hybrid-ckpts",
        nargs="*",
        default=None,
        help="Hybrid scorer checkpoints as name=path. Defaults to outputs/hybrid_world_scorer if present.",
    )
    parser.add_argument(
        "--tc-occwm-ckpts",
        nargs="*",
        default=None,
        help="TC-OccWM checkpoints as name=path. Defaults to outputs/tc_occwm if present.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=["proposal", "lattice", "counterfactual", "ego_hold", "all", "combined"],
        default="proposal",
        help="'all' evaluates proposal/lattice/counterfactual/ego_hold separately; 'combined' concatenates them.",
    )
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/val_tables")
    parser.add_argument("--prefix", default="val_table")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--disable-uncertainty-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint setting for the occupancy-derived uncertainty selector feature.",
    )
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
    num_candidates = int(args.num_candidates or cfg.planning.num_candidates)
    batch_size = int(args.batch_size or cfg.training.batch_size)
    weights = {
        "collision": float(args.w_collision),
        "offroad": float(args.w_offroad),
        "ade": float(args.w_ade),
        "fde": float(args.w_fde),
        "progress": float(args.w_progress),
        "comfort": float(args.w_comfort),
    }

    dataset = build_dataset(cfg, spec, args.split, args.synthetic)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if args.synthetic else int(cfg.data.num_workers),
    )
    proposal_encoder = proposal = None
    candidate_sources = candidate_sources_from_arg(args.candidate_source)
    if any(source in {"proposal", "combined"} for source in candidate_sources):
        proposal_encoder, proposal = build_proposal(cfg, spec, device, args.proposal_ckpt, num_candidates)

    ckpts = [parse_ckpt_arg(item) for item in args.ckpts] if args.ckpts else default_ckpts()
    available_ckpts = [(name, path) for name, path in ckpts if Path(path).exists()]
    missing_ckpts = [(name, path) for name, path in ckpts if not Path(path).exists()]
    for name, path in missing_ckpts:
        print(f"skip missing checkpoint {name}: {path}")
    world_models = []
    for name, path in available_ckpts:
        built = try_build_world(cfg, spec, device, name, path)
        if built is not None:
            world_models.append(built)
    no_rollout_ckpts = (
        [parse_ckpt_arg(item) for item in args.no_rollout_ckpts]
        if args.no_rollout_ckpts
        else default_no_rollout_ckpts()
    )
    available_no_rollout = [(name, path) for name, path in no_rollout_ckpts if Path(path).exists()]
    missing_no_rollout = [(name, path) for name, path in no_rollout_ckpts if not Path(path).exists()]
    for name, path in missing_no_rollout:
        print(f"skip missing no-rollout checkpoint {name}: {path}")
    no_rollout_models = []
    for name, path in available_no_rollout:
        built = try_build_no_rollout(cfg, spec, device, name, path)
        if built is not None:
            no_rollout_models.append(built)
    hybrid_ckpts = (
        [parse_ckpt_arg(item) for item in args.hybrid_ckpts]
        if args.hybrid_ckpts
        else default_hybrid_ckpts()
    )
    available_hybrid = [(name, path) for name, path in hybrid_ckpts if Path(path).exists()]
    missing_hybrid = [(name, path) for name, path in hybrid_ckpts if not Path(path).exists()]
    for name, path in missing_hybrid:
        print(f"skip missing hybrid checkpoint {name}: {path}")
    hybrid_models = []
    for name, path in available_hybrid:
        built = try_build_hybrid(cfg, spec, device, name, path)
        if built is not None:
            hybrid_models.append(built)
    tc_occwm_ckpts = (
        [parse_ckpt_arg(item) for item in args.tc_occwm_ckpts]
        if args.tc_occwm_ckpts
        else default_tc_occwm_ckpts()
    )
    available_tc_occwm = [(name, path) for name, path in tc_occwm_ckpts if Path(path).exists()]
    missing_tc_occwm = [(name, path) for name, path in tc_occwm_ckpts if not Path(path).exists()]
    for name, path in missing_tc_occwm:
        print(f"skip missing TC-OccWM checkpoint {name}: {path}")
    tc_occwm_models = []
    for name, path in available_tc_occwm:
        built = try_build_tc_occwm(
            cfg,
            spec,
            device,
            name,
            path,
            args.disable_uncertainty_features,
        )
        if built is not None:
            tc_occwm_models.append(built)

    rows: list[dict[str, Any]] = []
    settings = build_settings()
    for candidate_source in candidate_sources:
        for setting in settings:
            meters: dict[str, MetricAccumulator] = {
                "expert": MetricAccumulator(),
                "proposal_first": MetricAccumulator(),
                "rule_current": MetricAccumulator(),
                "rule_based_cost": MetricAccumulator(),
                "oracle_gt_score": MetricAccumulator(),
            }
            meters.update({f"world:{name}": MetricAccumulator() for name, _, _ in world_models})
            meters.update({f"no_rollout:{name}": MetricAccumulator() for name, _, _ in no_rollout_models})
            meters.update({f"hybrid:{name}": MetricAccumulator() for name, _, _ in hybrid_models})
            meters.update({f"tc_occwm:{name}": MetricAccumulator() for name, _, _ in tc_occwm_models})

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"val table {candidate_source}/{setting.name}")):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                b = batch["bev"].shape[0]
                with torch.no_grad():
                    base_candidates = build_candidates(
                        candidate_source,
                        batch,
                        cfg,
                        num_candidates,
                        proposal_encoder,
                        proposal,
                    )

                    candidates = perturb_candidates(base_candidates, setting)
                    noisy_bev = perturb_bev(batch["bev"], cfg, setting)
                    token = perturb_token(b, device, setting)
                    target_scores, components = target_score(candidates, batch, cfg, weights)
                    valid = valid_mask(candidates, cfg)
                    oracle_idx = select_from_scores(oracle_gt_score(components, weights), valid)
                    first_idx = torch.zeros(b, dtype=torch.long, device=device)

                    expert_candidates = batch["future_ego"][:, None].expand(-1, candidates.shape[1], -1, -1).clone()
                    expert_scores, expert_components = target_score(expert_candidates, batch, cfg, weights)
                    expert_valid = valid_mask(expert_candidates, cfg)
                    meters["expert"].update(selected_metrics(expert_components, expert_scores, first_idx, expert_valid), b)
                    meters["proposal_first"].update(selected_metrics(components, target_scores, first_idx, valid), b)
                    current_rule_scores = rule_current_score(candidates, batch, cfg, components)
                    current_rule_idx = select_from_scores(current_rule_scores, valid)
                    meters["rule_current"].update(selected_metrics(components, target_scores, current_rule_idx, valid), b)
                    rule_scores = rule_based_score(candidates, batch, cfg, components)
                    rule_idx = select_from_scores(rule_scores, valid)
                    meters["rule_based_cost"].update(selected_metrics(components, target_scores, rule_idx, valid), b)
                    meters["oracle_gt_score"].update(selected_metrics(components, target_scores, oracle_idx, valid), b)

                    for name, enc_w, world in world_models:
                        z_world = enc_w(noisy_bev)
                        pred_scores = world_score(world, z_world, candidates, token)
                        best = select_from_scores(pred_scores, valid)
                        meters[f"world:{name}"].update(selected_metrics(components, target_scores, best, valid), b)

                    for name, enc_s, scorer in no_rollout_models:
                        z_score = enc_s(noisy_bev)
                        pred_scores = no_rollout_score(scorer, z_score, candidates, token)
                        best = select_from_scores(pred_scores, valid)
                        meters[f"no_rollout:{name}"].update(selected_metrics(components, target_scores, best, valid), b)

                    for name, enc_h, scorer in hybrid_models:
                        z_hybrid = enc_h(noisy_bev)
                        pred_scores = hybrid_score(scorer, z_hybrid, candidates, token)
                        best = select_from_scores(pred_scores, valid)
                        meters[f"hybrid:{name}"].update(selected_metrics(components, target_scores, best, valid), b)

                    for name, enc_t, scorer in tc_occwm_models:
                        z_tc = enc_t(noisy_bev)
                        pred_scores = tc_occwm_score(scorer, z_tc, candidates, token, cfg, components, candidate_source)
                        best = select_from_scores(pred_scores, valid)
                        meters[f"tc_occwm:{name}"].update(selected_metrics(components, target_scores, best, valid), b)

            for method, meter in meters.items():
                add_row(rows, method, setting.name, candidate_source, meter.compute())

    write_outputs(rows, Path(args.output_dir), args.prefix)


if __name__ == "__main__":
    main()
