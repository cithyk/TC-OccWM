from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from tqdm import tqdm

from rwc.config import bev_spec_from_config, load_config
from rwc.data.bev_rasterizer import AgentBox, BEVRasterizer
from rwc.data.trajectory import trajectory_box_collision_np
from rwc.utils.io import ensure_dir

DEFAULT_AGENT_CATEGORY_PREFIXES = ("vehicle.", "human.pedestrian.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples to cache. Use 0 or a negative value to cache the full split.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Write synthetic cache samples for pipeline testing.")
    return parser.parse_args()


def write_synthetic_cache(cfg, split: str, max_samples: int) -> None:
    if max_samples <= 0:
        raise ValueError("Synthetic cache needs --max-samples > 0.")
    spec = bev_spec_from_config(cfg)
    out_dir = ensure_dir(Path(cfg.data.cache_root or "cache") / split)
    rasterizer = BEVRasterizer(spec)
    for idx in tqdm(range(max_samples), desc=f"writing synthetic cache {split}"):
        sample = rasterizer.synthetic_scene(np.random.default_rng(int(cfg.seed) + idx), int(cfg.planning.horizon_steps), float(cfg.planning.dt))
        np.savez_compressed(
            out_dir / f"{idx:08d}.npz",
            bev=sample["bev"],
            ego=sample["ego"],
            future_ego=sample["future_ego"],
            future_agents=sample["future_agents"],
            future_occupancy=sample["future_occupancy"],
            risk=sample["risk"],
            scene_id=np.array(f"synthetic_{idx:06d}"),
            sample_token=np.array(f"synthetic_{idx:06d}"),
        )
    print(f"Wrote {max_samples} samples to {out_dir}")


def yaw_from_quaternion(q) -> float:
    _, _, yaw = q.yaw_pitch_roll
    return float(yaw)


def global_to_ego(point_xyz: np.ndarray, ego_translation: np.ndarray, ego_rotation) -> np.ndarray:
    return np.asarray(ego_rotation.inverse.rotate(point_xyz - ego_translation), dtype=np.float32)


def annotation_to_agent_box(ann: dict, ego_translation: np.ndarray, ego_rotation) -> AgentBox:
    from pyquaternion import Quaternion

    center = global_to_ego(np.asarray(ann["translation"], dtype=np.float32), ego_translation, ego_rotation)
    ann_rot = Quaternion(ann["rotation"])
    yaw = yaw_from_quaternion(ego_rotation.inverse * ann_rot)
    # nuScenes stores sample_annotation["size"] as [width, length, height].
    width, length, _ = ann["size"]
    return AgentBox(x=float(center[0]), y=float(center[1]), length=float(length), width=float(width), yaw=float(yaw))


def agent_category_prefixes(cfg) -> tuple[str, ...]:
    prefixes = cfg.data.get("agent_category_prefixes", DEFAULT_AGENT_CATEGORY_PREFIXES)
    return tuple(str(prefix) for prefix in prefixes)


def is_traffic_participant(ann: dict, prefixes: tuple[str, ...]) -> bool:
    category = str(ann.get("category_name", ""))
    return any(category.startswith(prefix) for prefix in prefixes)


def future_ego_trajectory(nusc, sample: dict, ego_translation: np.ndarray, ego_rotation, horizon: int) -> np.ndarray | None:
    future = np.zeros((horizon, 4), dtype=np.float32)
    cursor = sample
    prev_xy = np.zeros(2, dtype=np.float32)
    for step in range(horizon):
        if not cursor["next"]:
            return None
        cursor = nusc.get("sample", cursor["next"])
        lidar = nusc.get("sample_data", cursor["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", lidar["ego_pose_token"])
        point = global_to_ego(np.asarray(pose["translation"], dtype=np.float32), ego_translation, ego_rotation)
        xy = point[:2]
        speed = float(np.linalg.norm(xy - prev_xy) / 0.5)
        yaw = 0.0
        if step > 0:
            delta = xy - prev_xy
            if np.linalg.norm(delta) > 1e-3:
                yaw = float(np.arctan2(delta[1], delta[0]))
        future[step] = np.array([xy[0], xy[1], yaw, speed], dtype=np.float32)
        prev_xy = xy
    return future


def future_agent_tracks(
    nusc,
    sample: dict,
    ego_translation: np.ndarray,
    ego_rotation,
    horizon: int,
    category_prefixes: tuple[str, ...],
    max_agents: int = 32,
) -> np.ndarray:
    tracks = np.zeros((horizon, max_agents, 5), dtype=np.float32)
    current_anns = [
        ann
        for token in sample["anns"]
        for ann in [nusc.get("sample_annotation", token)]
        if is_traffic_participant(ann, category_prefixes)
    ]
    current_anns = sorted(current_anns, key=lambda ann: np.linalg.norm(np.asarray(ann["translation"]) - ego_translation))[:max_agents]
    for agent_idx, ann in enumerate(current_anns):
        cursor = ann
        for step in range(horizon):
            token = cursor["next"]
            if not token:
                break
            cursor = nusc.get("sample_annotation", token)
            box = annotation_to_agent_box(cursor, ego_translation, ego_rotation)
            tracks[step, agent_idx] = np.array([box.x, box.y, box.length, box.width, box.yaw], dtype=np.float32)
    return tracks


def rasterize_future_occupancy(rasterizer: BEVRasterizer, bev: np.ndarray, future_agents: np.ndarray) -> np.ndarray:
    horizon = future_agents.shape[0]
    occ = np.zeros((horizon, 3, rasterizer.spec.height, rasterizer.spec.width), dtype=np.float32)
    drivable_idx = rasterizer.channel_to_idx.get("drivable")
    route_idx = rasterizer.channel_to_idx.get("route")
    for step in range(horizon):
        tmp = rasterizer.empty()
        boxes = [
            AgentBox(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]))
            for a in future_agents[step]
            if a[2] > 0 and a[3] > 0
        ]
        rasterizer.rasterize_boxes(tmp, boxes)
        occ[step, 0] = tmp[rasterizer.channel_to_idx.get("agents", 0)]
        if drivable_idx is not None:
            occ[step, 1] = 1.0 - bev[drivable_idx]
        if route_idx is not None:
            occ[step, 2] = bev[route_idx]
    return occ


def sample_map_masks(nusc_map, rasterizer: BEVRasterizer, ego_xy_global: np.ndarray, ego_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    height = rasterizer.spec.height
    width = rasterizer.spec.width
    patch_box = (
        float(ego_xy_global[0]),
        float(ego_xy_global[1]),
        float(rasterizer.spec.x_max - rasterizer.spec.x_min),
        float(rasterizer.spec.y_max - rasterizer.spec.y_min),
    )
    patch_angle = float(np.degrees(ego_yaw))
    try:
        masks = nusc_map.get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=["drivable_area", "lane", "lane_connector"],
            canvas_size=(height, width),
        ).astype(np.float32)
        drivable = masks[0]
        lane = np.maximum(masks[1], masks[2])
    except Exception:
        drivable = np.ones((height, width), dtype=np.float32)
        lane = np.zeros((height, width), dtype=np.float32)
    return drivable, lane


def trajectory_offroad(future_ego: np.ndarray, drivable: np.ndarray, rasterizer: BEVRasterizer) -> float:
    rc = np.round(rasterizer.xy_to_rc(future_ego[:, :2])).astype(int)
    bad = 0
    total = 0
    for r, c in rc:
        total += 1
        if r < 0 or r >= drivable.shape[0] or c < 0 or c >= drivable.shape[1] or drivable[r, c] < 0.5:
            bad += 1
    return float(bad > 0 and total > 0)


def trajectory_collision(future_ego: np.ndarray, future_agents: np.ndarray) -> float:
    return trajectory_box_collision_np(future_ego, future_agents)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.synthetic:
        write_synthetic_cache(cfg, args.split, args.max_samples)
        return
    if not cfg.data.nuscenes_root:
        raise SystemExit("data.nuscenes_root is empty. Fill configs/default.yaml on the server or use --synthetic.")
    if not cfg.data.cache_root:
        raise SystemExit("data.cache_root is empty. Fill configs/default.yaml on the server or use --synthetic.")

    try:
        from nuscenes.map_expansion.map_api import NuScenesMap  # type: ignore
        from nuscenes.nuscenes import NuScenes  # type: ignore
        from nuscenes.utils.splits import create_splits_scenes  # type: ignore
        from pyquaternion import Quaternion
    except ImportError as exc:
        raise SystemExit("Install nuScenes dependencies first: pip install nuscenes-devkit pyquaternion") from exc

    nusc = NuScenes(version=str(cfg.data.version), dataroot=str(cfg.data.nuscenes_root), verbose=True)
    spec = bev_spec_from_config(cfg)
    rasterizer = BEVRasterizer(spec)
    out_dir = ensure_dir(Path(cfg.data.cache_root) / args.split)
    category_prefixes = agent_category_prefixes(cfg)

    split_scenes = set(create_splits_scenes().get(args.split, []))
    maps = {
        name: NuScenesMap(dataroot=str(cfg.data.nuscenes_root), map_name=name)
        for name in ["boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"]
    }
    full_split = args.max_samples <= 0
    sample_tokens = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        if split_scenes and scene["name"] not in split_scenes:
            continue
        sample_tokens.append(sample["token"])
        if not full_split and len(sample_tokens) >= args.max_samples:
            break
    limit_text = "full split" if full_split else f"up to {args.max_samples} samples"
    print(f"Selected {len(sample_tokens)} sample tokens for {args.split} ({limit_text}).")

    written = 0
    for token in tqdm(sample_tokens, desc=f"writing nuScenes cache {args.split}"):
        sample = nusc.get("sample", token)
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", lidar["ego_pose_token"])
        ego_translation = np.asarray(pose["translation"], dtype=np.float32)
        ego_rotation = Quaternion(pose["rotation"])
        ego_yaw = yaw_from_quaternion(ego_rotation)

        future_ego = future_ego_trajectory(nusc, sample, ego_translation, ego_rotation, int(cfg.planning.horizon_steps))
        if future_ego is None:
            continue
        future_agents = future_agent_tracks(
            nusc,
            sample,
            ego_translation,
            ego_rotation,
            int(cfg.planning.horizon_steps),
            category_prefixes,
        )

        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        nusc_map = maps[log["location"]]
        drivable, lane = sample_map_masks(nusc_map, rasterizer, ego_translation[:2], ego_yaw)

        bev = rasterizer.empty()
        if "drivable" in rasterizer.channel_to_idx:
            bev[rasterizer.channel_to_idx["drivable"]] = drivable
        if "lane" in rasterizer.channel_to_idx:
            bev[rasterizer.channel_to_idx["lane"]] = lane
        if "route" in rasterizer.channel_to_idx:
            rasterizer.rasterize_polyline(bev, future_ego[:, :2], "route", radius=1)
        current_boxes = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if not is_traffic_participant(ann, category_prefixes):
                continue
            box = annotation_to_agent_box(ann, ego_translation, ego_rotation)
            if spec.x_min <= box.x <= spec.x_max and spec.y_min <= box.y <= spec.y_max:
                current_boxes.append(box)
        rasterizer.rasterize_boxes(bev, current_boxes)
        rasterizer.rasterize_ego(bev)
        future_occupancy = rasterize_future_occupancy(rasterizer, bev, future_agents)
        risk = np.array(
            [
                trajectory_collision(future_ego, future_agents),
                trajectory_offroad(future_ego, drivable, rasterizer),
                0.0,
                float(np.clip(future_ego[-1, 0] / max(spec.x_max, 1.0), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        ego = np.array([0.0, 0.0, 0.0, future_ego[0, 3]], dtype=np.float32)
        np.savez_compressed(
            out_dir / f"{written:08d}.npz",
            bev=bev,
            ego=ego,
            future_ego=future_ego,
            future_agents=future_agents,
            future_occupancy=future_occupancy,
            risk=risk,
            scene_id=np.array(sample["scene_token"]),
            sample_token=np.array(token),
        )
        written += 1
    print(f"Wrote {written} real nuScenes cache samples to {out_dir}")


if __name__ == "__main__":
    main()
