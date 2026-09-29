from __future__ import annotations

import torch
import torch.nn.functional as F


def world_model_loss(
    rollout: dict[str, torch.Tensor],
    target_occupancy: torch.Tensor,
    target_risk: torch.Tensor,
    candidate_is_expert: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Multi-task world-model loss.

    Args:
        rollout: model outputs for [B,N,T,...].
        target_occupancy: [B,T,C,H,W].
        target_risk: [B,4], collision/offroad/redlight/progress for expert.
        candidate_is_expert: [B,N] marks the expert candidate.
    """
    occ_pred = rollout["future_occupancy"]
    risk_pred = rollout["risk"]
    b, n, t, c, h, w = occ_pred.shape
    occ_target = target_occupancy
    if occ_target.shape[-2:] != (h, w):
        occ_target = F.interpolate(
            occ_target.flatten(0, 1),
            size=(h, w),
            mode="nearest",
        ).view(occ_target.shape[0], occ_target.shape[1], occ_target.shape[2], h, w)
    occ_target = occ_target[:, None].expand_as(occ_pred)
    occ_loss = F.binary_cross_entropy_with_logits(occ_pred, occ_target)

    risk_target = target_risk[:, None, None, :3].expand(*risk_pred.shape[:3], 3)
    progress_target = target_risk[:, None, None, 3:4].expand(*risk_pred.shape[:3], 1)
    risk_loss = F.binary_cross_entropy_with_logits(risk_pred[..., :3], risk_target)
    progress_loss = F.smooth_l1_loss(risk_pred[..., 3:4], progress_target)

    uncertainty = F.softplus(risk_pred[..., 4]).mean()

    score_proxy = risk_pred[..., 3] - torch.sigmoid(risk_pred[..., :3]).sum(dim=-1)
    expert_score = (score_proxy * candidate_is_expert[..., None]).sum(dim=1)
    non_expert = (~candidate_is_expert).float()
    hard_negative = (score_proxy * non_expert[..., None]).max(dim=1).values
    rank_loss = F.relu(0.2 - expert_score + hard_negative).mean()

    loss = occ_loss + risk_loss + 0.5 * progress_loss + 0.2 * rank_loss + 0.01 * uncertainty
    return {
        "loss": loss,
        "occ": occ_loss.detach(),
        "risk": risk_loss.detach(),
        "progress": progress_loss.detach(),
        "rank": rank_loss.detach(),
        "uncertainty": uncertainty.detach(),
    }
