# import


# [[file:~/Works/char2token2mention/utils/loss_func.org::*import][import:1]]
import torch
from torch.nn.functional import normalize

# import:1 ends here

# mcl


# [[file:~/Works/char2token2mention/utils/loss_func.org::*mcl][mcl:1]]
def mcl(
    output_mat,
    label_range,
    label,
    concept_mask,
    concept_mask_not,
    eye_mask,
    eps,
):
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    pred = (pred > label_range).sum(0)
    acc = (pred == label).float().mean()
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_score = pos_score.clamp_min(-1 + eps)
    pos_loss = -((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum()
    pos_loss = pos_loss / (concept_mask_not.sum() - concept_mask_not.shape[0])
    neg_score = (output_mat_sim * concept_mask).clamp(0, 1 - eps)
    neg_mask = neg_score > 0
    neg_loss = -(neg_score * torch.log(1 - neg_score)).sum() / neg_mask.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result


# mcl:1 ends here

# cel

# cosine embedding loss


# [[file:~/Works/char2token2mention/utils/loss_func.org::*cel][cel:1]]
def cel(
    output_mat,
    label_range,
    label,
    concept_mask,
    concept_mask_not,
    eye_mask,
    eps,
):
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    pred = (pred > label_range).sum(0)
    acc = (pred == label).float().mean()
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = (1 - pos_score).sum()
    neg_score = (output_mat_sim * concept_mask).clamp_min(0)
    neg_mask = neg_score > 0
    neg_loss = neg_score.sum()
    prob = neg_mask.sum() / concept_mask.sum()
    loss = (pos_loss + neg_loss) / ((dt_size - 1) * dt_size)

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result


# cel:1 ends here

# mrl

# margin ranking loss


# [[file:~/Works/char2token2mention/utils/loss_func.org::*mrl][mrl:1]]
def mrl(
    output_mat, label_range, label, concept_mask, concept_mask_not, eye_mask
):
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    pred = (pred > label_range).sum(0)
    acc = (pred == label).float().mean()

    diag = torch.diag(output_mat_sim)
    pos_score = output_mat_sim * concept_mask_not
    pos_score = pos_score.sum(-1) - diag
    pos_score[pos_score == 0] = diag[pos_score == 0]

    score = (1 - pos_score).unsqueeze(-1) + output_mat_sim
    score_mask = concept_mask * (score > 0)
    score = score * score_mask
    score_max, _ = score.max(-1)
    loss = score_max.mean()

    result = {
        "loss": loss,
        "acc": acc,
    }
    return result


# mrl:1 ends here

# [[file:~/Works/char2token2mention/utils/loss_func.org::*mrl][mrl:2]]
def mrlV2(
    output_mat, label_range, label, concept_mask, concept_mask_not, eye_mask
):
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    pred = (pred > label_range).sum(0)
    acc = (pred == label).float().mean()

    diag = torch.diag(output_mat_sim)
    pos_score = output_mat_sim * concept_mask_not
    pos_score = pos_score.sum(-1) - diag
    pos_score[pos_score == 0] = diag[pos_score == 0]

    neg_score, _ = (output_mat_sim * concept_mask).max(-1)
    score = (1 - pos_score).unsqueeze(-1) + neg_score
    score_mask = score > 0
    neg_score_log = -torch.log(
        (1 - neg_score.clamp_max(1 - 1e-4) * score_mask)
    )
    score = score * score_mask * neg_score_log
    loss = score.mean()

    result = {
        "loss": loss,
        "acc": acc,
    }
    return result


# mrl:2 ends here
