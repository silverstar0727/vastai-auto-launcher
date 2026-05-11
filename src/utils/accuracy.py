import torch


def recalls_and_ndcgs_for_ks(batch_rank_items, batch_positive_items, ks=[5, 10, 25]):
    """
    # Args
        batch_rank_items : torch Tensor, shape of (N, n_valid_items)
            score가 높은 순으로 정렬된 item indexes
        batch_positive_items : torch Tensor, shape of (N, n_positive_samples)
            positive sample index
        ks : list
    # Returns
        metrics : dict
    """

    # (N, n_positive_items)
    n_batches = batch_positive_items.shape[0]
    batch_n_positives = torch.full(
        (n_batches, 1),
        batch_positive_items.shape[1],
        device=batch_positive_items.device,
        dtype=torch.long,
    )

    metrics = {}
    for k in sorted(ks, reverse=True):
        # (N, k)
        batch_rank_items_at_k = batch_rank_items[:, :k]

        # (N, k) --> (N, k, 1)
        batch_rank_items_at_k = batch_rank_items_at_k.view(batch_rank_items_at_k.shape[0], k, 1)

        # (N, n_pos) --> (N, 1, n_pos)
        batch_positive_items = batch_positive_items.view(batch_rank_items_at_k.shape[0], 1, -1)

        # hits : (N, k, n_pos)
        batch_hits = batch_rank_items_at_k == batch_positive_items
        # hits : (N, k)
        batch_hits = batch_hits.sum(dim=-1)

        metrics["Recall_%d" % k] = _recall(batch_hits, batch_n_positives, k)
        metrics["NDCG_%d" % k] = _ndcg(batch_hits, batch_n_positives, k)
    return metrics


def _recall(hits, answer_count, k):
    n1 = hits.sum(1)
    n2 = torch.min(torch.Tensor([k]).to(hits.device), answer_count.float())

    recall = (n1 / n2).mean()
    return recall


def _ndcg(hits, answer_count, k):
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
    ndcg = (dcg / idcg).mean()
    return ndcg


def hr_mrr_for_ks(batch_rank_items, batch_positive_items, ks=[1, 50]):
    """HR@k (Hit Rate, top-k 에 정답 하나라도 있으면 1) + MRR (정답 첫 등장 reciprocal rank)."""
    n_batches = batch_rank_items.shape[0]
    metrics = {}

    for k in ks:
        topk = batch_rank_items[:, :k]                                # (N, k)
        topk_3d = topk.view(n_batches, k, 1)                          # (N, k, 1)
        pos_3d = batch_positive_items.view(n_batches, 1, -1)          # (N, 1, n_pos)
        matches = (topk_3d == pos_3d).any(dim=-1)                     # (N, k) bool

        hr = matches.any(dim=-1).float().mean()
        metrics[f"HR_{k}"] = hr

    # MRR: 가장 큰 k 기준으로 first-hit reciprocal rank
    k_max = max(ks)
    topk = batch_rank_items[:, :k_max]
    topk_3d = topk.view(n_batches, k_max, 1)
    pos_3d = batch_positive_items.view(n_batches, 1, -1)
    matches = (topk_3d == pos_3d).any(dim=-1).float()                 # (N, k)
    positions = torch.arange(1, k_max + 1, device=matches.device).float()
    # 일치 위치에 reciprocal rank, 외 0
    rr = matches * (1.0 / positions[None, :])
    # 각 user 의 best (first hit 가 가장 큰 1/rank)
    mrr = rr.max(dim=-1).values.mean()
    metrics[f"MRR_{k_max}"] = mrr
    return metrics
