# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance


from models.loss_utils import sigmoid, compute_aff

from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)





def make_loss():
    if args.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(args.margin, args.normalize_embeddings)
    elif args.loss == 'MultiBatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = MultiBatchHardTripletLossWithMasks(args.margin, args.normalize_embeddings, args.weights)
        print('MultiBatchHardTripletLossWithMasks')
        print('Weights (final/cloud/image): {}'.format(args.weights))
    elif args.loss == 'TruncatedSmoothAP':
        loss_fn = TruncatedSmoothAP(tau1=args.ap_tau1, 
                                    similarity=args.ap_similarity,
                                    positives_per_query=args.ap_positives_per_query)
    else:
        raise NotImplementedError
    
    return loss_fn




class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class MultiBatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings, weights):
        assert len(weights) == 3
        self.weights = weights
        self.final_loss = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)
        self.cloud_loss = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)
        self.image_loss = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)



    def __call__(self, x, positives_mask, negatives_mask):
        # Loss on the final global descriptor
        final_loss, final_stats, final_hard_triplets = self.final_loss(x['embedding'], positives_mask, negatives_mask)
        final_stats = {'final_{}'.format(e): final_stats[e] for e in final_stats}

        loss = 0.

        stats = final_stats
        if self.weights[0] > 0.:
            loss = self.weights[0] * final_loss + loss

        # Loss on the cloud-based descriptor
        if 'cloud_embedding' in x:
            cloud_loss, cloud_stats, _ = self.cloud_loss(x['cloud_embedding'], positives_mask, negatives_mask)
            cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
            stats.update(cloud_stats)
            if self.weights[1] > 0.:
                loss = self.weights[1] * cloud_loss + loss

        # Loss on the image-based descriptor
        if 'image_embedding' in x:
            image_loss, image_stats, _ = self.image_loss(x['image_embedding'], positives_mask, negatives_mask)
            image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
            stats.update(image_stats)
            if self.weights[2] > 0.:
                loss = self.weights[2] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None






class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.loss_fn = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask):
        embeddings = x['embedding']
        return self.loss_fn(embeddings, positives_mask, negatives_mask)






class BatchHardTripletLossWithMasksHelper:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist,
                 'normalized_loss': loss.item() * self.loss_fn.reducer.triplets_past_filter,
                 # total loss per batch
                 'total_loss': self.loss_fn.reducer.loss * self.loss_fn.reducer.triplets_past_filter
                 }

        return loss, stats, hard_triplets







class TruncatedSmoothAP:
    def __init__(self, tau1: float = 0.01, similarity: str = 'cosine', positives_per_query: int = 4):
        # We reversed the notation compared to the paper (tau1 is sigmoid on similarity differences)
        # tau1: sigmoid temperature applied on similarity differences
        # positives_per_query: number of positives per query to consider
        # negatives_only: if True in denominator we consider positives and negatives; if False we consider all elements
        #                 (with except to the anchor itself)

        self.tau1 = tau1
        self.similarity = similarity
        self.positives_per_query = positives_per_query

    def __call__(self, embeddings, positives_mask, negatives_mask):
        embeddings = embeddings['embedding']

        device = embeddings.device

        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # Ranking of the retrieval set
        # For each element we ignore elements that are neither positives nor negatives

        # Compute cosine similarity scores
        # 1st dimension corresponds to q, 2nd dimension to z
        s_qz = compute_aff(embeddings, similarity=self.similarity)

        # Find the positives_per_query closest positives for each query
        s_positives = s_qz.detach().clone()
        s_positives.masked_fill_(torch.logical_not(positives_mask), np.NINF)
        #closest_positives_ndx = torch.argmax(s_positives, dim=1).view(-1, 1)  # Indices of closests positives for each query
        closest_positives_ndx = torch.topk(s_positives, k=self.positives_per_query, dim=1, largest=True, sorted=True)[1]
        # closest_positives_ndx is (batch_size, positives_per_query)  with positives_per_query closest positives
        # per each batch element

        n_positives = positives_mask.sum(dim=1)     # Number of positives for each anchor

        # Compute the rank of each example x with respect to query element q as per Eq. (2)
        s_diff = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
        s_sigmoid = sigmoid(s_diff, temp=self.tau1)

        # Compute the nominator in Eq. 2 and 5 - for q compute the ranking of each of its positives with respect to other positives of q
        # Filter out z not in Positives
        pos_mask = positives_mask.unsqueeze(1)
        pos_s_sigmoid = s_sigmoid * pos_mask

        # Filter out z on the same position as the positive (they have value = 0.5, as the similarity difference is zero)
        mask = torch.ones_like(pos_s_sigmoid).scatter(2, closest_positives_ndx.unsqueeze(2), 0.)
        pos_s_sigmoid = pos_s_sigmoid * mask

        # Compute the rank for each query and its positives_per_query closest positive examples with respect to other positives
        r_p = torch.sum(pos_s_sigmoid, dim=2) + 1.
        # r_p is (batch_size, positives_per_query) matrix

        # Consider only positives and negatives in the denominator
        # Compute the denominator in Eq. 5 - add sum of Indicator function for negatives (or non-positives)
        neg_mask = negatives_mask.unsqueeze(1)
        neg_s_sigmoid = s_sigmoid * neg_mask
        r_omega = r_p + torch.sum(neg_s_sigmoid, dim=2)

        # Compute R(i, S_p) / R(i, S_omega) ration in Eq. 2
        r = r_p / r_omega

        # Compute metrics              mean ranking of the positive example, recall@1
        stats = {}
        # Mean number of positives per query
        stats['positives_per_query'] = n_positives.float().mean(dim=0).item()
        # Mean ranking of selected positive examples (closests positives)
        temp = s_diff.detach() > 0
        temp = torch.logical_and(temp[:, 0], negatives_mask)        # Take the best positive
        hard_ranking = temp.sum(dim=1)
        stats['best_positive_ranking'] = hard_ranking.float().mean(dim=0).item()
        # Recall at 1
        stats['recall'] = {1: (hard_ranking <= 1).float().mean(dim=0).item()}

        # r is (N, positives_per_query) tensor
        # Zero entries not corresponding to real positives - this happens when the number of true positives is lower than positives_per_query
        valid_positives_mask = torch.gather(positives_mask, 1, closest_positives_ndx)   # () tensor
        masked_r = r * valid_positives_mask
        n_valid_positives = valid_positives_mask.sum(dim=1)

        # Filter out rows (queries) without any positive to avoid division by zero
        valid_q_mask = n_valid_positives > 0
        masked_r = masked_r[valid_q_mask]

        ap = (masked_r.sum(dim=1) / n_valid_positives[valid_q_mask]).mean()
        loss = 1. - ap

        stats['loss'] = loss.item()
        stats['ap'] = ap.item()
        stats['avg_embedding_norm'] = embeddings.norm(dim=1).mean().item()
        
        return loss, stats, None
