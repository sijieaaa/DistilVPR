

import torch
import torch.nn.functional as F


from hyptorch.nn import ToPoincare

from hyptorch.pmath import dist, dist_matrix


from selfagent.rkdg import compute_rkdg_loss



from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)








# ----------------- cross logit dist ----------------- #

def compute_crosslogitdist_loss(logit_stu, logit_tea, loss_type, logit_norm_fn):
    assert logit_stu.shape == logit_tea.shape
    assert len(logit_stu.shape) == 2

    if loss_type == 'MSE':
        distil_loss_fn = torch.nn.MSELoss()
    elif loss_type == 'L1':
        distil_loss_fn = torch.nn.L1Loss()
    elif loss_type == 'KLDiv':
        distil_loss_fn = torch.nn.KLDivLoss()
    else: 
        raise NotImplementedError

    if logit_norm_fn == 'softmax':
        logit_norm_fn_stu = F.softmax
        logit_norm_fn_tea = F.softmax
    elif logit_norm_fn == 'log_softmax':
        logit_norm_fn_stu = F.log_softmax
        logit_norm_fn_tea = F.softmax
    elif logit_norm_fn == 'identity':
        logit_norm_fn_stu = lambda x, dim: x
        logit_norm_fn_tea = lambda x, dim: x
    elif logit_norm_fn == 'l2':
        logit_norm_fn_stu = F.normalize
        logit_norm_fn_tea = F.normalize # p=2, dim=1x
    else:
        raise NotImplementedError
    

    
    if args.logit_mapping_fn == 'identity':
        logit_mapping_fns_stu = [lambda x: x]
        logit_mapping_fns_tea = [lambda x: x]
    elif args.logit_mapping_fn == 'topoincare':
        logit_mapping_fns_stu = [ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)]
        logit_mapping_fns_tea = [ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)]
    elif args.logit_mapping_fn == 'id_topoin':
        logit_mapping_fns_stu = [
            lambda x: x,
            ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)
        ]
        logit_mapping_fns_tea = [
            lambda x: x,
            ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)
        ]


    weights_st2ss = args.crosslogitdistloss_weight_st2ss
    weights_st2ss = weights_st2ss.split('_')
    weights_st2ss = [float(w) for w in weights_st2ss]
    assert len(weights_st2ss) == len(logit_mapping_fns_stu)

    weights_tt2st = args.crosslogitdistloss_weight_tt2st
    weights_tt2st = weights_tt2st.split('_')
    weights_tt2st = [float(w) for w in weights_tt2st]
    assert len(weights_tt2st) == len(logit_mapping_fns_stu)

    

    logit_stu_normalized = logit_norm_fn_stu(logit_stu, dim=1)
    logit_tea_normalized = logit_norm_fn_tea(logit_tea, dim=1)



    distil_loss_st2ss = 0
    distil_loss_tt2st = 0
    i_fn = 0
    for logit_mapping_fn_stu, logit_mapping_fn_tea in zip(logit_mapping_fns_stu, logit_mapping_fns_tea):

        logit_stu_normalized_mapped = logit_mapping_fn_stu(logit_stu_normalized)
        logit_tea_normalized_mapped = logit_mapping_fn_tea(logit_tea_normalized)


        # for full matrix
        diffmat_ss = logit_stu_normalized_mapped.unsqueeze(0) - logit_stu_normalized_mapped.unsqueeze(1)
        distmat_ss = torch.norm(diffmat_ss, p=2, dim=-1)
        diffmat_st = logit_stu_normalized_mapped.unsqueeze(0) - logit_tea_normalized_mapped.unsqueeze(1)
        distmat_st = torch.norm(diffmat_st, p=2, dim=-1)
        diffmat_tt = logit_tea_normalized_mapped.unsqueeze(0) - logit_tea_normalized_mapped.unsqueeze(1)
        distmat_tt = torch.norm(diffmat_tt, p=2, dim=-1)


        _distil_loss_st2ss = distil_loss_fn(distmat_ss, distmat_st.detach()) * weights_st2ss[i_fn]
        _distil_loss_tt2st = distil_loss_fn(distmat_st, distmat_tt.detach()) * weights_tt2st[i_fn]   

        distil_loss_st2ss += _distil_loss_st2ss
        distil_loss_tt2st += _distil_loss_tt2st

        i_fn += 1


    return distil_loss_st2ss, distil_loss_tt2st














# ----------------- cross logit sim ----------------- #

def compute_crosslogitsim_loss(logit_stu, logit_tea, loss_type, logit_norm_fn):
    assert logit_stu.shape == logit_tea.shape
    assert len(logit_stu.shape) == 2

    if loss_type == 'MSE':
        distil_loss_fn = torch.nn.MSELoss()
    elif loss_type == 'L1':
        distil_loss_fn = torch.nn.L1Loss()
    elif loss_type == 'KLDiv':
        distil_loss_fn = torch.nn.KLDivLoss()
    else: 
        raise NotImplementedError

    if logit_norm_fn == 'softmax':
        logit_norm_fn_stu = F.softmax
        logit_norm_fn_tea = F.softmax
    elif logit_norm_fn == 'log_softmax':
        logit_norm_fn_stu = F.log_softmax
        logit_norm_fn_tea = F.softmax
    elif logit_norm_fn == 'identity':
        logit_norm_fn_stu = lambda x, dim: x
        logit_norm_fn_tea = lambda x, dim: x
    elif logit_norm_fn == 'l2':
        logit_norm_fn_stu = F.normalize
        logit_norm_fn_tea = F.normalize # p=2, dim=1x
    else:
        raise NotImplementedError
    
    if args.logit_mapping_fn == 'identity':
        logit_mapping_fns_stu = [lambda x: x]
        logit_mapping_fns_tea = [lambda x: x]
    elif args.logit_mapping_fn == 'topoincare':
        logit_mapping_fns_stu = [ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)]
        logit_mapping_fns_tea = [ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)]
    elif args.logit_mapping_fn == 'id_topoin':
        logit_mapping_fns_stu = [
            lambda x: x,
            ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)
        ]
        logit_mapping_fns_tea = [
            lambda x: x,
            ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)
        ]

    weights_st2ss = args.crosslogitsimloss_weight_st2ss
    weights_st2ss = weights_st2ss.split('_')
    weights_st2ss = [float(w) for w in weights_st2ss]
    assert len(weights_st2ss) == len(logit_mapping_fns_stu)

    weights_tt2st = args.crosslogitsimloss_weight_tt2st
    weights_tt2st = weights_tt2st.split('_')
    weights_tt2st = [float(w) for w in weights_tt2st]
    assert len(weights_tt2st) == len(logit_mapping_fns_stu)



    logit_stu_normalized = logit_norm_fn_stu(logit_stu, dim=1)
    logit_tea_normalized = logit_norm_fn_tea(logit_tea, dim=1)


    distil_loss_st2ss = 0
    distil_loss_tt2st = 0

    i_fn = 0
    for logit_mapping_fn_stu, logit_mapping_fn_tea in zip(logit_mapping_fns_stu, logit_mapping_fns_tea):
        logit_stu_normalized_mapped = logit_mapping_fn_stu(logit_stu_normalized)
        logit_tea_normalized_mapped = logit_mapping_fn_tea(logit_tea_normalized)


        norm_s = torch.norm(logit_stu_normalized_mapped, p=2, dim=1, keepdim=True) # [b,1]
        normsqmat_ss = norm_s @ norm_s.T # [b,b]
        norm_t = torch.norm(logit_tea_normalized_mapped, p=2, dim=1, keepdim=True)
        normsqmat_tt = norm_t @ norm_t.T
        normsqmat_st = norm_s @ norm_t.T


        # for full matrix
        simmat_ss = logit_stu_normalized_mapped @ logit_stu_normalized_mapped.T
        simmat_st = logit_stu_normalized_mapped @ logit_tea_normalized_mapped.T
        simmat_tt = logit_tea_normalized_mapped @ logit_tea_normalized_mapped.T

        if args.crosslogitsimloss_divide:
            simmat_ss = simmat_ss / normsqmat_ss
            simmat_st = simmat_st / normsqmat_st
            simmat_tt = simmat_tt / normsqmat_tt


        _distil_loss_st2ss = distil_loss_fn(simmat_ss, simmat_st.detach()) * weights_st2ss[i_fn]
        _distil_loss_tt2st = distil_loss_fn(simmat_st, simmat_tt.detach()) * weights_tt2st[i_fn]


        distil_loss_st2ss += _distil_loss_st2ss
        distil_loss_tt2st += _distil_loss_tt2st

        i_fn += 1
    

    return distil_loss_st2ss, distil_loss_tt2st
















# ----------------- cross logit geodesic ----------------- #

def compute_crosslogitgeodist_loss(logit_stu, logit_tea, loss_type, logit_norm_fn):
    assert logit_stu.shape == logit_tea.shape
    assert len(logit_stu.shape) == 2

    if loss_type == 'MSE':
        distil_loss_fn = torch.nn.MSELoss()
    elif loss_type == 'L1':
        distil_loss_fn = torch.nn.L1Loss()
    elif loss_type == 'KLDiv':
        distil_loss_fn = torch.nn.KLDivLoss()
    else: 
        raise NotImplementedError

    if logit_norm_fn == 'softmax':
        logit_norm_fn_stu = F.softmax
        logit_norm_fn_tea = F.softmax
    elif logit_norm_fn == 'log_softmax':
        logit_norm_fn_stu = F.log_softmax
        logit_norm_fn_tea = F.softmax
    elif logit_norm_fn == 'identity':
        logit_norm_fn_stu = lambda x, dim: x
        logit_norm_fn_tea = lambda x, dim: x
    elif logit_norm_fn == 'l2':
        logit_norm_fn_stu = F.normalize
        logit_norm_fn_tea = F.normalize # p=2, dim=1x
    else:
        raise NotImplementedError
    

    
    # geodesic distance is for poincare ball
    logit_mapping_fns_stu = [ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)]
    logit_mapping_fns_tea = [ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)]



    weights_st2ss = args.crosslogitgeodistloss_weight_st2ss
    weights_st2ss = weights_st2ss.split('_')
    weights_st2ss = [float(w) for w in weights_st2ss]
    

    weights_tt2st = args.crosslogitgeodistloss_weight_tt2st
    weights_tt2st = weights_tt2st.split('_')
    weights_tt2st = [float(w) for w in weights_tt2st]


    

    logit_stu_normalized = logit_norm_fn_stu(logit_stu, dim=1)
    logit_tea_normalized = logit_norm_fn_tea(logit_tea, dim=1)



    distil_loss_st2ss = 0
    distil_loss_tt2st = 0
    i_fn = 0
    for logit_mapping_fn_stu, logit_mapping_fn_tea in zip(logit_mapping_fns_stu, logit_mapping_fns_tea):

        logit_stu_normalized_mapped = logit_mapping_fn_stu(logit_stu_normalized)
        logit_tea_normalized_mapped = logit_mapping_fn_tea(logit_tea_normalized)


        # for full matrix
        geodistmat_ss = dist_matrix(logit_stu_normalized_mapped, logit_stu_normalized_mapped, c=args.curvature) # [b,b]
        geodistmat_st = dist_matrix(logit_stu_normalized_mapped, logit_tea_normalized_mapped, c=args.curvature)
        geodistmat_tt = dist_matrix(logit_tea_normalized_mapped, logit_tea_normalized_mapped, c=args.curvature)



        _distil_loss_st2ss = distil_loss_fn(geodistmat_ss, geodistmat_st.detach()) * weights_st2ss[i_fn]
        _distil_loss_tt2st = distil_loss_fn(geodistmat_st, geodistmat_tt.detach()) * weights_tt2st[i_fn]   

        distil_loss_st2ss += _distil_loss_st2ss
        distil_loss_tt2st += _distil_loss_tt2st

        i_fn += 1


    return distil_loss_st2ss, distil_loss_tt2st




















def compute_singledigit_loss(logit_stu, logit_tea, positives_mask=None, negatives_mask=None, adaptor=None):



    # crosslogitdist distillation
    crosslogitdistloss_st2ss,  crosslogitdistloss_tt2st = compute_crosslogitdist_loss(
            logit_stu, logit_tea, args.crosslogitdistloss_type, args.crosslogitdistloss_logit_norm_fn)


    # crosslogitsim distillation
    crosslogitsimloss_st2ss,  crosslogitsimloss_tt2st = compute_crosslogitsim_loss(
            logit_stu, logit_tea, args.crosslogitsimloss_type, args.crosslogitsimloss_logit_norm_fn)




    # crosslogitgeodist distillation
    crosslogitgeodistloss_st2ss,  crosslogitgeodistloss_tt2st = compute_crosslogitgeodist_loss(
            logit_stu, logit_tea, args.crosslogitgeodistloss_type, args.crosslogitgeodistloss_logit_norm_fn)



    distil_loss = 0


    distil_loss += crosslogitdistloss_st2ss
    distil_loss += crosslogitdistloss_tt2st


    distil_loss += crosslogitsimloss_st2ss
    distil_loss += crosslogitsimloss_tt2st


    distil_loss += crosslogitgeodistloss_st2ss
    distil_loss += crosslogitgeodistloss_tt2st


    return distil_loss





def compute_distil_loss(output_dict_stu, output_dict_tea, positives_mask=None, negatives_mask=None, adaptor=None):



    logit_stu = output_dict_stu['embedding']
    device = logit_stu.device
    logit_tea = output_dict_tea['embedding'].to(device)



    # embedding
    distil_loss_logit = compute_singledigit_loss(logit_stu, logit_tea, positives_mask, negatives_mask, adaptor)



    distil_loss = (
        distil_loss_logit * args.distil_logit_weight
    )


    return distil_loss




def compute_task_loss(output_dict_stu, output_dict_tea, positives_mask=None, negatives_mask=None, adaptor=None, task_loss_fn_stu=None):
    
    # logit/embedding
    logit_stu = output_dict_stu['embedding']
    logit_stu_dict = {'embedding': logit_stu}
    loss_logit, _, _ = task_loss_fn_stu(logit_stu_dict, positives_mask, negatives_mask)



    task_loss = (
        loss_logit * args.task_logit_weight
    )

    return task_loss







def compute_all_loss(output_dict_stu, output_dict_tea, positives_mask=None, negatives_mask=None, adaptor=None, task_loss_fn_stu=None):

    positives_mask = positives_mask.to(output_dict_stu['embedding'].device)
    negatives_mask = negatives_mask.to(output_dict_stu['embedding'].device)

    
    # task loss
    task_loss = compute_task_loss(output_dict_stu, output_dict_tea, positives_mask, negatives_mask, adaptor, task_loss_fn_stu)


    # distil loss
    if args.distil_method == 'ours':
        distil_loss = compute_distil_loss(output_dict_stu, output_dict_tea, positives_mask, negatives_mask, adaptor)

    elif args.distil_method == 'ours_rkdg':
        distil_loss_ours = compute_distil_loss(output_dict_stu, output_dict_tea, positives_mask, negatives_mask, adaptor)
        distil_loss_rkdg = compute_rkdg_loss(output_dict_stu, output_dict_tea)
        distil_loss_rkdg *= args.rkdgloss_weight
        distil_loss = distil_loss_ours + distil_loss_rkdg


    elif args.distil_method == 'nokd':
        distil_loss = 0

    elif args.distil_method == 'rkdg':
        distil_loss = compute_rkdg_loss(output_dict_stu, output_dict_tea)
        distil_loss *= args.rkdgloss_weight

    else:
        raise NotImplementedError



    loss = 0
    loss += task_loss
    loss += distil_loss


    return loss




