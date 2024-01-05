

import torch
import torch.nn.functional as F
from hyptorch.nn import ToPoincare
from hyptorch.pmath import dist, dist_matrix



from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)




def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res





def compute_rkdg_loss(output_dict_stu, output_dict_tea, squared=False, eps=1e-12, distance_weight=1, angle_weight=1, geodesic_weight=1):
    f_s = output_dict_stu['embedding']
    f_t = output_dict_tea['embedding']

    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)





    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss_d = F.smooth_l1_loss(d, t_d)



    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss_a = F.smooth_l1_loss(s_angle, t_angle)




    # -- RKD geodesic distance loss
    to_poincare = ToPoincare(c=args.curvature, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)
    # to_poincare = ToPoincare(c=1, ball_dim=args.student_output_dim, riemannian=False, clip_r=None)
    logit_s_poincare = to_poincare(stu)
    logit_t_poincare = to_poincare(tea)

    geodistmat_ss = dist_matrix(logit_s_poincare, logit_s_poincare, c=args.curvature) # [b,b]
    geodistmat_tt = dist_matrix(logit_t_poincare, logit_t_poincare, c=args.curvature)

    mean_ss = geodistmat_ss[geodistmat_ss > 0].mean()
    mean_tt = geodistmat_tt[geodistmat_tt > 0].mean()

    geodistmat_ss = geodistmat_ss / mean_ss
    geodistmat_tt = geodistmat_tt / mean_tt

    loss_g = F.smooth_l1_loss(geodistmat_ss, geodistmat_tt.detach())







    loss = distance_weight * loss_d \
            + angle_weight * loss_a \
            + geodesic_weight * loss_g


    return loss







if __name__ == '__main__':
    

    b = 10
    c = 128

    output_dict_stu = {'embedding': torch.randn(b, c)}

    output_dict_tea = {'embedding': torch.randn(b, c)}


    loss = compute_rkdg_loss(output_dict_stu, output_dict_tea, squared=False, eps=1e-12, distance_weight=1, angle_weight=1)
