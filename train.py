


from tools.utils import set_seed
set_seed(7)

import os
import torch
import tqdm

import torch.optim.lr_scheduler



from evaluate import evaluate

from tools.utils import MinkLocParams

from tools.utils import get_datetime

from models.loss import make_loss
from models.model_factory import model_factory
from models.minkloc_multimodal import MinkLocMultimodal
import time

from tqdm import tqdm

from torchvision.models.resnet import ResNet
from torch.nn import Identity

import os
from network.distil_imagefes import DistilImageFE
from network.general_imagefes import GeneralImageFE



from compute_distil_loss import compute_all_loss
import torch.nn as nn


import torch
from tools.utils import *
from datasets.make_dataloaders import make_dataloaders





from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)








def test_after_epoch(model, device):

    model.eval()
    test_stats = evaluate(model, device, silent=False)

    test_stats = test_stats[args.dataset]
    recall_top1p = test_stats['ave_one_percent_recall']
    recall_topN = test_stats['ave_recall']
    recall_top1 = recall_topN[0]


    ave_recall_for_each_scene = test_stats['ave_recall_for_each_scene']
    ave_recall_for_each_scene_top1 = ave_recall_for_each_scene[:,0]

    


    return recall_top1p, recall_top1, ave_recall_for_each_scene_top1








def make_params_l(model):
    params_l = []

    if isinstance(model, DistilImageFE):
        params_l.append({'params': model.parameters(), 'lr': args.image_lr}) # 1e-4

    elif isinstance(model, GeneralImageFE):
        params_l.append({'params': model.parameters(), 'lr': args.image_lr})

    elif isinstance(model, ResNet):
        params_l.append({'params': model.parameters(), 'lr': args.image_lr})

    elif isinstance(model, Identity):
        params_l.append({'params': model.parameters(), 'lr': args.image_lr})

    
    elif isinstance(model, MinkLocMultimodal):
        params_l.append({'params': model.image_fe.parameters(), 'lr': args.image_lr})
        params_l.append({'params': model.cloud_fe.parameters(), 'lr': args.cloud_lr})



    else:
        raise NotImplementedError

    return params_l










def do_train(dataloaders, params: MinkLocParams):


    # ---- make teacher model
    if args.model == 'minklocmmcat':
        planes = [int(e) for e in args.minklocmm_cloud_planes.split('_')]
        layers = [int(e) for e in args.minklocmm_cloud_layers.split('_')]
        model = model_factory(
                            fuse_method='cat', 
                            cloud_fe_size=args.minklocmm_cloud_fe_dim, 
                            image_fe_size=args.minklocmm_image_fe_dim,
                            cloud_planes=planes,
                            cloud_layers=layers,
                            cloud_topdown=args.minklocmm_cloud_topdown,
                            image_useallstages=args.minklocmm_image_useallstages,
                            image_fe=args.minklocmm_image_fe
                            )
    else: 
        raise NotImplementedError




        
    # ---- make student model
    if args.student_model == 'None':
        model_stu = None
    elif args.student_model == 'resnet18':
        model_stu = GeneralImageFE(
                            image_fe='resnet18',
                            num_other_stage_blocks=None,
                            num_stage3_blocks=None,
                            image_pool_method='GeM',
                            image_useallstages=False,
                            output_dim=args.student_output_dim
        )
    elif args.student_model == 'resnet34':
        model_stu = GeneralImageFE(
                            image_fe='resnet34',
                            num_other_stage_blocks=None,
                            num_stage3_blocks=None,
                            image_pool_method='GeM',
                            image_useallstages=False,
                            output_dim=args.student_output_dim
        )
    else: 
        raise NotImplementedError

    



    adaptor = nn.Identity()


    if args.teacher_weights_path is not None:
        assert os.path.exists(args.teacher_weights_path)
        print('loading weights...')
        print(f'{args.teacher_weights_path}')
        teacher_weights = torch.load(args.teacher_weights_path)
        model.load_state_dict(teacher_weights)







    params_l = make_params_l(model)
    params_l_stu = make_params_l(model_stu)



    # vanilla training loss function
    print(f'loss...{args.loss}')
    if args.loss == 'MultiBatchHardTripletMarginLoss':
        loss_fn = make_loss()
        loss_fn_stu = make_loss()

    else:
        raise NotImplementedError









    optimizer = torch.optim.Adam(params_l, 
                                 weight_decay=args.weight_decay)
    optimizer_stu = torch.optim.Adam(params_l_stu, 
                                     weight_decay=args.weight_decay)


    if args.scheduler == 'MultiStepLR':
        milestones = [int(e) for e in args.milestones.split('_')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        milestones=milestones, 
                                                        gamma=0.1
                                                        )
        scheduler_stu = torch.optim.lr_scheduler.MultiStepLR(optimizer_stu, 
                                                            milestones=milestones, 
                                                            gamma=0.1
                                                            )
                                 
    else:
        raise Exception




    # -- device
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        model_stu.to(device)
        adaptor.to(device)
    else:
        device = "cpu"
    print('Model device: {}'.format(device))




    # ---- teacher embedding as None
    print('teacher embedding as None...')
    teacher_embeddings_dict = None





    # # ------------------------ getting teacher embedding first ------------------------
    if args.train_mode == 'train_student':
        assert args.teacher_weights_path is not None
        assert args.train_mode == 'train_student'
        print('getting teacher embeddings dict...')
        if args.teacher_weights_path is not None:
            model.eval()
            # TODO: dict currently only support layer1-layer3!
            teacher_embeddings_dict = {
                'filenames':[],
                'embedding':[],
            } 
            with torch.no_grad():
                for i_batch, batch_dict in tqdm(enumerate(dataloaders['train_preloading'])):
                    for e in batch_dict:
                        if hasattr(batch_dict[e], 'to'):
                            batch_dict[e] = batch_dict[e].to(device)

                    output_dict = model(batch_dict)
                    torch.cuda.empty_cache()
                    teacher_embeddings_dict['filenames'].extend(batch_dict['filenames'])
                    teacher_embeddings_dict['embedding'].extend(output_dict['embedding'].detach().cpu())

            teacher_embeddings_dict['embedding'] = torch.stack(teacher_embeddings_dict['embedding'])
            teacher_embeddings_dict_new = {}
            for filename, embedding in zip(
                                        teacher_embeddings_dict['filenames'],
                                        teacher_embeddings_dict['embedding'],
                                        ):
                teacher_embeddings_dict_new[filename] = {
                    'embedding':embedding,
                }
            
            teacher_embeddings_dict = teacher_embeddings_dict_new
            del teacher_embeddings_dict_new
            if args.dataset == 'oxford':
                assert len(teacher_embeddings_dict) == 21711
            elif args.dataset == 'boreas':
                assert len(teacher_embeddings_dict) == 6006







    for epoch in range(args.epochs):
        t0 = time.time()
        txt = []



        print('training...')
        print(f'train_mode: {args.train_mode}')
        print(f'T: {args.model}')
        if args.train_mode == 'train_student':
            print(f'S: {args.student_model}')

        for i_batch, (batch_dict, positives_mask, negatives_mask, filenames) in tqdm(enumerate(dataloaders['train'])):

            n_positives = torch.sum(positives_mask).item()
            n_negatives = torch.sum(negatives_mask).item()
            if n_positives == 0 or n_negatives == 0:
                print('WARNING: Skipping batch without positive or negative examples')
                continue


            optimizer.zero_grad()
            optimizer_stu.zero_grad()






            # ---- force teacher to learn single stage
            if args.train_mode == 'train_teacher':
                assert args.train_mode == 'train_teacher'
                assert len(batch_dict)==1
                model_stu.train()
                _batch_dict = batch_dict[0]
                _batch_dict = {e:_batch_dict[e].to(device) for e in _batch_dict}
                output_dict = model(_batch_dict)
                loss, temp_stats, _ = loss_fn(output_dict, positives_mask, negatives_mask)
                loss.backward()
                optimizer.step()



            
            # ---- or loading teacher embeddings dict
            if teacher_embeddings_dict is None:
                output_dict = {'embedding':None}
            else:
                output_dict = {}
                output_dict['embedding'] = torch.stack([teacher_embeddings_dict[filename]['embedding'] for filename in filenames])
                output_dict['embedding'] = output_dict['embedding'].to(device)




            # ---- student learning single stage
            if args.train_mode == 'train_student':
                assert args.train_mode == 'train_student'
                assert len(batch_dict)==1
                model_stu.train()
                _batch_dict = batch_dict[0]
                _batch_dict = {e:_batch_dict[e].to(device) for e in _batch_dict}
                output_dict_stu = model_stu(_batch_dict)
                loss_stu = compute_all_loss(
                    output_dict_stu=output_dict_stu,
                    output_dict_tea=output_dict,
                    positives_mask=positives_mask,
                    negatives_mask=negatives_mask,
                    adaptor=None,
                    task_loss_fn_stu=loss_fn_stu,
                )
                loss_stu.backward()
                optimizer_stu.step()









            if isinstance(model, DistilImageFE):
                now_image_lr = optimizer.param_groups[0]["lr"]
                now_cloud_lr = 0
                now_image_lr_stu = optimizer_stu.param_groups[0]["lr"]
                now_cloud_lr_stu = 0
            elif isinstance(model, GeneralImageFE):
                now_image_lr = optimizer.param_groups[0]["lr"]
                now_cloud_lr = 0
                now_image_lr_stu = optimizer_stu.param_groups[0]["lr"]
                now_cloud_lr_stu = 0
            elif isinstance(model, MinkLocMultimodal):
                now_image_lr = optimizer.param_groups[0]["lr"]
                now_cloud_lr = optimizer.param_groups[1]["lr"]
                now_image_lr_stu = optimizer_stu.param_groups[0]["lr"]
                # now_cloud_lr_stu = optimizer_stu.param_groups[1]["lr"]

            else:
                raise Exception

                

            

            torch.cuda.empty_cache()  





        if scheduler is not None:
            scheduler.step()
            scheduler_stu.step()




        model.eval()
        model_stu.eval()
        


        # # --------------------------- test teacher ---------------------------
        if args.train_mode == 'train_teacher':
            print('testing teacher...')
            recall_top1p, recall_top1, ave_recall_for_each_scene_top1 = test_after_epoch(model, device)


            txt.append(f'recall_top1p                     {recall_top1p:.2f}')
            txt.append(f'recall_top1                      {recall_top1:.2f}')
            txt.append(f'ave_recall_for_each_scene_top1   {ave_recall_for_each_scene_top1.round(2)}')
            txt.append(f'Epoch {epoch}\t  Lr {now_image_lr}_{now_cloud_lr}\t  Time {(time.time()-t0):.2f}')
            txt.append(f'----------------------------------teacher---- {args.exp_name}') 

            with open('results.txt', 'a') as f:
                for each_line in txt:
                    f.writelines(each_line)
                    f.writelines('\n')
                    print(each_line)

            with open(f'results/{args.exp_name}.txt', 'a') as f:
                for each_line in txt:
                    f.writelines(each_line)
                    f.writelines('\n')







        # --------------------------- test student ---------------------------
        if args.train_mode == 'train_student':
            print('testing student...')
            txt = []
            recall_top1p_stu, recall_top1_stu, ave_recall_for_each_scene_top1_stu = test_after_epoch(model_stu, device)


            txt.append(f'recall_top1p                     {recall_top1p_stu:.2f}')
            txt.append(f'recall_top1                      {recall_top1_stu:.2f}')
            txt.append(f'ave_recall_for_each_scene_top1   {ave_recall_for_each_scene_top1_stu.round(2)}')
            txt.append(f'Epoch {epoch}\t  Lr {now_image_lr_stu}\t  Time {(time.time()-t0):.2f}')
            txt.append(f'----------------------------------student---- {args.exp_name}') 

            with open('results.txt', 'a') as f:
                for each_line in txt:
                    f.writelines(each_line)
                    f.writelines('\n')
                    print(each_line)

            with open(f'results/{args.exp_name}.txt', 'a') as f:
                for each_line in txt:
                    f.writelines(each_line)
                    f.writelines('\n')











if __name__ == '__main__':

    datetime_start = get_datetime()
    if not os.path.exists('results'):
        os.mkdir('results')

    with open('results.txt', 'w') as f:
        f.writelines('\n')

    with open(f'results/{args.exp_name}.txt', 'w') as f:
        f.writelines('\n')





    params = MinkLocParams(args.config, args.model_config)

    
    print(args.exp_name)

    dataloaders = make_dataloaders()
    


    do_train(dataloaders, params)



    datetime_end = get_datetime()
    with open('results.txt', 'a') as f:
        f.writelines('\n')
        f.writelines(datetime_start)
        f.writelines('\n')
        f.writelines(datetime_end)

    with open(f'results/{args.exp_name}.txt', 'a') as f:
        f.writelines('\n')
        f.writelines(datetime_start)
        f.writelines('\n')
        f.writelines(datetime_end)






