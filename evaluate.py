



import os


from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import torch
import tqdm

from tools.utils import MinkLocParams
from models.model_factory import model_factory

from datasets.dataloader_dataset import *



from models.minkloc import MinkLoc
from network.distil_imagefes import DistilImageFE

DEBUG = False








def evaluate(model, device, silent=True):
    assert len(args.eval_database_files) == len(args.eval_query_files)





    lidar2image_ndx = pickle.load(open(args.lidar2image_ndx_path, 'rb'))



    stats = {}

    for database_file, query_file in zip(args.eval_database_files, args.eval_query_files):


        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)


        p = os.path.join(args.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        if args.dataset in ['oxford','oxfordadafusion']:
            p = os.path.join(args.dataset_folder, query_file)
            with open(p, 'rb') as f:
                query_sets = pickle.load(f)


        elif args.dataset == 'boreas':
            p = os.path.join(args.dataset_folder, query_file)
            with open(p, 'rb') as f:
                query_sets = pickle.load(f)
        
        else:
            raise Exception


        temp = evaluate_dataset(model, device, database_sets, query_sets, silent=silent, lidar2image_ndx=lidar2image_ndx)
        stats[location_name] = temp

    return stats









def evaluate_dataset(model, device, database_sets, query_sets, silent=True, lidar2image_ndx=None):



    if args.dataset == 'oxfordadafusion':
        recall = np.zeros(25)
    else:
        recall = np.zeros(25)
        
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()







    # -- new
    if args.dataset == 'boreas':
        database_embeddings = get_latent_vectors_with_merged(model=model, 
                                                             sets_unmerged=database_sets, 
                                                             device=device, 
                                                             lidar2image_ndx=lidar2image_ndx)
        query_embeddings = database_embeddings.copy()
        print('boreas direct copy')
    else:
        database_embeddings = get_latent_vectors_with_merged(model=model, 
                                                             sets_unmerged=database_sets, 
                                                             device=device, 
                                                             lidar2image_ndx=lidar2image_ndx)
        
        query_embeddings = get_latent_vectors_with_merged(model=model, 
                                                          sets_unmerged=query_sets, 
                                                          device=device, 
                                                          lidar2image_ndx=lidar2image_ndx)





    recall_for_each_scene = np.zeros([len(query_sets), 25])

    # i is database
    # j is query
    for i in tqdm.tqdm(range(len(query_sets)), disable=silent):
        for j in range(len(query_sets)):
            if i == j:
                continue
            # pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
            #                                                     database_sets)
            pair_recall, pair_similarity, pair_opr = get_recall_torch(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets)
            
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            # for x in pair_similarity:
            #     similarity.append(x)

            recall_for_each_scene[i] += np.array(pair_recall)



    # count 23*22=506
    ave_recall = recall / count
    # average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)

    ave_recall_for_each_scene = recall_for_each_scene / (len(query_sets) - 1)





    stats = {'ave_one_percent_recall': ave_one_percent_recall, 
             'ave_recall': ave_recall,
             'ave_recall_for_each_scene':ave_recall_for_each_scene,
             }
    


    return stats






def get_latent_vectors_with_merged(model, sets_unmerged, device, lidar2image_ndx):
        # Adapted from original PointNetVLAD code
    sets_merged = []
    scene_sets_lengths = []
    for scene_set in sets_unmerged:
        scene_sets_lengths.append(len(scene_set))
        for eachitem in scene_set.values():
            sets_merged.append(eachitem)




    if DEBUG:
        embeddings = np.random.rand(len(sets_merged), 256)
        return embeddings


    dataset = DataloaderDataset(sets_merged, device, lidar2image_ndx)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=args.val_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn,
                                 )


    model.eval()
    embeddings_l = []


    for i_batch, batch_dict in tqdm.tqdm(enumerate(dataloader)):

        batch_dict = {e: batch_dict[e].to(device) for e in batch_dict}

        with torch.no_grad():
            output = model(batch_dict)

        embedding = output['embedding']

        # embedding is (1, 256) tensor
        if args.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings


        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)


    embeddings = np.vstack(embeddings_l)
    assert len(embeddings)==len(dataset)



    current_id = 0
    embeddings_unmerged = []
    for length in scene_sets_lengths:
        embeddings_this_scene = embeddings[current_id:current_id+length]
        embeddings_unmerged.append(embeddings_this_scene)
        current_id += length


    assert len(sets_merged) == sum(scene_sets_lengths)
    assert len(embeddings_unmerged) == len(sets_unmerged)
    


    return embeddings_unmerged








def get_latent_vectors(model, set, device, params, lidar2image_ndx):
        # Adapted from original PointNetVLAD code

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings


    dataset = DataloaderDataset(set_dict=set, 
                                device=device, 
                                lidar2image_ndx=lidar2image_ndx)
    
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=args.val_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn
                                 )

    model.eval()
    embeddings_l = []


    for i_batch, batch_dict in enumerate(dataloader):
        None

        batch_dict['coords'] = batch_dict['coords'] .to(device)
        batch_dict['features'] = batch_dict['features'] .to(device)
        batch_dict['images'] = batch_dict['images'] .to(device)
        batch_dict['clouds'] = batch_dict['clouds'] .to(device)


        with torch.no_grad():
            output = model(batch_dict)

        embedding = output['embedding']

        # embedding is (1, 256) tensor
        if args.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings


        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)


    embeddings = np.vstack(embeddings_l)
    assert len(embeddings)==len(dataset)



    return embeddings








def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    if args.dataset == 'oxfordadafusion':
        num_neighbors = 25
    else:
        num_neighbors = 25

    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)


    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        # mat = queries_output[i].unsqueeze(0) @ database_output.T
        # distances, indices = torch.topk(mat, k=num_neighbors, dim=1, largest=True, sorted=True)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    # similarity = torch.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall









def get_recall_torch(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]



    if args.dataset == 'oxfordadafusion':
        num_neighbors = 25
    else:
        num_neighbors = 25



    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    # 1 - kdtree
    # database_nbrs = KDTree(database_output)
    # # 2 - torch
    database_output = torch.from_numpy(database_output)
    queries_output = torch.from_numpy(queries_output)


    database_output = torch.nn.functional.normalize(database_output, p=2, dim=1)  # Normalize embeddings
    queries_output = torch.nn.functional.normalize(queries_output, p=2, dim=1)  # Normalize embeddings
    

    mat = queries_output @ database_output.T
    distances_full, indices_full = torch.topk(mat, k=num_neighbors, dim=1, largest=True, sorted=True)


    # database_output = database_output.cpu().numpy()
    # queries_output = queries_output.cpu().numpy()



    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)


    # return 1, 1, 1


    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        distances = distances_full[i].cpu().numpy()
        indices = indices_full[i].cpu().numpy()

        for j in range(len(indices)):
            if indices[j] in true_neighbors:
                recall[j] += 1
                break


        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1



    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100

    return recall, None, one_percent_recall









if __name__ == "__main__":




    from tools.options import Options
    args = Options().parse()



    epoch = 0
    for epoch in range(1):
        weights_path = os.path.join(args.models_dir, f'recall_top1_best_ep{epoch}_67.58.pth')
        weights_path = 'teacher_weights/boreas__T:minklocmmcat__resnet18__img256__pc128__32_64_64__1_1_1__1__allstgF__b128__trainteacher/models/r1_best_ep48_93.05.pth'





        print('Config path: {}'.format(args.config))
        print('Model config path: {}'.format(args.model_config))
        if weights_path is None:
            w = 'RANDOM WEIGHTS'
        else:
            w = weights_path
        print('Weights: {}'.format(w))
        print('')



        params = MinkLocParams(args.config, args.model_config)
        # params.print()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print('Device: {}'.format(device))



        if args.model == 'None':
            model = None

        elif args.model == 'imageonly':
            model = DistilImageFE(image_fe=args.image_fe,
                                input_type='image', # 'image' or 'sph_cloud'
                                num_other_stage_blocks=args.num_other_stage_blocks,
                                num_stage3_blocks=args.num_stage3_blocks,
                                pool_method=args.imageonly_pool_method)

        elif args.model == 'cloudonly':
            model = MinkLoc(in_channels=1, feature_size=128, output_dim=128,
                            planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                            conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')

        elif args.model == 'minklocmmcat':
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





        if weights_path is not None:
            assert os.path.exists(weights_path), 'Weights do not exist: {}'.format(weights_path)
            print('Loading weights: {}'.format(weights_path))
            model.load_state_dict(torch.load(weights_path, map_location=device))




        model.to(device)

        test_stats = evaluate(model, device, silent=False)


        test_stats = test_stats[args.dataset]
        recall_top1p = test_stats['ave_one_percent_recall']
        recall_topN = test_stats['ave_recall']
        recall_top1 = recall_topN[0]

        ave_recall_for_each_scene = test_stats['ave_recall_for_each_scene']
        ave_recall_for_each_scene_top1 = ave_recall_for_each_scene[:,0]

        
        print(f'recall_top1p                       {recall_top1p:.2f}')
        print(f'recall_top1                        {recall_top1:.2f}')
        print(f'ave_recall_for_each_scene_top1     {ave_recall_for_each_scene_top1.round(2)}')