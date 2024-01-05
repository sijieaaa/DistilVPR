


import torch
import numpy as np
from datasets.oxford import OxfordDataset
import MinkowskiEngine as ME


from tools.utils_adafusion import pc_array_to_voxel


from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)







def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e





def make_collate_fn_bak(dataset: OxfordDataset):

    # set_transform: the transform to be applied to all batch elements
    def collate_fn_bak(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors

        filenames = [e['filename'] for e in data_list]

        result = {
            'positives_mask': positives_mask, 
            'negatives_mask': negatives_mask,
            'filenames': filenames
            }
        
        if 'clouds' in data_list[0]:

            coords = [e['coords'] for e in data_list]
            clouds = [e['clouds'] for e in data_list]

            coords = ME.utils.batched_coordinates(coords)
            clouds = torch.cat(clouds, dim=0)
            assert coords.shape[0]==clouds.shape[0]
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)

            result['coords'] = coords
            result['clouds'] = clouds
            result['features'] = feats

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0)       # Produces (N, C, H, W) tensor

        return result
    

    return collate_fn_bak








def make_collate_fn(dataset: OxfordDataset):




    def collate_fn(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors

        filenames = [e['filename'] for e in data_list]

        # result = {
        #     'positives_mask': positives_mask, 
        #     'negatives_mask': negatives_mask,
        #     'filenames': filenames
        #     }
        

        if 'clouds' in data_list[0]:
            coords = [e['coords'] for e in data_list]
            clouds = [e['clouds'] for e in data_list]

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]

        # if 'voxels' in data_list[0]:
        #     voxels = [e['voxels'] for e in data_list]

        

        big_batch = []
        batch_split_size = args.train_batch_split_size
        for i in range(0, len(data_list), batch_split_size):
            temp = coords[i:i + batch_split_size]
            imgs = images[i:i + batch_split_size]
            imgs = torch.stack(imgs, dim=0)
            c = ME.utils.batched_coordinates(temp)
            f = torch.ones((c.shape[0], 1), dtype=torch.float32)
            # v = voxels[i:i + batch_split_size]
            # v = torch.stack(v, dim=0)
            minibatch = {
                'coords': c, 
                'features': f,
                'images': imgs,
                # 'voxels': v
                }
            big_batch.append(minibatch)




        return big_batch, positives_mask, negatives_mask, filenames
    


    return collate_fn

