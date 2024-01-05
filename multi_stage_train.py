




import torch








# def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
def multistaged_training_step_distil(batch, positives_mask, negatives_mask, model, phase, device, optimizer, loss_fn_stu,
                                     compute_all_loss, output_dict_tea):
    """
    multi-stage training step for distillation
    """
    assert phase in ['train', 'val']
    # batch: {{'coords':, 'features':}*16}
    # batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []

    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['embedding'])

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)


    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        
        _embeddings_dict = {
            'embedding': embeddings,
            }
        


        # -- distil loss_fn
        loss = compute_all_loss(
            output_dict_stu=_embeddings_dict,
            output_dict_tea=output_dict_tea,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
            adaptor=None,
            task_loss_fn_stu=loss_fn_stu,
        )



        # stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad



    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                embeddings = y['embedding']

                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])

                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors













# def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
def multistaged_training_step(batch, positives_mask, negatives_mask, model, phase, device, optimizer, loss_fn):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    # batch: {{'coords':, 'features':}*16}
    # batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []

    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['embedding'])

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)


    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        
        _embeddings_dict = {
            'embedding': embeddings,
            }
        

        # -- vanilla loss_fn
        loss, stats, _ = loss_fn(_embeddings_dict, positives_mask, negatives_mask)


        # stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad


    # # Delete intermediary values
    # embeddings_l, embeddings, y, loss = None, None, None, None

    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                embeddings = y['embedding']

                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])

                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors













# def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
def multistaged_training_step_multimodal(batch, positives_mask, negatives_mask, model, phase, device, optimizer, loss_fn,
                                         train_step_type):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    # batch: {{'coords':, 'features':}*16}
    # batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    embeddings_cloud_l = []
    embeddings_image_l = []

    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['embedding'])
            embeddings_cloud_l.append(y['cloud_embedding'])
            embeddings_image_l.append(y['image_embedding'])

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)
    embeddings_cloud = torch.cat(embeddings_cloud_l, dim=0)
    embeddings_image = torch.cat(embeddings_image_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
            embeddings_cloud.requires_grad_(True)
            embeddings_image.requires_grad_(True)
        
        _embeddings_dict = {
            'embedding': embeddings,
            'cloud_embedding': embeddings_cloud,
            'image_embedding': embeddings_image
            }
        
        loss, stats, _ = loss_fn(_embeddings_dict, positives_mask, negatives_mask)
        # stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad
            embeddings_cloud_grad = embeddings_cloud.grad
            embeddings_image_grad = embeddings_image.grad

    # # Delete intermediary values
    # embeddings_l, embeddings, y, loss = None, None, None, None

    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                embeddings = y['embedding']
                embeddings_cloud = y['cloud_embedding']
                embeddings_image = y['image_embedding']

                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                if train_step_type == 'multi_sep':
                    # ---- separate backward ----
                    embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                    # embeddings_cloud.backward(gradient=embeddings_cloud_grad[i: i+minibatch_size])
                    # embeddings_image.backward(gradient=embeddings_image_grad[i: i+minibatch_size])
                elif train_step_type == 'multi_joint':
                    # ---- joint backward ----
                    embeddings_tobackward = torch.cat([
                        embeddings, 
                        embeddings_cloud,
                        # embeddings_image,
                        ], dim=-1)
                    embeddings_tobackward_grad = torch.cat([
                        embeddings_grad[i: i+minibatch_size], 
                        embeddings_cloud_grad[i: i+minibatch_size], 
                        # embeddings_image_grad[i: i+minibatch_size]
                        ], dim=-1)
                    embeddings_tobackward.backward(gradient=embeddings_tobackward_grad)
                else:
                    raise NotImplementedError



                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

