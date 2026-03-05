import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import h5py

from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
import faiss

import kitti_dataset
import oord_dataset
import matplotlib.pyplot as plt
from loss import TripletLoss, InfoNCE

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the visible GPUs

def get_args():
    parser = argparse.ArgumentParser(description='BEVPlace++')
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    parser.add_argument('--path', type=str, default='../../oord_data/', help='dataset path')
    parser.add_argument('--batchSize', type=int, default=8,  
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    # parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
    parser.add_argument('--cacheBatchSize', type=int, default=8, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=1e-3, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--loss', type=str, default='infonce', choices=['triplet','infonce'])
    parser.add_argument('--threads', type=int, default=16, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')


    parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')
    parser.add_argument('--network', type=str, default='rein', choices=['rein', 'e18rein','e34rein','erein', 'rerein'])

    # parser.add_argument('--load_from', type=str, default='runs/Aug08_10-17-29', help='Path to load checkpoint from, for resuming training or testing.')# original model 
    # parser.add_argument('--load_from', type=str, default='runs/Jan07_17-57-05', help='Path to load checkpoint from, for resuming training or testing.')#infonce, pos-neg = 24-26, temp 0.1 
    parser.add_argument('--load_from', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')#infonce, pos-neg = 24-26, temp 0.1 


    parser.add_argument('--ckpt', type=str, default='best', 
            help='Load_from from latest or best checkpoint.', choices=['latest', 'best'])
    

    opt = parser.parse_args()
    return opt


def validate_epoch(model, dataset):
    model.eval()
    val_loader = DataLoader(dataset=dataset, num_workers=opt.threads, 
                            batch_size=opt.batchSize, shuffle=False, 
                            collate_fn=kitti_dataset.collate_fn)
    
    criterion = TripletLoss().to(device)
    val_loss = 0
    n_batches = (len(dataset) + opt.batchSize - 1) // opt.batchSize

    with torch.no_grad():
        for iteration, (query, positives, negatives, indices) in enumerate(val_loader, 1):
            B = query.shape[0]
            input = torch.cat([query, positives, negatives])
            input = input.to(device)

            _, _, global_descs = model(input)
            global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])

            loss = 0
            num_negs = negatives.shape[0] // B
            for i in range(len(global_descs_Q)):
                max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
                loss += max_loss

            val_loss += (loss / opt.batchSize).item()

    avg_val_loss = val_loss / n_batches
    return avg_val_loss

def validate_epoch_infonce(model, dataset):
    model.eval()
    val_loader = DataLoader(dataset=dataset, num_workers=opt.threads, 
                            batch_size=opt.batchSize, shuffle=False, 
                            collate_fn=kitti_dataset.collate_fn)
    
    criterion = InfoNCE(negative_mode='paired').to(device)
    val_loss = 0
    n_batches = (len(dataset) + opt.batchSize - 1) // opt.batchSize

    with torch.no_grad():
        for iteration, (query, positives, negatives, indices) in enumerate(val_loader, 1):
            B = query.shape[0]
            input = torch.cat([query, positives, negatives])
            input = input.to(device)

            _, _, global_descs = model(input)
            global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])

            loss = 0
            num_negs = negatives.shape[0] // B
            global_descs_N = global_descs_N.view(B, num_negs, -1)

            # Compute InfoNCE loss
            loss = criterion(global_descs_Q, global_descs_P, global_descs_N)

            val_loss += loss.item() # InfoNCE returns mean by default

    avg_val_loss = val_loss / n_batches
    return avg_val_loss

def train_epoch(epoch, model, train_set):
    epoch_loss = 0
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize
    
    criterion = TripletLoss().to(device)
    
    model.eval()

    if epoch >= 0:
        print('====> Building Cache for Hard Mining')
        # Disable mining for all subsets initially
        for s in train_set.datasets:
            s.mining = False

        # Sequential loader (shuffle=False) to map indices correctly
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
            batch_size=opt.batchSize, shuffle=False, 
            collate_fn=kitti_dataset.collate_fn)
        
        # Temporary storage for all features
        all_features = np.zeros((len(train_set), model.global_feat_dim), dtype=np.float32)

        with torch.no_grad():
            for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
                query = query.to(device)
                _, _, global_descs = model(query)
                all_features[indices, :] = global_descs.detach().cpu().numpy()

        # Distribute features back to each subset
        start_idx = 0
        for s in train_set.datasets:
            end_idx = start_idx + len(s)
            s.h5feat = all_features[start_idx:end_idx] # Give subset its specific features
            s.mining = True
            start_idx = end_idx

    # ACTUAL TRAINING: Now use shuffle=True to mix sequences
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True, 
                collate_fn=kitti_dataset.collate_fn)
    
    model.train()
    # ... rest of the training loop remains the same ...

    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):

        B, C, H, W = query.shape
        input = torch.cat([query, positives, negatives])

        input = input.to(device)
        
        _, _, global_descs = model(input)

        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])
        

        optimizer.zero_grad()

        # no need to train the kps feature
        loss = 0
        num_negs = negatives.shape[0]//B
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
            loss += max_loss
        
        loss /= opt.batchSize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                n_batches, batch_loss), flush = True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * n_batches) + iteration)
            

    optimizer.zero_grad()    
    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)


def train_epoch_infonce(epoch, model, train_set, optimizer, scheduler):
    epoch_loss = 0
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize
    
    criterion = InfoNCE(negative_mode='paired').to(device)
    
    model.eval()

    if epoch >= 0:
        print('====> Building Cache for Hard Mining')
        # Disable mining for all subsets initially
        for s in train_set.datasets:
            s.mining = False

        # Sequential loader (shuffle=False) to map indices correctly
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
            batch_size=opt.batchSize, shuffle=False, 
            collate_fn=kitti_dataset.collate_fn)
        
        # Temporary storage for all features
        all_features = np.zeros((len(train_set), model.global_feat_dim), dtype=np.float32)

        with torch.no_grad():
            for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
                query = query.to(device)
                _, _, global_descs = model(query)
                all_features[indices, :] = global_descs.detach().cpu().numpy()

        # Distribute features back to each subset
        start_idx = 0
        for s in train_set.datasets:
            end_idx = start_idx + len(s)
            s.h5feat = all_features[start_idx:end_idx] # Give subset its specific features
            s.mining = True
            start_idx = end_idx

    # ACTUAL TRAINING: Now use shuffle=True to mix sequences
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True, 
                collate_fn=kitti_dataset.collate_fn)
    
    model.train()
    # ... rest of the training loop remains the same ...

    model.train()
    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
        B, C, H, W = query.shape
        input = torch.cat([query, positives, negatives]).to(device)
        
        _, _, global_descs = model(input)
        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])

        optimizer.zero_grad()

        num_negs = negatives.shape[0] // B
        global_descs_N = global_descs_N.view(B, num_negs, -1)
        
        loss = criterion(global_descs_Q, global_descs_P, global_descs_N)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        
        if iteration % 50 == 0:
            # Print current LR to monitor warmup/decay
            current_lr = optimizer.param_groups[0]['lr']
            print(f"==> Epoch[{epoch}]({iteration}/{n_batches}): Loss: {batch_loss:.4f} | LR: {current_lr:.8f}")
            

    # optimizer.zero_grad()   
    scheduler.step()
    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
# def infer(eval_set, return_local_feats = False):
#     test_data_loader = DataLoader(dataset=eval_set, 
#                 num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)

#     model.eval()
#     model.to('cuda')
#     with torch.no_grad():
        
#         all_global_descs = []
#         all_local_feats = []
#         for _, (imgs, _) in enumerate(tqdm(test_data_loader)):
#             imgs = imgs.to(device)
#             _ , local_feat, global_desc = model(imgs)
#             all_global_descs.append(global_desc.detach().cpu().numpy())
#             if return_local_feats:
#                 all_local_feats.append(local_feat.detach().cpu().numpy())
           
#     if return_local_feats:
#         return np.concatenate(all_local_feats, axis=0), np.concatenate(all_global_descs, axis=0)
#     else:
#         return np.concatenate(all_global_descs, axis=0)

def infer(eval_set, return_local_feats=False, cache_path='local_feats_cache.h5', permute_local=None):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)

    model.eval()
    model.to('cuda')
    
    all_global_descs = []
    
    if return_local_feats:
        f = h5py.File(cache_path, 'w')
        dset_local = None

    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(test_data_loader)):
            imgs = imgs.to('cuda')
            
            _ , local_feat, global_desc = model(imgs)
            
            # 1. Global Features (Keep in RAM)
            all_global_descs.append(global_desc.detach().cpu().numpy())
            
            # 2. Local Features (Save to Disk)
            if return_local_feats:
                # Move to CPU
                l_batch = local_feat.detach().cpu().numpy()
                
                # --- FIX: Transpose BEFORE saving ---
                # Example: If PyTorch is (N, C, H, W) and you want (N, H, W, C)
                # You would pass permute_local=(0, 2, 3, 1)
                if permute_local is not None:
                    l_batch = l_batch.transpose(permute_local)
                
                # Initialize dataset on first batch
                if dset_local is None:
                    dset_local = f.create_dataset("local_feats", data=l_batch, 
                                                  maxshape=(None, *l_batch.shape[1:]))
                else:
                    dset_local.resize((dset_local.shape[0] + l_batch.shape[0]), axis=0)
                    dset_local[-l_batch.shape[0]:] = l_batch

    global_feats_final = np.concatenate(all_global_descs, axis=0)

    if return_local_feats:
        return dset_local, global_feats_final
    else:
        return global_feats_final
    
def testPCA(eval_set, epoch=0, write_tboard=False):
    # TODO global descriptor PCA for faster inference speed
    pass
    # return recalls

def getClusters(cluster_set, num_feat=128, num_cluster = 64):
    n_descriptors = 10000
    n_per_image = 25
    n_im = ceil(n_descriptors/n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            all_feats = h5.create_dataset("descriptors", 
                        [n_descriptors, num_feat], 
                        dtype=np.float32)

            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                local_feat, _, _ = model(query)
                local_feat = local_feat.view(query.size(0), num_feat, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*opt.cacheBatchSize*n_per_image
                for ix in range(local_feat.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix*n_per_image
                    all_feats[startix:startix+n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(n_im/opt.cacheBatchSize)), flush=True)
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(num_feat,num_cluster, niter=niter, verbose=False)
        kmeans.train(all_feats[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path+'/'+filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_out_path+'/'+'model_best.pth.tar')


if __name__ == "__main__":
    opt = get_args()

    device = torch.device("cuda")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)    

    if opt.network == 'rein':
        from model.REIN import REIN
    elif opt.network == 'erein':
        from model.EREIN import REIN
    elif opt.network == 'e18rein':
        from model.E18REIN import REIN
    elif opt.network == 'e34rein':
        from model.E34REIN import REIN
    elif opt.network == 'rerein':
        from model.REREIN import REIN
    else:
        raise ValueError(f"Unknown network: {opt.network}")

    model = REIN()
    model = model.cuda()
    
    # initialize netvlad with pre-trained or cluster
    if opt.load_from:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.load_from,  'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.load_from, 'model_best.pt')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            # if opt.network == 'rerein':
            #     # model.load_state_dict(checkpoint['state_dict'])
            #     new_state_dict = {}
            #     state_dict = checkpoint
            #     for k, v in state_dict.items():
            #         # Example: change 'model.model.conv1' to 'backbone.model.conv1'
            #         name = k.replace('model.model', 'backbone.model') 
            #         new_state_dict[name] = v

            #     # 4. Load the fixed dictionary
            #     model.load_state_dict(new_state_dict, strict=False)
            #     # model.load_state_dict(checkpoint)
            #     model = model.to(device)
            #     print("=> loaded checkpoint '{}'"
            #         .format(resume_ckpt))
            if opt.network == 'rerein':
                print("=> Specialized loading for REREIN backbone...")
                new_state_dict = {}
                state_dict = checkpoint
                for k, v in state_dict.items():
                    # Map backbone weights: e.g., 'model.conv1' -> 'backbone.model.conv1'
                    # Adjust the string replacement based on your specific saved key structure
                    name = k.replace('model.model', 'backbone.model') #if k.startswith('model.') else f'backbone.{k}'
                    new_state_dict[name] = v

                # Load backbone weights (strict=False because NetVLAD is missing from checkpoint)
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"=> Missing keys (expected): {msg.missing_keys}")
                
                # --- INITIALIZE NETVLAD SINCE IT WAS NOT IN CHECKPOINT ---
                initcache = join(opt.cachePath, 'desc_cen.hdf5')
                if not isfile(initcache):
                    print('===> Checkpoint only had backbone. Calculating clusters for NetVLAD...')
                    # Use a training subset to build clusters
                    cluster_set = oord_dataset.TrainingDataset(dataset_path=opt.path)
                    getClusters(cluster_set, num_feat=128, num_cluster=64)
                
                with h5py.File(initcache, mode='r') as h5:
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    model.pooling.init_params(clsts, traindescs)
                print("=> NetVLAD initialized from clusters.")
            else: 
                model.load_state_dict(checkpoint['state_dict'])
                model = model.to(device)
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    else:
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = oord_dataset.TrainingDataset()
            print('===> Calculating descriptors and clusters')
            getClusters(train_set)
        with h5py.File(initcache, mode='r') as h5: 
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pooling.init_params(clsts, traindescs) 
            model = model.cuda()

    if opt.mode.lower() == 'train':
        # preparing tensorboard
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))

        logdir = writer.file_writer.get_logdir()
        try:
            makedirs(logdir)
        except:
            pass

        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)


        print('===> Loading dataset(s)')
        train_sequences = ['Twolochs_2', 'Maree_1', 'Bellmouth_2', 'Hydro_1']
        train_subsets = []

        for seq in train_sequences:
            print(f'Loading training sequence: {seq}')
            train_subsets.append(oord_dataset.TrainingDataset(dataset_path=opt.path, seq=seq))

        train_set = ConcatDataset(train_subsets)
        # initilize model weights
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
            model.parameters()), lr=opt.lr, weight_decay=opt.weightDecay)    
        def lr_lambda(current_epoch):
            # 5 Epoch Linear Warmup
            if current_epoch < 5:
                return float(current_epoch + 1) / 5
            # Step Decay: Multiply by lrGamma every lrStep
            return opt.lrGamma ** (current_epoch // opt.lrStep)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        best_score = 0

        # for epoch in range(opt.nEpochs):
        #     avg_loss = train_epoch(epoch, model, train_set)
        #     print('===> Testing')

        #     # Evaluate on each validation sequence
        #     recalls_oord = []
        #     val_losses = []

        #     eval_seq =  [('Bellmouth_2', 'Bellmouth_1'),
        #     ('Twolochs_2', 'Twolochs_1'),
        #     ('Hydro_1','Hydro_2'),
        #     ('Hydro_1','Hydro_3'),
        #     ('Maree_1','Maree_2')]
        #     eval_datasets = []
        #     eval_global_descs = []

        #     for seq in eval_seq:
        #         test_set = oord_dataset.InferDataset(seq=seq,dataset_path=opt.path)
        #         val_set = oord_dataset.TrainingDataset(seq=seq)
        #         # Infer for recall
        #         global_descs = infer(test_set)
        #         eval_global_descs.append(global_descs)
        #         eval_datasets.append(test_set)

        #         # ALSO calculate validation loss
        #         val_loss = validate_epoch(model, val_set)
        #         val_losses.append(val_loss)

        #     recalls_oord, _ = oord_dataset.evaluateResults(eval_global_descs, eval_datasets)

        #     for ii in range(len(recalls_oord)):
        #         writer.add_scalars('val/recall', {'OORD_' + eval_seq[ii+1]: recalls_oord[ii]}, epoch)
        #         writer.add_scalars('val/loss', {'OORD_' + eval_seq[ii+1]: val_losses[ii]}, epoch)

        #     mean_recall = np.mean(recalls_oord)
        #     mean_val_loss = np.mean(val_losses)
            
        #     print('===> Mean Recall on OORD : %0.2f'%(mean_recall*100))
        #     print('===> Mean Validation Loss: %.6f' % mean_val_loss)

        #     is_best = mean_recall > best_score 
        #     if is_best: best_score = mean_recall

        #     saveCheckpoint({
        #             'epoch': epoch,
        #             'state_dict': model.state_dict(),
        #             'recalls': mean_recall,
        #             'best_score': best_score,
        #             'optimizer' : optimizer.state_dict(),
        #     }, is_best, logdir)

        #     with open(join(logdir, 'epoch_losses.txt'), 'a') as f:
        #         f.write(f"Epoch {epoch}: Val Loss = {mean_val_loss:.6f}, Mean Recall = {mean_recall:.4f}\n")
        for epoch in range(opt.nEpochs):
            if opt.loss == 'triplet':
                avg_loss = train_epoch(epoch, model, train_set)
            if opt.loss == 'infonce':
                avg_loss = train_epoch_infonce(epoch,model,train_set,optimizer,scheduler)
            print('===> Testing / Validation')
            model.eval()
            
            # Define the specific pairs for validation as requested
            eval_pairs = [
                ('Bellmouth_2', 'Bellmouth_1'),
                ('Twolochs_2', 'Twolochs_1'),
                ('Hydro_1', 'Hydro_2'),
                ('Hydro_1', 'Hydro_3'),
                ('Maree_1', 'Maree_2')
            ]
            
            pair_recalls = []
            
            for db_name, q_name in eval_pairs:
                print(f"Validating pair: DB={db_name}, Query={q_name}")
                
                # Load the two datasets for the pair
                db_set = oord_dataset.InferDataset(seq=db_name, dataset_path=opt.path)
                query_set = oord_dataset.InferDataset(seq=q_name, dataset_path=opt.path)
                
                # Extract global descriptors for both
                # Note: 'infer' returns a numpy array of global descriptors
                db_global_descs = infer(db_set)
                query_global_descs = infer(query_set)
                
                # evaluateResults expects a list of global descriptors and a list of datasets
                # index 0 is used as the Database in oord_dataset.evaluateResults
                res_recalls, _ = oord_dataset.evaluateResults(
                    [db_global_descs, query_global_descs], 
                    [db_set, query_set]
                )
                
                recall_val = res_recalls[0] # Recall for this specific pair
                pair_recalls.append(recall_val)
                
                # Log individual pair performance to Tensorboard
                writer.add_scalar(f'val_recall/{db_name}_{q_name}', recall_val, epoch)

            # Calculate the average recall across all pairs
            mean_recall = np.mean(pair_recalls)
            print(f'===> Mean Recall on Validation Pairs: {mean_recall*100:.2f}%')
            writer.add_scalar('val/mean_recall', mean_recall, epoch)

            # Check if this is the best model based on average pair recall
            is_best = mean_recall > best_score 
            if is_best: 
                best_score = mean_recall

            # Save checkpoint
            saveCheckpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': mean_recall,
                    'best_score': best_score,
                    'optimizer' : optimizer.state_dict(),
            }, is_best, logdir)

            with open(join(logdir, 'epoch_losses.txt'), 'a') as f:
                f.write(f"Epoch {epoch}: Mean Pair Recall = {mean_recall:.4f}\n")
        print('===> Best Recall: %0.2f'%(mean_recall*100))
        writer.close()

    elif opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        print('====> Extracting Features of OORD and calculating recalls')
        # eval_seq = [('Maree_1','Maree_2')]
        # eval_seq = [('Bellmouth_2', 'Bellmouth_1')]
        # eval_seq = [('Hydro_1','Hydro_2') ]#, ('Hydro_2', 'Hydro_1')]
        # eval_seq =  [('Bellmouth_2', 'Bellmouth_1'),
        #             # ('Bellmouth_2','Bellmouth_3'),
        #             # ('Bellmouth_2','Bellmouth_4'),
        #             ('Twolochs_2', 'Twolochs_1'),
        #             ('Hydro_1','Hydro_2'),
        #             ('Hydro_1','Hydro_3'),
        #             ('Maree_1','Maree_2')]
        eval_seq = [('Twolochs_2', 'Twolochs_1')]
        for sub_seq in eval_seq:
            print(f"Processing {sub_seq}")
            eval_datasets = []
            eval_global_descs = []
            eval_local_feats = []
            result = [] 
            for seq in sub_seq:   
                test_set = oord_dataset.InferDataset(seq=seq,dataset_path=opt.path, sample_inteval=5)   
                test_set.printpath()
                # local_feats, global_descs = infer(test_set,return_local_feats=True, cache_path= seq + '.h5', permute_local=(0,2,3,1))
                global_descs = infer(test_set,return_local_feats=False)
                eval_global_descs.append(global_descs)
                eval_datasets.append(test_set)
                # eval_local_feats.append(local_feats)

            out_name = sub_seq[0] + '_' + sub_seq[1] + '_4paper/'
            #recalls_oord,success_rate,mean_trans_err,mean_rot_err,result = oord_dataset.evaluateResults(eval_global_descs, eval_datasets, eval_local_feats, out_name)
            recalls_oord, result = oord_dataset.evaluateResults(eval_global_descs, eval_datasets)
            
            print('\n################# Recall avg on OORD ########################\n')
            mean_recall = np.mean(recalls_oord)
            # Calculate mean FN, FP, TN
            num_datasets = len(result)


            mean_tp = sum(entry['TP'] for entry in result) / num_datasets
            mean_fn = sum(entry["FN"] for entry in result) / num_datasets
            mean_fp = sum(entry["FP"] for entry in result) / num_datasets
            mean_tn = sum(entry["TN"] for entry in result) / num_datasets
            mean_ap = sum(entry["AP"] for entry in result) /num_datasets

            recall = 100*mean_tp / (mean_tp + mean_fn) if (mean_tp + mean_fn) > 0 else 0
            precision = 100* mean_tp / (mean_tp + mean_fp) if (mean_tp + mean_fp) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    

            print(f"Mean TP: {mean_tp}")
            print(f"Mean FN: {mean_fn}")
            print(f"Mean FP: {mean_fp}")
            print(f"Mean TN: {mean_tn}")
            print(f"AP: {mean_ap}")
            print(f"Recall@1: {recall:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"F1 Score: {f1_score:.2f}")
            #print(f"Pose success rate: {success_rate:.2f}")
            #print(f"Mean translational error: {mean_trans_err:.2f}")
            #print(f"Mean rotation error: {mean_rot_err:.2f}")

