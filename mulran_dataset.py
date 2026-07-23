import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data
from tqdm import tqdm 

import h5py

import faiss
from RANSAC import rigidRansac

mulran_seq_split_points = {"DCC01":765, "KAIST02":1500, "KAIST03":1420, "Riverside03":1740, 'DCC02': 1500, 'Riverside01': 1120}

class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path = '/media/manh/manh/mulran/',sample_inteval=1):
        super().__init__()

        self.sample_inteval = sample_inteval
        # bev path
        imgs_p = os.listdir(dataset_path+'/cartesian/' + seq + '_resize/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+'/cartesian/'+seq+'_resize/'+imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+'/'+seq+'/pose.txt')[::sample_inteval]
        self.db_split_index = mulran_seq_split_points[seq]//sample_inteval 
        cen2018path = os.path.join(dataset_path, 'cen2018', seq + '_resize/') 
        cen2018 = os.listdir(cen2018path)
        self.cen2018 = [cen2018path + cen2018[i] for i in range(0, len(cen2018), sample_inteval)]

    def getkeypoint(self, index):
        # --- Load the image in grayscale ---
        feature_img = cv2.imread(self.cen2018[index], cv2.IMREAD_GRAYSCALE)
        if feature_img is None:
            raise FileNotFoundError(f"Could not load image: {self.cen2018(index)}")

        keypoints = []

        # --- Iterate over pixels and create KeyPoints for pixels > 0 ---
        rows, cols = feature_img.shape
        for y in range(rows):
            for x in range(cols):
                if feature_img[y, x] > 0:
                    kp = cv2.KeyPoint(x=float(x), y=float(y), size=1)
                    keypoints.append(kp)
        if len(keypoints) == 0:
            print(f'No keypoints found in image: {self.cen2018[index]}')
            fast = cv2.FastFeatureDetector_create()
            img = cv2.imread(self.imgs_path[index], cv2.IMREAD_GRAYSCALE)
            keypoints = fast.detect(img, None)
            
        return keypoints
    
    def __getitem__(self, index):
        
        img = cv2.imread(self.imgs_path[index], 0)
        if 0:  #test rotation
            mat = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2 ), np.random.randint(0,360), 1)
            img = cv2.warpAffine(img, mat, img.shape[:2])

        img = (img.astype(np.float32))/256 
        img = img[np.newaxis, :, :].repeat(3,0)
        
        return  img, index

    def __len__(self):
        return len(self.imgs_path)
    
#Evaluate function with F1 score calculation, no images saving

# def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):

#     if match_results_save_path is not None: 
#         os.system('mkdir -p ' + match_results_save_path)
#         all_errs = []
#         local_feats = local_feats.transpose(0,2,3,1)

#     gt_thres = 20
#     faiss_index = faiss.IndexFlatL2(global_descs.shape[1]) 
#     faiss_index.add(global_descs[:dataset.db_split_index])

#     # Query range
#     eval_start_split_point = dataset.db_split_index + int(200 / dataset.sample_inteval)
#     query_descs = global_descs[eval_start_split_point:]

#     dist, predictions = faiss_index.search(query_descs, 1)

#     # --- NEW: store evaluation data like code 1 ---
#     all_eval_data = []

#     all_positives = 0
#     tp = 0

#     for q_idx, pred in enumerate(predictions):

#         query_idx = eval_start_split_point + q_idx

#         gt_dis = (dataset.poses[query_idx] - dataset.poses[:dataset.db_split_index])**2
#         positives = np.where(np.sum(gt_dis[:, [3,7]], axis=1) < gt_thres**2)[0]

#         has_gt = len(positives) > 0
#         is_correct = pred[0] in positives if has_gt else False
#         dist_val = dist[q_idx][0]

#         # --- store for PR/F1 ---
#         all_eval_data.append((dist_val, is_correct, has_gt))

#         if has_gt:
#             all_positives += 1
#             if is_correct:
#                 tp += 1

#         # ================= DRAWING =================
#         if match_results_save_path is not None and has_gt:

#             index = pred[0]

#             query_im = dataset[query_idx][0].transpose(1,2,0)*256
#             db_im = dataset[index][0].transpose(1,2,0)*256

#             query_im = query_im.astype(np.uint8)
#             db_im = db_im.astype(np.uint8)

#             im_side = db_im.shape[0]

#             query_kps = dataset.getkeypoint(query_idx)
#             db_kps = dataset.getkeypoint(index)
                
#             query_des = [local_feats[query_idx][int(kp.pt[1]),int(kp.pt[0])] for kp in query_kps]
#             db_des = [local_feats[index][int(kp.pt[1]),int(kp.pt[0])] for kp in db_kps]

#             query_des = np.array(query_des)
#             db_des = np.array(db_des)
                
#             matcher = cv2.BFMatcher()
#             matches = matcher.knnMatch(query_des, db_des, k=2)

#             all_match = [m[0] for m in matches]
#             points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
#             points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

#             H, mask, _ = rigidRansac(
#                 (np.array([[im_side//2,im_side//2]]-points1)*0.64),
#                 (np.array([[im_side//2,im_side//2]]-points2))*0.64
#             )

#             q_pose = dataset.poses[query_idx]
#             q_pose = np.hstack((q_pose[:12].reshape(3,4)[:2,:2], q_pose[:12].reshape(3,4)[:2,3].reshape(-1,1)))
#             q_pose = np.vstack((q_pose,np.array([[0,0,1]])))

#             db_pose = dataset.poses[index]
#             db_pose = np.hstack((db_pose[:12].reshape(3,4)[:2,:2], db_pose[:12].reshape(3,4)[:2,3].reshape(-1,1)))
#             db_pose = np.vstack((db_pose,np.array([[0,0,1]])))

#             relative_gt = np.linalg.inv(db_pose).dot(q_pose)
#             relative_H = np.vstack((H, np.array([[0,0,1]])))

#             err = np.linalg.inv(relative_H).dot(relative_gt)
#             err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
#             err_trans = np.sqrt(err[0,2]**2 + err[1,2]**2)

#             all_errs.append([err_trans, err_theta])

#     # ================= BASIC RECALL =================
#     recall_top1 = tp / all_positives if all_positives > 0 else 0

#     # ================= NEW: GLOBAL PR + F1 =================
#     if len(all_eval_data) > 0:
#         dists = np.array([x[0] for x in all_eval_data])
#         is_correct = np.array([x[1] for x in all_eval_data])
#         has_gt = np.array([x[2] for x in all_eval_data])

#         thresholds = np.linspace(np.min(dists), np.max(dists), 100)
#         thresholds = np.unique(np.concatenate([thresholds, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]))
#         thresholds.sort()

#         best_f1, best_p, best_r, best_t = 0, 0, 0, 0
#         total_gt = np.sum(has_gt)

#         print("\n--- Evaluation Summary (Varied Thresholds) ---")
#         for t in thresholds:
#             accepted = dists < t

#             tp_ = np.sum(accepted & is_correct)
#             fp_ = np.sum(accepted & (~is_correct))
#             fn_ = np.sum((~accepted) & has_gt)

#             precision = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
#             recall = tp_ / total_gt if total_gt > 0 else 0
#             f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#             if f1 > best_f1:
#                 best_f1, best_p, best_r, best_t = f1, precision, recall, t

#         print(f"Max F1: {best_f1:.4f} @ Threshold: {best_t:.2f} (P: {best_p:.4f}, R: {best_r:.4f})")

#     # ================= RETURN =================
#     if match_results_save_path is not None:
#         all_errs = np.array(all_errs)
#         success_loc = (all_errs[:,0] < 25) & (all_errs[:,1] < 10)
#         success_rate = np.sum(success_loc) / all_positives if all_positives > 0 else 0
#         mean_trans_err = np.mean(all_errs[success_loc, 1]) if np.sum(success_loc) > 0 else 0
#         mean_rot_err = np.mean(all_errs[success_loc, 0]) if np.sum(success_loc) > 0 else 0

#         return recall_top1, success_rate, mean_trans_err, mean_rot_err
#     else:
#         return recall_top1

#Evaluate function without F1 score calculation and save images 

def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):

    if match_results_save_path is not None: 
        os.system('mkdir -p ' + match_results_save_path)
        all_errs = []
        local_feats = local_feats.transpose(0,2,3,1)

    gt_thres = 20  # gt threshold
    faiss_index = faiss.IndexFlatL2(global_descs.shape[1]) 
    faiss_index.add(global_descs[:dataset.db_split_index])

    _, predictions = faiss_index.search(global_descs[dataset.db_split_index+int(200/dataset.sample_inteval):], 1)  #top1
    
    
    eval_start_split_point = dataset.db_split_index+int(200/dataset.sample_inteval)
    all_positives = 0
    tp = 0
    for q_idx, pred in enumerate(predictions):

        query_idx = eval_start_split_point+q_idx
        gt_dis = (dataset.poses[query_idx] - dataset.poses[:dataset.db_split_index])**2
        positives = np.where(np.sum(gt_dis[:,[3,7,11]],axis=1) < gt_thres**2 )[0]
        if len(positives)>0:
            all_positives+=1
            if pred[0] in positives:
                tp += 1

            if match_results_save_path is not None:

                index = pred[0]


                query_im = dataset[query_idx][0].transpose(1,2,0)*256
                db_im = dataset[index][0].transpose(1,2,0)*256

                query_im = query_im.astype(np.uint8)
                db_im = db_im.astype(np.uint8)

                # fast = cv2.FastFeatureDetector_create()
                im_side = db_im.shape[0]

                # query_kps = fast.detect(query_im, None)
                # db_kps = fast.detect(db_im, None)
                query_kps = dataset.getkeypoint(query_idx)
                db_kps = dataset.getkeypoint(index)
                
                query_des = [local_feats[query_idx][int(kp.pt[1]),int(kp.pt[0])] for kp in query_kps]
                db_des = [local_feats[index][int(kp.pt[1]),int(kp.pt[0])] for kp in db_kps]
                # query_des = [local_feats[query_idx][int(kp.pt[1]), int(kp.pt[0])] for kp in query_kps]
                # db_des = [local_feats[index][int(kp.pt[1]), int(kp.pt[0])] for kp in db_kps]
                query_des = np.array(query_des)
                db_des = np.array(db_des)
                
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(query_des, db_des, k=2)
                
                

                all_match = [m[0] for m in matches]
                points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
                points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                H, mask, max_csc_num = rigidRansac((np.array([[im_side//2,im_side//2]]-points1)*0.64),(np.array([[im_side//2,im_side//2]]-points2))*0.64)# cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)
                
                q_pose = dataset.poses[query_idx]

                q_pose = np.hstack((q_pose[:12].reshape(3,4)[:2,:2], q_pose[:12].reshape(3,4)[:2,3].reshape(-1,1)))
                q_pose = np.vstack((q_pose,np.array([[0,0,1]])))

                db_pose = dataset.poses[index]
                db_pose = np.hstack((db_pose[:12].reshape(3,4)[:2,:2], db_pose[:12].reshape(3,4)[:2,3].reshape(-1,1)))
                db_pose = np.vstack((db_pose,np.array([[0,0,1]])))

                relative_gt = np.linalg.inv(db_pose).dot((q_pose))
                relative_H = np.vstack((H, np.array([[0,0,1]])))
                
                err = np.linalg.inv(relative_H).dot(relative_gt)
                err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
                err_trans = np.sqrt(err[0,2]**2+err[1,2]**2)

                if err_theta>10 or err_trans>25:
                    print('bug')
                all_errs.append([err_trans, err_theta])
                              
                good_match = [all_match[i] for i in range(len(mask)) if  mask[i]]
                db_im = db_im*3
                db_im[:,:,:2]=0


                im = cv2.drawMatches(query_im.astype(np.uint8), query_kps, db_im.astype(np.uint8), db_kps, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                out_im = np.zeros((im.shape[0]*2, db_im.shape[1]*3,3))
                out_im[:im.shape[0], :db_im.shape[1]] = query_im
                out_im[:im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im
                out_im[:im.shape[0], db_im.shape[1]*2:] = db_im+query_im

                out_im[-im.shape[0]:, :db_im.shape[1]*2] = im
                

                H = relative_H 
                mat = cv2.getRotationMatrix2D((query_im.shape[0]//2, query_im.shape[0]//2), np.arctan2(-H[0,1], H[0,0])/np.pi*180, 1.0)
                mat[0,2] -= H[1,2]/0.64
                mat[1,2] -= H[0,2]/0.64
                mat = np.vstack((mat,np.array([[0,0,1]])))
                mat = np.linalg.inv(mat)[:2,:]
                im_warp = cv2.warpAffine(db_im, mat, query_im.shape[:2])

                im_warp[:,:,:2]=0
                out_im[-im.shape[0]:, db_im.shape[1]*2:db_im.shape[1]*3] = im_warp+query_im   

                # add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                color = (0, 255, 0)
                thickness = 2
                text1 = f"err_trans: {err_trans:.2f}"
                text2 = f"err_theta: {err_theta:.2f}"
                
                cv2.putText(out_im, text1, (20, 40), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(out_im, text2, (20, 80), font, font_scale, color, thickness, cv2.LINE_AA)             
                cv2.imwrite(match_results_save_path+str(1000000+query_idx)[1:]+".png", out_im)

    
      
    recall_top1 = tp / all_positives #tp/(tp+fp)
    

    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        success_loc = (all_errs[:,0]<25) & (all_errs[:,1]<10)
        success_rate = np.sum(success_loc)/all_positives
        mean_trans_err = np.mean(all_errs[success_loc,1])
        mean_rot_err = np.mean(all_errs[success_loc,0]) 
        return recall_top1, success_rate, mean_trans_err, mean_rot_err
    else:
        return recall_top1


#Evaluate function for generating result file for visualizations

# def evaluateResults(global_descs, dataset, save_file_path="mulran_results.txt"):
#     """
#     Evaluates place recognition for MulRan single-session sequences.
#     Format: [Query_Filename] [Predicted_DB_Filename] [Is_Correct]
#     """
#     gt_thres = 20  # MulRan standard threshold (meters)
    
#     # 1. Define Split Points (following the original mulran_dataset.py logic)
#     db_split_idx = dataset.db_split_index
#     # Queries usually start 200m/frames after the DB ends to avoid trivial overlap
#     eval_start_idx = db_split_idx + int(200 / dataset.sample_inteval)
    
#     # 2. Setup Database (Indices 0 to db_split_idx)
#     db_descriptors = global_descs[:db_split_idx]
#     faiss_index = faiss.IndexFlatL2(db_descriptors.shape[1]) 
#     faiss_index.add(db_descriptors)
    
#     # Pre-calculate DB poses for distance checking
#     # MulRan poses are flattened 3x4 matrices; indices [3, 7, 11] are X, Y, Z
#     db_poses = dataset.poses[:db_split_idx]
    
#     # 3. Setup Queries (Indices eval_start_idx to end)
#     query_descriptors = global_descs[eval_start_idx:]
#     _, predictions = faiss_index.search(query_descriptors, 1)

#     with open(save_file_path, "w") as f:
#         f.write("query_filename db_filename is_correct\n")

#         for i, pred in enumerate(tqdm(predictions, desc="Evaluating MulRan")):
#             q_idx = eval_start_idx + i
#             db_pred_idx = pred[0]
            
#             # Get filenames from image paths
#             q_name = os.path.basename(dataset.imgs_path[q_idx])
#             db_name = os.path.basename(dataset.imgs_path[db_pred_idx])
            
#             # 4. Correctness Logic (3D Euclidean distance)
#             q_pose = dataset.poses[q_idx]
#             # Calculate distance to all potential DB entries
#             diff = db_poses - q_pose
#             # Translation components are at 3, 7, 11
#             dists_sq = np.sum(diff[:, [3, 7]]**2, axis=1)
#             gt_indices = np.where(dists_sq < gt_thres**2)[0]
            
#             is_correct = "False"
#             if len(gt_indices) > 0:
#                 if db_pred_idx in gt_indices:
#                     is_correct = "True"
#                 else:
#                     is_correct = "False"
#             else:
#                 # Same as OORD request: handle queries with no valid DB match
#                 is_correct = "No_GT_Available"
            
#             f.write(f"{q_name} {db_name} {is_correct}\n")

#     print(f"MulRan evaluation complete. Results saved to {save_file_path}")
        
def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices


class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path = '../../mulran/',seq='DCC01'):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+'/cartesian/' + seq + '_resize/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+'/cartesian/'+seq+'_resize/'+imgs_p[i] for i in range(0,len(imgs_p))]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+'/'+seq+'/pose.txt')[:mulran_seq_split_points[seq]]

        # gt_pose, only first 3000 frames of KITTI for training
        # self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')[:3000]

        
        # neg, pos threshold
        self.pos_thres = 20
        self.neg_thres = 25 # 

        # compute pos and negs for each query
        self.num_neg = 14
        self.positives = []
        self.negatives = []
        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            dises = np.sqrt(np.sum(((q_pose-self.poses)**2)[:,[3,7]],axis=1))            
            indexes = np.argsort(dises)

            remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:] #exclude query itself

            negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
            self.negatives.append(negs)

        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images



    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            pos_idx = self.positives[index][0]
        
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        query = cv2.imread(self.imgs_path[index])
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(0,360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])
        
        query = query.transpose(2,0,1)


        positive = cv2.imread(join(self.imgs_path[pos_idx]))#           
        mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2,0,1)
        

    
        query = (query.astype(np.float32))/256
        positive = (positive.astype(np.float32)/256)

        negatives = []

        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_path[neg_i])
            mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
            negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
            negative = negative.transpose(2,0,1)
            negative = (negative)/256
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)
