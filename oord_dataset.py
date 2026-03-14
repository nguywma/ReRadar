import os
from os.path import join, exists
import numpy as np
import pandas as pd
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data
from scipy.spatial.distance import cdist

import h5py
from tqdm import tqdm
import faiss
from scipy.spatial import cKDTree
from RANSAC import rigidRansac




class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path = '../../oord_data/', sample_inteval = 5):
        super().__init__()
        self.sample_inteval = sample_inteval
        # for cartesian
        imgs_p = os.listdir(dataset_path+'cartesian/'+seq + '_resize/')
        imgs_p.sort()

        self.imgs_path = [dataset_path+'cartesian/'+seq+'_resize/'+ imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]
        self.img = dataset_path+'cartesian/'+seq+'/' 
        # print(self.imgs_path)
        # #for polar
        # imgs_p = os.listdir(dataset_path +seq)
        # imgs_p.sort()

        # self.imgs_path = [dataset_path+seq+'/'+ imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]
        # self.img = dataset_path+seq+'/' 

        # gt_pose
        if 'Bellmouth_1' in seq:
            mapped_seq = '2021-11-25-12-01-20'
        elif 'Bellmouth_2' in seq:
            mapped_seq = '2021-11-25-12-31-19'
        elif 'Bellmouth_3' in seq:
            mapped_seq = '2021-11-26-15-35-34'
        elif 'Bellmouth_4' in seq:
            mapped_seq = '2021-11-26-16-12-01'
        elif 'Hydro_1' in seq:
            mapped_seq = '2021-11-27-14-37-20'
        elif 'Hydro_2' in seq:
            mapped_seq = '2021-11-27-15-24-02'
        elif 'Hydro_3' in seq:
            mapped_seq = '2021-11-27-16-03-26'
        elif 'Maree_1' in seq:
            mapped_seq = '2021-11-28-15-54-55'
        elif 'Maree_2' in seq:
            mapped_seq = '2021-11-28-16-43-37'
        elif 'Twolochs_1' in seq:
            mapped_seq = '2021-11-29-11-40-37'
        elif 'Twolochs_2' in seq:
            mapped_seq = '2021-11-29-12-19-16'
        else:
            mapped_seq = seq  # fallback, if none matched
        self.poses = pd.read_csv(dataset_path + 'pose/' + mapped_seq + '/gps.csv')
        self.imu = pd.read_csv(dataset_path + 'pose/' + mapped_seq + '/imu.csv')
        self.timestamps = [int(os.path.splitext(os.path.basename(path))[0]) for path in self.imgs_path] 
        self.cen2018path = dataset_path + 'cen2018/' + seq + '_resize/'
        self.posespath = dataset_path + 'pose/' + mapped_seq + '/gps.csv'
        self.cen2018 = [self.cen2018path + imgs_p[i] for i in range (0,len(imgs_p), sample_inteval)]
        # print(self.cen2018)
    def __getitem__(self, index):
        
        img = cv2.imread(self.imgs_path[index], 0)  
        img = (img.astype(np.float32))/256 
        img = img[np.newaxis, :, :].repeat(3,0)
        
        return  img, index
    def __len__(self):
        return len(self.imgs_path)
    
    def printpath(self):
        print(f'image path: {self.img}')
        print(f'gps file path: {self.posespath}')
        print(f'cen2018 feature path: {self.cen2018path}')
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
    
    @staticmethod
    def get_radar_positions(gps_file, radar_timestamps):

        # Use kd-tree for fast lookup
        gt_tss = gps_file.timestamp.to_numpy()
        keys = np.expand_dims(gt_tss, axis=-1)
        tree = cKDTree(keys)
        query = np.array(radar_timestamps)
        query = np.expand_dims(query, axis=-1)
        _, out = tree.query(query)
        gt_idxs = out.tolist()

        # Build output
        pos = {}
        for radar_timestamp, gt_idx in zip(radar_timestamps, gt_idxs):
            pos[radar_timestamp] = np.array(
                (gps_file.iloc[gt_idx].utm_northing, 
                gps_file.iloc[gt_idx].utm_easting))
        
        return pos
    
    # @staticmethod
    # def get_yaw(imu_file, radar_timestamp, gps_file):
    #     """
    #     Compute yaw from IMU at a radar timestamp.
    #     Can differentiate 180° rotations.
    #     Returns 3x3 SE(2) rotation matrix.
    #     """
    #     imu_cols = ['timestamp', 'magnetometer.x', 'magnetometer.y', 'magnetometer.z',
    #                 'accelerometer.x', 'accelerometer.y', 'accelerometer.z']
    #     imu_data = imu_file[imu_cols].dropna()
    #     if len(imu_data) < 2:
    #         raise ValueError("Insufficient IMU data")

    #     # Interpolate IMU readings at radar_timestamp
    #     mag_x = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.x'])
    #     mag_y = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.y'])
    #     mag_z = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.z'])
    #     acc_x = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.x'])
    #     acc_y = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.y'])
    #     acc_z = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.z'])

    #     # --- Step 1: tilt compensation ---
    #     roll = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
    #     pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

    #     mag_x_prime = mag_x * np.cos(pitch) + mag_z * np.sin(pitch)
    #     mag_y_prime = mag_x * np.sin(roll) * np.sin(pitch) + mag_y * np.cos(roll) - mag_z * np.sin(roll) * np.cos(pitch)

    #     # --- Step 2: compute yaw using full rotation matrix ---
    #     # Build SE(2) rotation from IMU directly
    #     # Keep both cos/sin to differentiate 180°
    #     cos_yaw = mag_x_prime / np.sqrt(mag_x_prime**2 + mag_y_prime**2)
    #     sin_yaw = -mag_y_prime / np.sqrt(mag_x_prime**2 + mag_y_prime**2)

    #     rotation = np.array([
    #         [cos_yaw, -sin_yaw, 0.0],
    #         [sin_yaw,  cos_yaw, 0.0],
    #         [0.0,      0.0,     1.0]
    #     ])

    #     return rotation
    # @staticmethod
    # def get_yaw(imu_file, radar_timestamp):
    #     # Columns needed from IMU
    #     imu_cols = ['timestamp', 'magnetometer.x', 'magnetometer.y', 'magnetometer.z',
    #                 'accelerometer.x', 'accelerometer.y', 'accelerometer.z']
    #     imu_data = imu_file[imu_cols].dropna()

    #     if len(imu_data) < 2:
    #         raise ValueError("Insufficient IMU data for yaw estimation")

    #     # Interpolate magnetometer and accelerometer data at radar_timestamp
    #     mag_x = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.x'])
    #     mag_y = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.y'])
    #     mag_z = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.z'])
    #     acc_x = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.x'])
    #     acc_y = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.y'])
    #     acc_z = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.z'])

    #     # Calculate roll and pitch for tilt compensation
    #     roll = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
    #     pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

    #     # Rotate magnetometer readings to account for tilt
    #     mag_x_prime = mag_x * np.cos(pitch) + mag_z * np.sin(pitch)
    #     mag_y_prime = mag_x * np.sin(roll) * np.sin(pitch) + mag_y * np.cos(roll) - mag_z * np.sin(roll) * np.cos(pitch)

    #     # Calculate yaw
    #     yaw_rad = np.arctan2(-mag_y_prime, mag_x_prime)

    #     # Build 2D transformation matrix (SE(2))
    #     cos_yaw = np.cos(yaw_rad)
    #     sin_yaw = np.sin(yaw_rad)
    #     rotation = np.array([
    #         [cos_yaw, -sin_yaw, 0.0],
    #         [sin_yaw,  cos_yaw, 0.0],
    #         [0,        0,       1]
    #     ])

    #     return rotation
    @staticmethod
    def get_yaw(imu_file, radar_timestamp, gps_file):
        """
        Compute yaw from GPS (if moving) or IMU (if stationary) at a radar timestamp.
        Returns 3x3 SE(2) rotation matrix.
        
        Priority:
        1. GPS Velocity Heading (if speed > 0.5 m/s)
        2. Magnetometer with Tilt Compensation (fallback)
        """
        
        # --- Part 1: Check GPS for Moving Heading ---
        use_gps = False
        gps_yaw = 0.0
        
        # Required GPS columns
        gps_cols = ['timestamp', 'velocity_heading', 'velocity_speed']
        
        # Ensure GPS data exists and is clean
        if gps_file is not None and not gps_file.empty:
            # Filter for rows where heading and speed are valid
            gps_data = gps_file[gps_cols].dropna()
            
            if len(gps_data) >= 2:
                # Interpolate Speed at the requested timestamp
                speed = np.interp(radar_timestamp, gps_data['timestamp'], gps_data['velocity_speed'])
                
                # Threshold: 0.5 m/s (approx 1.8 km/h) to consider the car "moving"
                if speed > 0:
                    # Interpolate Heading
                    # We must interpolate sin/cos components separately to handle 
                    # the 360-degree wrap-around (e.g., transition from 359 to 1).
                    headings_rad = np.radians(gps_data['velocity_heading'])
                    
                    heading_sin = np.sin(headings_rad)
                    heading_cos = np.cos(headings_rad)
                    
                    interp_sin = np.interp(radar_timestamp, gps_data['timestamp'], heading_sin)
                    interp_cos = np.interp(radar_timestamp, gps_data['timestamp'], heading_cos)
                    
                    # Reconstruct angle
                    gps_yaw = np.arctan2(interp_sin, interp_cos)
                    use_gps = True

        # --- Part 2: Calculate Rotation Matrix ---
        if use_gps:
            # Use GPS Heading
            cos_yaw = np.cos(gps_yaw)
            sin_yaw = np.sin(gps_yaw)
            
        else:
            # Use IMU (Magnetometer + Accelerometer)
            # Fallback to the logic provided in your snippet
            imu_cols = ['timestamp', 'magnetometer.x', 'magnetometer.y', 'magnetometer.z',
                        'accelerometer.x', 'accelerometer.y', 'accelerometer.z']
            imu_data = imu_file[imu_cols].dropna()
            
            if len(imu_data) < 2:
                raise ValueError("Insufficient IMU data")

            # Interpolate IMU readings at radar_timestamp
            mag_x = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.x'])
            mag_y = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.y'])
            mag_z = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['magnetometer.z'])
            acc_x = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.x'])
            acc_y = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.y'])
            acc_z = np.interp(radar_timestamp, imu_data['timestamp'], imu_data['accelerometer.z'])

            # Tilt compensation
            roll = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
            pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

            mag_x_prime = mag_x * np.cos(pitch) + mag_z * np.sin(pitch)
            mag_y_prime = mag_x * np.sin(roll) * np.sin(pitch) + mag_y * np.cos(roll) - mag_z * np.sin(roll) * np.cos(pitch)

            # Compute sin/cos directly from the compensated vector
            # Prevent division by zero
            norm = np.sqrt(mag_x_prime**2 + mag_y_prime**2)
            if norm < 1e-6:
                cos_yaw = 1.0
                sin_yaw = 0.0
            else:
                cos_yaw = mag_x_prime / norm
                sin_yaw = -mag_y_prime / norm

        # --- Part 3: Construct Rotation Matrix ---
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0]
        ])

        return rotation


        
        
class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path = '../../oord_data/',seq='Twolochs_2', infer_inteval=5,sample_inteval=2):
        # bev path
        # imgs_p = os.listdir(dataset_path+'cartesian/'+seq+ 'resize/')
        # self.sample_inteval = sample_inteval
        # self.imgs_path = [dataset_path+seq+'/'+ imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]
        # self.img = dataset_path+seq+'/' 

        imgs_p = os.listdir(dataset_path+'cartesian/'+seq + '_resize/')


        imgs_p.sort()

        # self.imgs_path = [dataset_path+'cartesian/'+seq+'_resize/'+ imgs_p[i] for i in range(0,len(imgs_p))]
        self.infer_path = [dataset_path+'cartesian/'+seq+'_resize/'+ imgs_p[i] for i in range(0,len(imgs_p), infer_inteval)]
        self.all_path = [dataset_path+'cartesian/'+seq+'_resize/'+ imgs_p[i] for i in range(0,len(imgs_p))]
        self.imgs_path = [x for x in self.all_path if x not in self.infer_path][::sample_inteval]

        self.img = dataset_path+'cartesian/'+seq+'_resize/' 
 
        # gt_pose
        if 'Bellmouth_1' in seq:
            mapped_seq = '2021-11-25-12-01-20'
        elif 'Bellmouth_2' in seq:
            mapped_seq = '2021-11-25-12-31-19'
        elif 'Bellmouth_3' in seq:
            mapped_seq = '2021-11-26-15-35-34'
        elif 'Bellmouth_4' in seq:
            mapped_seq = '2021-11-26-16-12-01'
        elif 'Hydro_1' in seq:
            mapped_seq = '2021-11-27-14-37-20'
        elif 'Hydro_2' in seq:
            mapped_seq = '2021-11-27-15-24-02'
        elif 'Hydro_3' in seq:
            mapped_seq = '2021-11-27-16-03-26'
        elif 'Maree_1' in seq:
            mapped_seq = '2021-11-28-15-54-55'
        elif 'Maree_2' in seq:
            mapped_seq = '2021-11-28-16-43-37'
        elif 'Twolochs_1' in seq:
            mapped_seq = '2021-11-29-11-40-37'
        elif 'Twolochs_2' in seq:
            mapped_seq = '2021-11-29-12-19-16'
        else:
            mapped_seq = seq  # fallback, if none matched

        self.poses = pd.read_csv(dataset_path + 'pose/' + mapped_seq + '/gps.csv')
        self.posespath = dataset_path+'pose/'+mapped_seq+'/gps.csv'
        self.timestamps = [int(os.path.splitext(os.path.basename(path))[0]) for path in self.imgs_path]
       
        # neg, pos threshold
        self.pos_thres = 25
        self.neg_thres = 27 # 

        # compute pos and negs for each query
        self.num_neg = 10
        self.positives = []
        self.negatives = []
        all_poses =  InferDataset.get_radar_positions(self.poses, self.timestamps)
        self.pose_array = np.array([all_poses[ts] for ts in self.timestamps])  # Convert to NumPy array
        print(f'len of pose_array{len(self.pose_array)}')
        for qi in range(len(self.timestamps)):  
            q_pose = all_poses[self.timestamps[qi]]
            dises = np.sqrt(np.sum(((q_pose-self.pose_array)**2),axis=1))            
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
        if self.cache is not None:
            h5 = h5py.File(self.cache, mode='r')
            # print(f'cache file: {self.cache}')
            self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            # print(f'h5feat len:{len(self.h5feat)}')
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
        if query is None:
            raise RuntimeError(f"Could not load image at index {index}: {self.imgs_path[index]}")
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

        # negatives = []

        # for neg_i in neg_idx:
        
        #     negative = cv2.imread(self.imgs_path[neg_i])
        #     mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
        #     negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
        #     negative = negative.transpose(2,0,1)
        #     negative = (negative)/256
            
        #     negatives.append(torch.from_numpy(negative.astype(np.float32)))

        # negatives = torch.stack(negatives, 0)
        negatives = []
        target_neg_count = 32

        # Loop until we reach 32 augmented samples
        while len(negatives) < target_neg_count:
            for neg_i in neg_idx: # neg_idx contains your 10 hard-mined indices
                if len(negatives) >= target_neg_count:
                    break
                    
                negative_img = cv2.imread(self.imgs_path[neg_i])
                # Use the "off-grid" rotation logic to challenge the C8 equivariance
                angle = np.random.uniform(0, 360)
                # Avoid multiples of 45 degrees
                if all(abs(angle - c8) > 5 for c8 in range(0, 361, 45)):
                    mat = cv2.getRotationMatrix2D((negative_img.shape[1]//2, negative_img.shape[0]//2), angle, 1)
                    negative_img = cv2.warpAffine(negative_img, mat, negative_img.shape[:2]) 
                    negative_img = negative_img.transpose(2,0,1) / 256.0
                    negatives.append(torch.from_numpy(negative_img.astype(np.float32)))

        negatives = torch.stack(negatives, 0) # Shape: [32, 3, H, W]
        return query, positive, negatives, index

    def __len__(self):
        return len(self.timestamps)


# def evaluateResults(global_descs, datasets, local_feats=None, match_results_save_path=None):

#     if match_results_save_path is not None: 
#         os.system('mkdir -p ' + match_results_save_path)
#         all_errs = []
#         if local_feats is not None:
#             print(f"number of local_feat datasets: {len(local_feats)}")
        
#         # --- REMOVED: In-place transpose on HDF5 dataset fails ---
#         # We assume data was saved as (N, H, W, C) using permute_local in infer()

#     gt_thres = 25  # Threshold for ground truth matching

#     # Initialize FAISS index using the first dataset's global descriptors
#     faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1]) 
#     faiss_index.add(global_descs[0])

#     recalls_oord = []
#     results = []

#     # Get reference dataset positions (first dataset)
#     db_pose_file = datasets[0].poses
#     db_timestamps = datasets[0].timestamps
#     db_imu = datasets[0].imu
#     db_positions = InferDataset.get_radar_positions(db_pose_file, db_timestamps)
#     print(f"length of db pos: {len(db_positions)}, length of db timestamp: {len(db_timestamps)}")

#     # Iterate over query datasets
#     for i in range(1, len(datasets)):
#         _, predictions = faiss_index.search(global_descs[i], 1)  # Top-1 search

#         all_positives = 0
#         tp = 0  
#         fn = 0
#         fp = 0
#         tn = 0
#         bug = 0
        
#         # Get ground truth positions for current dataset
#         query_pose_file = datasets[i].poses
#         query_timestamps = datasets[i].timestamps 
#         query_imu = datasets[i].imu 
#         query_positions = InferDataset.get_radar_positions(query_pose_file, query_timestamps)
        
#         print(f"length of query pos: {len(query_positions)}, length of query timestamp: {len(query_timestamps)}")
#         print(f"len of prediction: {len(predictions)}")

#         for q_idx, pred in enumerate(tqdm(predictions, desc="Evaluating")):
#             # Query position
#             query_timestamp = datasets[i].timestamps[q_idx]
#             if query_timestamp not in query_positions:
#                 continue  # Skip if timestamp has no matched position
            
#             pos1 = query_positions[query_timestamp]  # Query UTM position

#             # Compute Euclidean distance to all reference positions
#             pos2_list = np.array(list(db_positions.values()))
#             gt_dis = np.linalg.norm(pos2_list - pos1, axis=1)

#             # Find positives within the threshold
#             positives = np.where(gt_dis < gt_thres)[0]  

#             if len(positives) > 0:
#                 all_positives += 1
#                 if pred[0] in positives:
#                     tp += 1
#                 else:
#                     fn += 1
#             else:
#                 if pred[0] in positives:
#                     fp += 1
#                 else:
#                     tn += 1

#             # --- VISUALIZATION & GEOMETRIC VERIFICATION ---
#             if match_results_save_path is not None and local_feats is not None:
#                 index = pred[0] # The database index matched
                
#                 # Load images (assuming they are formatted correctly)
#                 query_im = datasets[i][q_idx][0].transpose(1,2,0)*256  
#                 db_im = datasets[0][index][0].transpose(1,2,0)*256
#                 query_im = query_im.astype(np.uint8)
#                 db_im = db_im.astype(np.uint8)  

#                 im_side = db_im.shape[0]

#                 # Get Keypoints
#                 query_kps = datasets[i].getkeypoint(q_idx)
#                 db_kps = datasets[0].getkeypoint(index)

#                 # --- OPTIMIZATION STARTS HERE ---
#                 # 1. Load the specific feature maps from Disk to RAM ONCE
#                 # local_feats[i] is the query HDF5 dataset, local_feats[0] is the DB HDF5 dataset
#                 q_feat_map = local_feats[i][q_idx]  # Shape: (H, W, C)
#                 db_feat_map = local_feats[0][index] # Shape: (H, W, C)

#                 # 2. Extract descriptors using list comprehension on the RAM object
#                 query_des = [q_feat_map[int(kp.pt[1]), int(kp.pt[0])] for kp in query_kps]
#                 db_des = [db_feat_map[int(kp.pt[1]), int(kp.pt[0])] for kp in db_kps]
#                 # --- OPTIMIZATION ENDS ---
                
#                 query_des = np.array(query_des)
#                 db_des = np.array(db_des)
                
#                 # Match local features
#                 matcher = cv2.BFMatcher()
#                 matches = matcher.knnMatch(query_des, db_des, k=2)

#                 all_match = [m[0] for m in matches]
#                 points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
#                 points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

#                 # Geometric Verification (RANSAC)
#                 H, mask, max_csc_num = rigidRansac(
#                     (np.array([[im_side//2,im_side//2]]-points1)*0.4),
#                     (np.array([[im_side//2,im_side//2]]-points2))*0.4
#                 )

#                 q_pose = InferDataset.get_yaw(query_imu, query_timestamp, query_pose_file)
#                 db_pose = InferDataset.get_yaw(db_imu, db_timestamps[index], db_pose_file)

#                 relative_gt = np.linalg.inv(db_pose).dot((q_pose))
#                 relative_H = np.vstack((H, np.array([[0,0,1]])))
                
#                 err = np.linalg.inv(relative_H).dot(relative_gt)
#                 err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
#                 err_trans = np.sqrt(err[0,2]**2+err[1,2]**2)

#                 if err_theta > 10 or err_trans > 25:
#                     # print('bug')
#                     bug += 1
#                 all_errs.append([err_trans, err_theta])
                                
#                 # Visualization logic
#                 good_match = [all_match[k] for k in range(len(mask)) if mask[k]]
#                 db_im_vis = db_im.copy() * 3
#                 db_im_vis[:,:,:2] = 0

#                 im = cv2.drawMatches(query_im, query_kps, db_im_vis.astype(np.uint8), db_kps, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
#                 out_im = np.zeros((im.shape[0]*2, db_im.shape[1]*3, 3))
#                 out_im[:im.shape[0], :db_im.shape[1]] = query_im
#                 out_im[:im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im
#                 out_im[:im.shape[0], db_im.shape[1]*2:] = db_im + query_im

#                 out_im[-im.shape[0]:, :db_im.shape[1]*2] = im
                
#                 H = relative_H 
#                 mat = cv2.getRotationMatrix2D((query_im.shape[0]//2, query_im.shape[0]//2), np.arctan2(-H[0,1], H[0,0])/np.pi*180, 1.0)
#                 mat[0,2] -= H[1,2]/0.4
#                 mat[1,2] -= H[0,2]/0.4
#                 mat = np.vstack((mat,np.array([[0,0,1]])))
#                 mat = np.linalg.inv(mat)[:2,:]
#                 im_warp = cv2.warpAffine(db_im, mat, query_im.shape[:2])

#                 im_warp[:,:,:2]=0
#                 out_im[-im.shape[0]:, db_im.shape[1]*2:db_im.shape[1]*3] = im_warp + query_im     
                
#                 # Add text
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 1.0
#                 color = (0, 255, 0)
#                 thickness = 2
#                 text1 = f"err_trans: {err_trans:.2f}"
#                 text2 = f"err_theta: {err_theta:.2f}"
                
#                 cv2.putText(out_im, text1, (20, 40), font, font_scale, color, thickness, cv2.LINE_AA)
#                 cv2.putText(out_im, text2, (20, 80), font, font_scale, color, thickness, cv2.LINE_AA)
                
#                 cv2.imwrite(match_results_save_path + str(1000000+q_idx)[1:] + ".png", out_im)

#     recall_top1 = tp / (tp + fn) if all_positives > 0 else 0
#     recalls_oord.append(recall_top1)
#     results.append({"TP": tp, "FN": fn, "FP": fp, "TN": tn, "AP": all_positives})
#     print(f"number of bugs: {bug}")

#     if match_results_save_path is not None:
#         all_errs = np.array(all_errs)
#         # Handle case where all_errs might be empty
#         if len(all_errs) > 0:
#             success_loc = (all_errs[:,0] < 25) & (all_errs[:,1] < 10)
#             success_rate = np.sum(success_loc) / all_positives if all_positives > 0 else 0
#             mean_trans_err = np.mean(all_errs[success_loc,0]) if np.any(success_loc) else 0
#             mean_rot_err = np.mean(all_errs[success_loc,1]) if np.any(success_loc) else 0
#         else:
#             success_rate, mean_trans_err, mean_rot_err = 0, 0, 0
            
#         return recalls_oord, success_rate, mean_trans_err, mean_rot_err, results
#     else:
#         return recalls_oord, results

def evaluateResults(global_descs, datasets, local_feats=None, match_results_save_path=None):
    if match_results_save_path is not None: 
        os.system('mkdir -p ' + match_results_save_path)
    
    all_errs = []
    gt_thres = 25
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1]) 
    faiss_index.add(global_descs[0])

    recalls_oord = []
    results = []

    db_pose_file = datasets[0].poses
    db_timestamps = datasets[0].timestamps
    db_imu = datasets[0].imu
    db_positions = InferDataset.get_radar_positions(db_pose_file, db_timestamps)

    for i in range(1, len(datasets)):
        _, predictions = faiss_index.search(global_descs[i], 1)
        all_positives, tp, fn, fp, tn, bug = 0, 0, 0, 0, 0, 0
        
        query_pose_file = datasets[i].poses
        query_timestamps = datasets[i].timestamps 
        query_imu = datasets[i].imu 
        query_positions = InferDataset.get_radar_positions(query_pose_file, query_timestamps)

        for q_idx, pred in enumerate(tqdm(predictions, desc="Evaluating")):
            query_timestamp = datasets[i].timestamps[q_idx]
            if query_timestamp not in query_positions:
                continue
            
            pos1 = query_positions[query_timestamp]
            pos2_list = np.array(list(db_positions.values()))
            gt_dis = np.linalg.norm(pos2_list - pos1, axis=1)
            positives = np.where(gt_dis < gt_thres)[0]  

            if len(positives) > 0:
                all_positives += 1
                if pred[0] in positives: tp += 1
                else: fn += 1
            else:
                if pred[0] in positives: fp += 1
                else: tn += 1

            # --- SIMPLIFIED DRAWING LOGIC ---
            if match_results_save_path is not None and local_feats is not None:
                index = pred[0]
                
                # Prepare Images
                query_im = (datasets[i][q_idx][0].transpose(1,2,0)*256).astype(np.uint8)
                db_im = (datasets[0][index][0].transpose(1,2,0)*256).astype(np.uint8)
                im_side = db_im.shape[0]

                # Get Keypoints and Descriptors
                query_kps = datasets[i].getkeypoint(q_idx)
                db_kps = datasets[0].getkeypoint(index)
                q_feat_map = local_feats[i][q_idx]
                db_feat_map = local_feats[0][index]
                
                query_des = np.array([q_feat_map[int(kp.pt[1]), int(kp.pt[0])] for kp in query_kps])
                db_des = np.array([db_feat_map[int(kp.pt[1]), int(kp.pt[0])] for kp in db_kps])
                
                # Match
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(query_des, db_des, k=2)
                all_match = [m[0] for m in matches]
                points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
                points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                # RANSAC
                H, mask, _ = rigidRansac(
                    (np.array([[im_side//2, im_side//2]] - points1) * 0.256),
                    (np.array([[im_side//2, im_side//2]] - points2) * 0.256)
                )

                # Error Calculation
                q_pose = InferDataset.get_yaw(query_imu, query_timestamp, query_pose_file)
                db_pose = InferDataset.get_yaw(db_imu, db_timestamps[index], db_pose_file)
                relative_gt = np.linalg.inv(db_pose).dot(q_pose)
                relative_H = np.vstack((H, np.array([[0,0,1]])))
                err = np.linalg.inv(relative_H).dot(relative_gt)
                err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
                err_trans = np.sqrt(err[0,2]**2 + err[1,2]**2)

                if err_theta > 10 or err_trans > 25: bug += 1
                all_errs.append([err_trans, err_theta])
                                
                # DRAW ONLY MATCHES
                good_match = [all_match[k] for k in range(len(mask)) if mask[k]]
                
                # Create the simple side-by-side match image
                match_vis = cv2.drawMatches(
                    query_im, query_kps, 
                    db_im, db_kps, 
                    good_match, None, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                
                # Overlay Error Text
                # cv2.putText(match_vis, f"Trans Err: {err_trans:.2f}m", (20, 40), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # cv2.putText(match_vis, f"Rot Err: {err_theta:.2f}deg", (20, 70), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Save
                cv2.imwrite(os.path.join(match_results_save_path, f"{q_idx:06d}.png"), match_vis)

    # ... (Rest of the evaluation summary logic remains the same)
    recall_top1 = tp / (tp + fn) if all_positives > 0 else 0
    recalls_oord.append(recall_top1)
    results.append({"TP": tp, "FN": fn, "FP": fp, "TN": tn, "AP": all_positives})
    
    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        if len(all_errs) > 0:
            success_loc = (all_errs[:,0] < 25) & (all_errs[:,1] < 10)
            success_rate = np.sum(success_loc) / all_positives if all_positives > 0 else 0
            mean_trans_err = np.mean(all_errs[success_loc,0]) if np.any(success_loc) else 0
            mean_rot_err = np.mean(all_errs[success_loc,1]) if np.any(success_loc) else 0
        else:
            success_rate, mean_trans_err, mean_rot_err = 0, 0, 0
        return recalls_oord, success_rate, mean_trans_err, mean_rot_err, results
    return recalls_oord, results

