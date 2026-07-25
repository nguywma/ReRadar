"""
boreas_dataset.py

Drop-in replacement for oord_dataset.py, adapted for the Boreas dataset layout:

    <dataset_path>/
        boreas-2020-11-26-13-58/
            radar/                  # raw polar radar pngs (Navtech format)
            cart/                   # cartesian radar images (output of cartesian.py)
            cen2018/                # cen2018 keypoint feature images (output of cen2018_feature_gen_new.py)
            applanix/
                radar_poses.csv     # per-radar-frame pose: timestamp, easting, northing, heading, ...
                raw_logs/           # raw NovAtel logs (unused here)
            calib/

Unlike OORD, Boreas ships a `radar_poses.csv` file that already gives one pose row
PER RADAR FRAME (indexed by the same timestamp used as the radar/cart/cen2018 png
filename), including a `heading` column. This means:
  - No sequence-name -> date-folder remapping is needed (each 'boreas-*' folder is
    self-contained).
  - No GPS/IMU interpolation is needed for yaw: `heading` is read straight from
    radar_poses.csv.
  - Position lookup is still done via a KD-tree nearest-timestamp match (as in
    oord_dataset.py) for robustness in case a frame's exact timestamp is missing
    from the poses csv, but in practice it should be an exact/near-exact match.

The public interface (InferDataset, TrainingDataset, evaluateResults, and the
static helpers get_radar_positions / get_yaw) mirrors oord_dataset.py exactly, so
this module is intended to be used as a straight swap in main.py, e.g.:

    import boreas_dataset as oord_dataset

or by changing the relevant `oord_dataset.*` calls in main.py to `boreas_dataset.*`
and updating `train_sequences` / `eval_pairs` / `--path` to Boreas sequence names
(e.g. 'boreas-2020-11-26-13-58') instead of the OORD ones.
"""

import os
from os.path import join, exists
import numpy as np
import pandas as pd
import cv2
import torch
import torch.utils.data as data
from scipy.spatial.distance import cdist

import h5py
from tqdm import tqdm
import faiss
from scipy.spatial import cKDTree
from RANSAC import rigidRansac


# ------------------------------------------------------------------------------------
# Pose CSV helpers
# ------------------------------------------------------------------------------------

# Candidate relative paths (within a sequence folder) for the per-radar-frame pose file
_POSE_CSV_CANDIDATES = [
    join('applanix', 'radar_poses.csv'),
    join('applanix', 'raw_logs', 'radar_poses.csv'),
]

# Candidate column names, in case different Boreas devkit versions / re-exports differ
_TIMESTAMP_COLS = ['timestamp', 'GPSTime', 'time']
_EASTING_COLS = ['easting', 'utm_easting']
_NORTHING_COLS = ['northing', 'utm_northing']
_HEADING_COLS = ['heading', 'yaw']


def _find_pose_csv(seq_dir):
    for cand in _POSE_CSV_CANDIDATES:
        path = join(seq_dir, cand)
        if exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find a radar pose csv under {seq_dir} "
        f"(tried: {_POSE_CSV_CANDIDATES})"
    )


def _resolve_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a {what} column in pose csv. "
                    f"Tried {candidates}, available columns: {list(df.columns)}")


def load_boreas_poses(seq_dir):
    """Load a Boreas sequence's per-radar-frame pose csv and normalize column names
    to 'timestamp', 'utm_easting', 'utm_northing', 'heading' (matching the naming
    convention used elsewhere in this codebase, e.g. InferDataset.get_radar_positions)."""
    csv_path = _find_pose_csv(seq_dir)
    df = pd.read_csv(csv_path)

    ts_col = _resolve_col(df, _TIMESTAMP_COLS, 'timestamp')
    e_col = _resolve_col(df, _EASTING_COLS, 'easting')
    n_col = _resolve_col(df, _NORTHING_COLS, 'northing')

    df = df.rename(columns={ts_col: 'timestamp', e_col: 'utm_easting', n_col: 'utm_northing'})

    try:
        h_col = _resolve_col(df, _HEADING_COLS, 'heading')
        df = df.rename(columns={h_col: 'heading'})
    except KeyError:
        # Heading isn't strictly required unless get_yaw() is used.
        pass

    df['timestamp'] = df['timestamp'].astype(np.int64)
    return df


# ------------------------------------------------------------------------------------
# Inference / evaluation dataset
# ------------------------------------------------------------------------------------

class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path='../../boreas_data/', sample_inteval=5,
                 cart_subfolder='cart', cen2018_subfolder='cen2018'):
        super().__init__()
        self.sample_inteval = sample_inteval
        self.seq_dir = join(dataset_path, seq)

        # cartesian images
        imgs_p = os.listdir(join(self.seq_dir, cart_subfolder))
        imgs_p.sort()

        self.imgs_path = [join(self.seq_dir, cart_subfolder, imgs_p[i])
                           for i in range(0, len(imgs_p), sample_inteval)]
        self.img = join(self.seq_dir, cart_subfolder) + '/'

        # gt pose - Boreas sequences are self-contained, no name remapping needed
        self.poses = load_boreas_poses(self.seq_dir)
        self.posespath = _find_pose_csv(self.seq_dir)
        # No separate IMU file is needed (heading comes from radar_poses.csv directly),
        # kept for interface-compatibility with evaluateResults / get_yaw signatures.
        self.imu = self.poses

        self.timestamps = [int(os.path.splitext(os.path.basename(path))[0]) for path in self.imgs_path]

        self.cen2018path = join(self.seq_dir, cen2018_subfolder) + '/'
        self.cen2018 = [join(self.seq_dir, cen2018_subfolder, imgs_p[i])
                         for i in range(0, len(imgs_p), sample_inteval)]

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index], 0)
        img = (img.astype(np.float32)) / 256
        img = img[np.newaxis, :, :].repeat(3, 0)
        return img, index

    def __len__(self):
        return len(self.imgs_path)

    def printpath(self):
        print(f'image path: {self.img}')
        print(f'pose file path: {self.posespath}')
        print(f'cen2018 feature path: {self.cen2018path}')

    def getkeypoint(self, index):
        feature_img = cv2.imread(self.cen2018[index], cv2.IMREAD_GRAYSCALE)
        if feature_img is None:
            raise FileNotFoundError(f"Could not load image: {self.cen2018[index]}")

        keypoints = []
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
        # Use kd-tree for fast lookup (radar_poses.csv is indexed per-frame, so this
        # should typically be an exact match, but we keep nearest-neighbor lookup
        # for robustness against any missing frames).
        gt_tss = gps_file.timestamp.to_numpy()
        keys = np.expand_dims(gt_tss, axis=-1)
        tree = cKDTree(keys)
        query = np.array(radar_timestamps)
        query = np.expand_dims(query, axis=-1)
        _, out = tree.query(query)
        gt_idxs = out.tolist()

        pos = {}
        for radar_timestamp, gt_idx in zip(radar_timestamps, gt_idxs):
            pos[radar_timestamp] = np.array(
                (gps_file.iloc[gt_idx].utm_northing,
                 gps_file.iloc[gt_idx].utm_easting))

        return pos

    @staticmethod
    def get_yaw(imu_file, radar_timestamp, gps_file):
        """
        Compute yaw (SE(2) rotation matrix) at a radar timestamp.

        Boreas's radar_poses.csv already provides a 'heading' column per radar
        frame, so (unlike OORD) no GPS-velocity / magnetometer fallback logic is
        needed here. `imu_file` is accepted only for interface-compatibility with
        oord_dataset.evaluateResults() and is unused.
        """
        if gps_file is None or 'heading' not in gps_file.columns:
            raise ValueError("gps_file must be the sequence's radar_poses dataframe "
                              "and contain a 'heading' column.")

        heading_rad = np.interp(radar_timestamp, gps_file['timestamp'], gps_file['heading'])

        cos_yaw = np.cos(heading_rad)
        sin_yaw = np.sin(heading_rad)

        rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0]
        ])

        return rotation


# ------------------------------------------------------------------------------------
# Training dataset
# ------------------------------------------------------------------------------------

class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path='../../boreas_data/', seq='boreas-2020-11-26-13-58',
                 infer_inteval=5, sample_inteval=2, cart_subfolder='cart'):
        self.seq_dir = join(dataset_path, seq)

        imgs_p = os.listdir(join(self.seq_dir, cart_subfolder))
        imgs_p.sort()

        self.infer_path = [join(self.seq_dir, cart_subfolder, imgs_p[i])
                            for i in range(0, len(imgs_p), infer_inteval)]
        self.all_path = [join(self.seq_dir, cart_subfolder, imgs_p[i])
                          for i in range(0, len(imgs_p))]
        self.imgs_path = [x for x in self.all_path if x not in self.infer_path][::sample_inteval]

        self.img = join(self.seq_dir, cart_subfolder) + '/'

        # gt pose - self-contained, no name remapping needed for Boreas
        self.poses = load_boreas_poses(self.seq_dir)
        self.posespath = _find_pose_csv(self.seq_dir)
        self.timestamps = [int(os.path.splitext(os.path.basename(path))[0]) for path in self.imgs_path]

        # neg, pos threshold
        self.pos_thres = 25
        self.neg_thres = 27

        # compute pos and negs for each query
        self.num_neg = 10
        self.positives = []
        self.negatives = []
        all_poses = InferDataset.get_radar_positions(self.poses, self.timestamps)
        self.pose_array = np.array([all_poses[ts] for ts in self.timestamps])
        print(f'len of pose_array{len(self.pose_array)}')
        for qi in range(len(self.timestamps)):
            q_pose = all_poses[self.timestamps[qi]]
            dises = np.sqrt(np.sum(((q_pose - self.pose_array) ** 2), axis=1))
            indexes = np.argsort(dises)

            remap_index = indexes[np.where(dises[indexes] < self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:]  # exclude query itself

            negs = indexes[np.where(dises[indexes] > self.neg_thres)[0]]
            self.negatives.append(negs)

        self.mining = False
        self.cache = None  # filepath of HDF5 containing feature vectors for images

    # refresh cache for hard mining
    def refreshCache(self):
        if self.cache is not None:
            h5 = h5py.File(self.cache, mode='r')
            self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1, -1) - pos_feat) ** 2, axis=1))

            min_idx = np.where(dis_pos == np.max(dis_pos))[0][0]
            pos_idx = np.random.choice(self.positives[index], 1)[0]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1, -1) - neg_feat) ** 2, axis=1))

            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg - 1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

        else:
            pos_idx = self.positives[index][0]

            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]

        query = cv2.imread(self.imgs_path[index])
        if query is None:
            raise RuntimeError(f"Could not load image at index {index}: {self.imgs_path[index]}")
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1] // 2, query.shape[0] // 2), np.random.randint(0, 360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])
        query = query.transpose(2, 0, 1)

        positive = cv2.imread(join(self.imgs_path[pos_idx]))
        mat = cv2.getRotationMatrix2D((positive.shape[1] // 2, positive.shape[0] // 2), np.random.randint(0, 360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2, 0, 1)

        query = (query.astype(np.float32)) / 256
        positive = (positive.astype(np.float32) / 256)

        negatives = []
        target_neg_count = 32

        while len(negatives) < target_neg_count:
            for neg_i in neg_idx:
                if len(negatives) >= target_neg_count:
                    break

                negative_img = cv2.imread(self.imgs_path[neg_i])
                angle = np.random.uniform(0, 360)
                if all(abs(angle - c8) > 5 for c8 in range(0, 361, 45)):
                    mat = cv2.getRotationMatrix2D((negative_img.shape[1] // 2, negative_img.shape[0] // 2), angle, 1)
                    negative_img = cv2.warpAffine(negative_img, mat, negative_img.shape[:2])
                    negative_img = negative_img.transpose(2, 0, 1) / 256.0
                    negatives.append(torch.from_numpy(negative_img.astype(np.float32)))

        negatives = torch.stack(negatives, 0)  # Shape: [32, 3, H, W]
        return query, positive, negatives, index

    def __len__(self):
        return len(self.timestamps)


# ------------------------------------------------------------------------------------
# Evaluation (identical logic to oord_dataset.evaluateResults; dataset-agnostic
# aside from relying on .poses / .timestamps / .imu / get_radar_positions / get_yaw,
# all of which InferDataset above provides with the same names.)
# ------------------------------------------------------------------------------------

def evaluateResults(global_descs, datasets, local_feats=None, match_results_save_path=None):

    if match_results_save_path is not None:
        os.system('mkdir -p ' + match_results_save_path)
        all_errs = []
        if local_feats is not None:
            print(f"number of local_feat datasets: {len(local_feats)}")

    gt_thres = 20  # Threshold for ground truth matching

    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1])
    faiss_index.add(global_descs[0])

    recalls_oord = []
    results = []

    db_pose_file = datasets[0].poses
    db_timestamps = datasets[0].timestamps
    db_imu = datasets[0].imu
    db_positions = InferDataset.get_radar_positions(db_pose_file, db_timestamps)
    print(f"length of db pos: {len(db_positions)}, length of db timestamp: {len(db_timestamps)}")

    for i in range(1, len(datasets)):
        _, predictions = faiss_index.search(global_descs[i], 1)  # Top-1 search

        all_positives = 0
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        bug = 0

        query_pose_file = datasets[i].poses
        query_timestamps = datasets[i].timestamps
        query_imu = datasets[i].imu
        query_positions = InferDataset.get_radar_positions(query_pose_file, query_timestamps)

        print(f"length of query pos: {len(query_positions)}, length of query timestamp: {len(query_timestamps)}")
        print(f"len of prediction: {len(predictions)}")

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
                if pred[0] in positives:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[0] in positives:
                    fp += 1
                else:
                    tn += 1

            if match_results_save_path is not None and local_feats is not None:
                index = pred[0]

                query_im = datasets[i][q_idx][0].transpose(1, 2, 0) * 256
                db_im = datasets[0][index][0].transpose(1, 2, 0) * 256
                query_im = query_im.astype(np.uint8)
                db_im = db_im.astype(np.uint8)

                im_side = db_im.shape[0]

                query_kps = datasets[i].getkeypoint(q_idx)
                db_kps = datasets[0].getkeypoint(index)

                q_feat_map = local_feats[i][q_idx]
                db_feat_map = local_feats[0][index]

                query_des = [q_feat_map[int(kp.pt[1]), int(kp.pt[0])] for kp in query_kps]
                db_des = [db_feat_map[int(kp.pt[1]), int(kp.pt[0])] for kp in db_kps]

                query_des = np.array(query_des)
                db_des = np.array(db_des)

                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(query_des, db_des, k=2)

                all_match = [m[0] for m in matches]
                points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match])
                points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                H, mask, max_csc_num = rigidRansac(
                    (np.array([[im_side // 2, im_side // 2]] - points1) * 0.64),
                    (np.array([[im_side // 2, im_side // 2]] - points2)) * 0.64
                )

                q_pose = InferDataset.get_yaw(query_imu, query_timestamp, query_pose_file)
                db_pose = InferDataset.get_yaw(db_imu, db_timestamps[index], db_pose_file)

                relative_gt = np.linalg.inv(db_pose).dot((q_pose))
                relative_H = np.vstack((H, np.array([[0, 0, 1]])))

                err = np.linalg.inv(relative_H).dot(relative_gt)
                err_theta = np.abs(np.arctan2(err[0, 1], err[0, 0]) / np.pi * 180)
                err_trans = np.sqrt(err[0, 2] ** 2 + err[1, 2] ** 2)

                if err_theta > 10 or err_trans > 25:
                    bug += 1
                all_errs.append([err_trans, err_theta])

                good_match = [all_match[k] for k in range(len(mask)) if mask[k]]
                db_im_vis = db_im.copy() * 3
                db_im_vis[:, :, :2] = 0

                im = cv2.drawMatches(query_im, query_kps, db_im_vis.astype(np.uint8), db_kps, good_match, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                out_im = np.zeros((im.shape[0] * 2, db_im.shape[1] * 3, 3))
                out_im[:im.shape[0], :db_im.shape[1]] = query_im
                out_im[:im.shape[0], db_im.shape[1]:db_im.shape[1] * 2] = db_im
                out_im[:im.shape[0], db_im.shape[1] * 2:] = db_im + query_im

                out_im[-im.shape[0]:, :db_im.shape[1] * 2] = im

                H = relative_H
                mat = cv2.getRotationMatrix2D((query_im.shape[0] // 2, query_im.shape[0] // 2),
                                               np.arctan2(-H[0, 1], H[0, 0]) / np.pi * 180, 1.0)
                mat[0, 2] -= H[1, 2] / 0.64
                mat[1, 2] -= H[0, 2] / 0.64
                mat = np.vstack((mat, np.array([[0, 0, 1]])))
                mat = np.linalg.inv(mat)[:2, :]
                im_warp = cv2.warpAffine(db_im, mat, query_im.shape[:2])

                im_warp[:, :, :2] = 0
                out_im[-im.shape[0]:, db_im.shape[1] * 2:db_im.shape[1] * 3] = im_warp + query_im

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                color = (0, 255, 0)
                thickness = 2
                text1 = f"err_trans: {err_trans:.2f}"
                text2 = f"err_theta: {err_theta:.2f}"

                cv2.putText(out_im, text1, (20, 40), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(out_im, text2, (20, 80), font, font_scale, color, thickness, cv2.LINE_AA)

                cv2.imwrite(match_results_save_path + str(1000000 + q_idx)[1:] + ".png", out_im)

    recall_top1 = tp / (tp + fn) if all_positives > 0 else 0
    recalls_oord.append(recall_top1)
    results.append({"TP": tp, "FN": fn, "FP": fp, "TN": tn, "AP": all_positives})
    print(f"number of bugs: {bug}")

    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        if len(all_errs) > 0:
            success_loc = (all_errs[:, 0] < 25) & (all_errs[:, 1] < 10)
            success_rate = np.sum(success_loc) / all_positives if all_positives > 0 else 0
            mean_trans_err = np.mean(all_errs[success_loc, 0]) if np.any(success_loc) else 0
            mean_rot_err = np.mean(all_errs[success_loc, 1]) if np.any(success_loc) else 0
        else:
            success_rate, mean_trans_err, mean_rot_err = 0, 0, 0

        return recalls_oord, success_rate, mean_trans_err, mean_rot_err, results
    else:
        return recalls_oord, results