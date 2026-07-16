# ReRadar: Robust Radar Global Localization via Rotation-Equivariant Descriptor Learning

ReRadar is a PyTorch framework for learning rotation-equivariant global descriptors for radar-based place recognition and global localization. 

## Requirements

Install dependencies from `requirement.txt`:

```bash
pip install -r requirement.txt
```

`requirement.txt`:

```
torch == 1.13.1
torchvision == 0.14.1
faiss-gpu
numpy
opencv-python
scikit-image
scikit-learn
tqdm
argparse
h5py
tensorboardX
imgaug
mmcv-full
e2cnn
```

Additionally, the following project-local modules are required: `kitti_dataset`, `oord_dataset`, `loss` (TripletLoss / InfoNCE), and a `model` package containing the REIN backbone variants.

## Dataset Structure
 
The `--path` argument should point to a dataset root organized as follows:
 OORD
```
--path
├── cartesian
│   ├── Bellmouth_1_resize
│   ├── Bellmouth_2_resize
│   ├── Twolochs_1_resize
│   ├── Twolochs_2_resize
│   ├── Hydro_1_resize
│   ├── Hydro_2_resize
│   ├── Hydro_3_resize
│   ├── Maree_1_resize
│   └── Maree_2_resize
└── pose
```
<!-- Mulran
```
--path
``` -->

## Usage

### Training

```bash
python main.py --mode train \
    --path ../../oord_data/ \
    --network rerein \
    --loss infonce \
    --batchSize 2  \
    --nEpochs 60 \
    --lr 1e-5
```

Key training behavior:
- Loads four training sequences (`Twolochs_2`, `Maree_1`, `Bellmouth_2`, `Hydro_1`) and concatenates them into a single training set.
- Each epoch first runs a sequential pass to cache global descriptors for hard-mining, then trains with shuffled batches.
- Validation is run after every epoch on five sequence pairs (Bellmouth, Twolochs, Hydro×2, Maree), computing recall per pair and averaging.
- Checkpoints (`checkpoint.pth.tar` and `model_best.pth.tar`) and epoch logs are saved under `runsPath/<timestamp>/`.
- TensorBoard scalars are written for training loss, learning rate, and per-pair/mean validation recall.

### Testing / Evaluation

```bash
python main.py --mode test \
    --path ../../oord_data/ \
    --network rerein \
    --load_from runs/re50_kitti_oord0 \
    --ckpt best
```

This extracts global descriptors for a configured sequence pair (default: `Twolochs_2` vs `Twolochs_1`), computes retrieval recall via `oord_dataset.evaluateResults`, and reports Recall@1, Precision, F1, and Average Precision (AP).

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `test` | `train` or `test` |
| `--path` | `../../oord_data/` | Path to the dataset root |
| `--batchSize` | `8` | Number of triplets per batch |
| `--cacheBatchSize` | `8` | Batch size used for caching/inference |
| `--nEpochs` | `60` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--lrStep` / `--lrGamma` | `10` / `0.5` | Step decay schedule (if scheduler is enabled) |
| `--weightDecay` | `1e-3` | Weight decay |
| `--loss` | `infonce` | `triplet` or `infonce` |
| `--threads` | `16` | DataLoader worker threads |
| `--seed` | `1024` | Random seed |
| `--network` | `rerein` | Backbone variant: `rein`, `erein`, `e18rein`, `e34rein`, `rerein` |
| `--load_from` | `runs/re50_kitti_oord0` | Checkpoint directory to resume/load from |
| `--ckpt` | `best` | `latest` or `best` checkpoint file |
| `--runsPath` | `./runs/` | Output directory for logs/checkpoints |
| `--cachePath` | `./cache/` | Output directory for NetVLAD cluster cache |

## Checkpoint Loading Notes

- If `--load_from` points to a valid directory containing `checkpoint.pth.tar` or `model_best.pth.tar`, that state dict is loaded directly (standard path).
- A special-cased path (`network == 're1rein'`) supports loading backbone-only checkpoints: weights are remapped under an `rem.encoder.` prefix, loaded with `strict=False`, and the NetVLAD pooling layer is then initialized from freshly computed k-means clusters if no cluster cache exists yet.
- If `--load_from` is not provided at all, the script computes NetVLAD clusters from scratch using `getClusters` before initializing the pooling layer.

## Outputs

- **Checkpoints**: `runsPath/<run_name>/checkpoint.pth.tar` and `model_best.pth.tar`.
- **Logs**: `flags.json` (run configuration), `epoch_losses.txt` (per-epoch loss/recall), TensorBoard event files.
- **Cluster cache**: `cachePath/desc_cen.hdf5` (centroids + sampled descriptors for NetVLAD initialization).

## Citation

If you use this code in your research, please cite the corresponding paper.