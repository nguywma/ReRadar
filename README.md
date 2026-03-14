# radarradar
Inference on oord: 
```bash
python main.py --network <model_name> --load_from <path_to_folder_contain_weight>
```
Example: 
```bash
python main.py --network rerein --load_from runs/re50_kitti_oord
```

Inference on mulran:
```bash
python mulran_main.py --load_from runs/re50_kitti_mulran
```
