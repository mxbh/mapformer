# MapFormer: Boosting Change Detection by Using Semantic Pre-change Information

This is the code used to generate the results in the paper. The repository is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and the [FHD repository](https://github.com/ZSVOS/FHD). Particularly, our main method components are implemented in `./mmseg/models/ccd/bc_heads.py`.

## Training
Before training, install the requirements, download the pre-trained backbone weights, and prepare the datasets (see below). Then run:
```
python tools/train.py ./configs/conditional_bcd/dynamicearthnet/mapformer.yaml \
  --work-dir=./runs/conditional_bcd/dynamicearthnet/mapformer
```
For other experiments, simply replace the paths for other configs (see `./configs`).

## Testing
To get the results on the test set, run:
```
python tools/test.py ./path/to/config.yaml \
  ./path/to/checkpoint.pth \
  --eval BC BC_precision BC_recall SC SCS mIoU \
  --samples-per-gpu=1
```

## Requirements
* Python >= 3.9
* See `requirements.txt`

## Data Preparation
### DynamicEarthNet
* Download DynamicEarthNet [here](https://mediatum.ub.tum.de/1650201).
* Structure the monthly data containing annotations in the following format:
```
├──./data/DynamicEarthNet/prepocessed/full/
│  ├──images/
│  │  ├──cube_id/
│  │  │  ├──2018-01-01.tif
│  │  │  ├──2018-02-01.tif
│  │  │  ...
│  │  │  └──2019-12-01.tif
│  │  ...
│  ├──labels/
│  │  ├──cube_id/
│  │  │  ├──2018-01-01.tif
│  │  │  ├──2018-02-01.tif
│  │  │  ...
│  │  │  └──2019-12-01.tif
│  │  ...
│  ├──splits/
│  │  │  ├──train35.txt
│  │  │  ├──val10.txt
│  │  │  └──test10.txt
```
* Use `./tools/convert_datasets/create_dynearthnet_tiles.py` to create tiles:
```
python ./tools/convert_datasets/create_dynearthnet_tiles.py \
  --data_dir=./data/DynamicEarthNet/prepocessed/ \
  --out_dir=./data/DynamicEarthNet/prepocessed/tiles500 \
  --tile_size=512 
```

### HRSCD
* Download HRSCD [here](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset).
* Structure the data in the following way:
```
├──./data/HRSCD/
│  ├──images/
│  │  ├──2006/
│  │  │  ├──D14/
│  │  │  │  ├──XXX.tif
│  │  │  │  ...
│  │  │  ├──D35/
│  │  │  │  ├──XXX.tif
│  │  │  │  ...
│  │  ├──2012/
│  │  │  ...
│  ├──labels/
│  │  ├──2006/
│  │  │  ├──D14/
│  │  │  │  ├──XXX.tif
│  │  │  │  ...
│  │  │  ├──D35/
│  │  │  │  ├──XXX.tif
│  │  │  │  │  ...
│  │  ├──2012/
│  │  │  ...
│  ├──change/
│  │  ├──D14/
│  │  │  ├──XXX.tif
│  │  │  ...
│  │  ├──D35/
│  │  │  ├──XXX.tif
│  │  │  ...
```
* Use `./tools/convert_datasets/create_hrscd_tiles.py` to create tiles:
```
python ./tools/convert_datasets/create_hrscd_tiles.py \
  --data_dir=./data/HRSCD/ \
  --out_dir=./data/HRSCD/preprocessed/tiles500 \
  --tile_size=500

# Used for faster inference, internally splitted into 500px tiles within the model.
python ./tools/convert_datasets/create_hrscd_tiles.py \
  --data_dir=./data/HRSCD/ \
  --out_dir=./data/HRSCD/preprocessed/tiles2000 \
  --tile_size=2000
```
* Manually put the provided split files in `./data/HRSCD/preprocessed/tilesXXX/splits/` (e.g., via symbolic links).

## Pre-trained Backbones
* Download MiT-b2 weights from [here](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia) (see [SegFormer repo](https://github.com/NVlabs/SegFormer#training)) and put them to `./model_ckpt/mit_b2.pth`.
* As the [open-cd codebase](https://github.com/likyoo/open-cd) uses the mmseg implementation of MiT, download the corresponding weights from [openmmlab](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth) as well and put them to `./model_ckpt/mit_b2_20220624-66e8bf70.pth`. 
