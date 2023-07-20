import os
import os.path as osp
from PIL import Image
import argparse
import numpy as np
import tifffile

Image.MAX_IMAGE_PIXELS = 10_000_000_000


parser = argparse.ArgumentParser(description='Create compressed tiles.')
parser.add_argument('--data_dir', type=str, help='Data dir')
parser.add_argument('--out_dir', type=str, help='Out dir')
parser.add_argument('--tile_size', type=int, help='Size of tiles')
parser.add_argument('--img_compression', type=str, default='DEFLATE')
parser.add_argument('--label_compression', type=str, default='LZW')
args = parser.parse_args()

tile_size = args.tile_size
img_size = 1024
n_tiles = img_size // tile_size
root_dir = args.data_dir
full_img_dir = osp.join(root_dir, 'full', 'images')
full_lbl_dir = osp.join(root_dir, 'full', 'labels')
full_split_dir = osp.join(root_dir, 'full', 'splits')
tile_img_dir = osp.join(args.out_dir, 'images')
tile_lbl_dir = osp.join(args.out_dir, 'labels')
tile_split_dir = osp.join(args.out_dir, 'splits')

# images
print('images...')
for site in os.listdir(full_img_dir):
    print(site + '...')
    for i in range(n_tiles):
        for j in range(n_tiles):
            os.makedirs(osp.join(tile_img_dir, site+f'_{i}_{j}'))
    for file in os.listdir(osp.join(full_img_dir, site)):
        full_img = tifffile.imread(osp.join(full_img_dir, site, file))
        for i in range(n_tiles):
            for j in range(n_tiles):
                tile = full_img[tile_size*i:tile_size*(i+1),
                                tile_size*j:tile_size*(j+1),
                                :]
                tile_path = osp.join(tile_img_dir, site+f'_{i}_{j}', file)
                tifffile.imwrite(tile_path, tile, compression=args.img_compression)

# labels            
print('labels...')
for site in os.listdir(full_lbl_dir):
    print(site + '...')
    for i in range(n_tiles):
        for j in range(n_tiles):
            os.makedirs(osp.join(tile_lbl_dir, site+f'_{i}_{j}'))
    for file in os.listdir(osp.join(full_lbl_dir, site)):
        full_lbl = tifffile.imread(osp.join(full_lbl_dir, site, file))
        for i in range(n_tiles):
            for j in range(n_tiles):
                tile = full_lbl[tile_size*i:tile_size*(i+1),
                                tile_size*j:tile_size*(j+1),
                                :]
                tile_path = osp.join(tile_lbl_dir, site+f'_{i}_{j}', file)
                tifffile.imwrite(tile_path, tile, compression=args.label_compression)

# splits
print('splits...')
os.makedirs(tile_split_dir)
for file in os.listdir(full_split_dir):
    with open(osp.join(full_split_dir, file), 'r') as f:
        sites = [s.strip() for s in f.readlines()]
    tile_sites = []
    for site in sites:
        for i in range(n_tiles):
            for j in range(n_tiles):
                tile_sites.append(site+f'_{i}_{j}')
    with open(osp.join(tile_split_dir, file), 'w') as f:
        f.writelines([t + '\n' for t in tile_sites])
print('Done.')

    
