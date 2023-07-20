import os
import os.path as osp
from PIL import Image
import numpy as np
import argparse

Image.MAX_IMAGE_PIXELS = 10_000_000_000

parser = argparse.ArgumentParser(description='Create compressed tiles.')
parser.add_argument('--data_dir', type=str, help='Data dir')
parser.add_argument('--out_dir', type=str, help='Out dir')
parser.add_argument('--tile_size', type=int, help='Size of the final tiles')
parser.add_argument('--img_compression', type=str, default='jpeg')
parser.add_argument('--label_compression', type=str, default='tiff_lzw')
args = parser.parse_args()

print('Images...')
for year in ['2006', '2012']:
    for d in ['D14', 'D35']:
        for file in os.listdir(osp.join(args.data_dir, 'images', year, d)):
            name, suffix = file.split('.')
            os.makedirs(osp.join(args.out_dir, 'images', year, d, name))
            img = np.array(Image.open(osp.join(args.data_dir, 'images', year, d, file)))
            for i in range(img.shape[0] // args.tile_size):
                for j in range(img.shape[1] // args.tile_size):
                    img_tile = img[args.tile_size * i:args.tile_size * (i+1),
                        args.tile_size * j:args.tile_size * (j+1)]
                    
                    img_tile_path = osp.join(args.out_dir, 'images', year, d, name, f'{i}_{j}.'+suffix)
                    Image.fromarray(img_tile).save(img_tile_path, compression=args.img_compression)

print('Semantic labels...')
for year in ['2006', '2012']:
    for d in ['D14', 'D35']:
        for file in os.listdir(osp.join(args.data_dir, 'labels', year, d)):
            name, suffix = file.split('.')
            os.makedirs(osp.join(args.out_dir, 'labels', year, d, name))
            sem = np.array(Image.open(osp.join(args.data_dir, 'labels', year, d, file)))
            for i in range(sem.shape[0] // args.tile_size):
                for j in range(sem.shape[1] // args.tile_size):
                    sem_tile = sem[args.tile_size * i:args.tile_size * (i+1),
                        args.tile_size * j:args.tile_size * (j+1)]
                    
                    sem_tile_path = osp.join(args.out_dir, 'labels', year, d, name, f'{i}_{j}.'+suffix)
                    Image.fromarray(sem_tile).save(sem_tile_path, compression=args.label_compression)

print('Binary change labels...')
for d in ['D14', 'D35']:
    for file in os.listdir(osp.join(args.data_dir, 'labels', 'change', d)):
        name, suffix = file.split('.')
        os.makedirs(osp.join(args.out_dir, 'labels', 'change', d, name))
        bc = np.array(Image.open(osp.join(args.data_dir, 'labels', 'change', d, file)))
        for i in range(bc.shape[0] // args.tile_size):
            for j in range(bc.shape[1] // args.tile_size):
                bc_tile = bc[args.tile_size * i:args.tile_size * (i+1),
                    args.tile_size * j:args.tile_size * (j+1)]
                
                bc_tile_path = osp.join(args.out_dir, 'labels', 'change', d, name, f'{i}_{j}.'+suffix)
                Image.fromarray(bc_tile).save(bc_tile_path, compression=args.label_compression)
