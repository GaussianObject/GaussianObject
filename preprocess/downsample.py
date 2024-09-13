import os
import argparse
from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', type=str, default='data/realcap/rabbit')
    parser.add_argument('--sparse_num', type=int, default=4)
    args = parser.parse_args()

    images_path = os.path.join(args.source_path, 'images')

    factors = (2, 4, 8)
    for factor in factors:
        images_path_resize = f'{images_path}_{factor}'
        if not os.path.exists(images_path_resize):
            os.mkdir(images_path_resize)
        for image_name in tqdm(sorted(os.listdir(images_path))):
            image = Image.open(os.path.join(images_path, image_name))
            orig_w, orig_h = image.size[0], image.size[1]
            resolution = round(orig_w / factor), round(orig_h / factor)
            image = image.resize(resolution)
            image.save(os.path.join(images_path_resize, image_name))
