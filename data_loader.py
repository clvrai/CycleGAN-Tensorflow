import os
from glob import glob

from scipy.misc import imread, imresize
import numpy as np
from tqdm import tqdm
import h5py

datasets = ['ae_photos', 'apple2orange', 'summer2winter_yosemite', 'horse2zebra',
            'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo',
            'maps', 'cityscapes', 'facades', 'iphone2dslr_flower']

def read_image(path):
    image = imread(path)
    if len(image.shape) != 3 or image.shape[2] != 3:
        print('Wrong image {} with shape {}'.format(path, image.shape))
        return None

    # range of pixel values = [-1.0, 1.0]
    image = image.astype(np.float32) / 255.0
    image = image * 2.0 - 1.0
    return image

def read_images(base_dir):
    ret = []
    for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
        data_dir = os.path.join(base_dir, dir_name)
        paths = glob(os.path.join(data_dir, '*.jpg'))
        print('# images in {}: {}'.format(data_dir, len(paths)))

        images = []
        for path in tqdm(paths):
            image = read_image(path)
            if image is not None:
                images.append(image)
        ret.append((dir_name, images))
    return ret

def store_h5py(base_dir, dir_name, images, image_size):
    f = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'w')
    for i in range(len(images)):
        grp = f.create_group(str(i))
        if images[i].shape[0] != image_size:
            image = imresize(images[i], (image_size, image_size, 3))
            # range of pixel values = [-1.0, 1.0]
            image = image.astype(np.float32) / 255.0
            image = image * 2.0 - 1.0
            grp['image'] = image
        else:
            grp['image'] = images[i]
    f.close()

def convert_h5py(task_name):
    print('Generating h5py file')
    base_dir = os.path.join('datasets', task_name)
    data = read_images(base_dir)
    for dir_name, images in data:
        if images[0].shape[0] == 256:
            store_h5py(base_dir, dir_name, images, 256)
        store_h5py(base_dir, dir_name, images, 128)

def read_h5py(task_name, image_size):
    base_dir = 'datasets/' + task_name
    paths = glob(os.path.join(base_dir, '*_{}.hy'.format(image_size)))
    if len(paths) != 4:
        convert_h5py(task_name)
    ret = []
    for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
        try:
            dataset = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'r')
        except:
            raise IOError('Dataset is not available. Please try it again')

        images = []
        for id in dataset:
            images.append(dataset[id]['image'].value.astype(np.float32))
        ret.append(images)
    return ret

def download_dataset(task_name):
    print('Download data %s' % task_name)
    cmd = './download_cyclegan_dataset.sh ' +  task_name
    os.system(cmd)

def get_data(task_name, image_size):
    assert task_name in datasets, 'Dataset {}_{} is not available'.format(
        task_name, image_size)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    base_dir = os.path.join('datasets', task_name)
    print('Check data %s' % base_dir)
    if not os.path.exists(base_dir):
        print('Dataset not found. Start downloading...')
        download_dataset(task_name)
        convert_h5py(task_name)

    print('Load data %s' % task_name)
    train_A, train_B, test_A, test_B = \
        read_h5py(task_name, image_size)
    return train_A, train_B, test_A, test_B
