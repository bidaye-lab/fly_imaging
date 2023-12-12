import numpy as np
from pathlib import Path

from tifffile import imwrite, imread, TiffFile
from scipy.io import loadmat


def fname(root_file, new_ending, new_root=''):


    # helper function: file name in parent directory
    f = Path(root_file)
    n = str(new_ending)

    fn  = f.parent / '{}_{}'.format(f.with_suffix('').name, n)

    if new_root:
        new_root = Path(new_root)
        fn = new_root / fn.relative_to(new_root.parent)
        fn.parent.mkdir(exist_ok=True, parents=True)

    return fn


def load_tiff(file, correct_offset=False, subtract_min=False):

    # load data
    img = imread(file)
    print(f'INFO loaded tiff stack from {file} with shape {img.shape}')

    # correct for scanimage offsets
    if correct_offset:
        tif = TiffFile(file)
        offsets = tif.scanimage_metadata['FrameData']['SI.hScan2D.channelOffsets']
        img[:, 0] -= offsets[0]
        img[:, 1] -= offsets[1]
        print(f'INFO added offsets: {offsets[0]} (channel 1) | {offsets[1]} (channel 2)')

    # assure that all intensities are positive
    if subtract_min:
        min1, min2 = img[:, 0].min(), img[:, 1].min()
        img[:, 0] -= min1
        img[:, 1] -= min2
        print(f'INFO subtracted minimum value: {min1} (channel 1) | {min2} (channel 2)')


    return img

def load_tiff_files(l_tif):

    # collect tiff files in list
    imgs = []

    for p_tif in l_tif:
        print(p_tif)
        img = load_tiff(p_tif)
        print(img.shape)
        imgs.append(img)

    # concatenate along first dimension
    stack = np.concatenate(imgs, axis=0)

    return stack

def write_tiff(file, arr):
    
    print(f'INFO writing images to {file}')
    imwrite(file, arr, photometric='minisblack')


def get_roi_zip_file(p_tif):

    g = p_tif.parent.glob('*RoiSet.zip')

    try:
        p_zip = next(g)
    except StopIteration:
        print(f'WARNING no *RoiSet.zip file found')
        return

    try:
        next(g)
        print(f'WARNING multiple *RoiSet.zip files found')
        return
    except StopIteration:
        return p_zip
    

def get_matlab_files(p_tif):

    p_ball = p_tif.parent / (p_tif.name.split('_')[0] + '.mat')
    if not p_ball.is_file():
        print('WARNING ball velocity matlab file not found')
        return

    g = p_tif.parent.glob('*-actions.mat')

    try:
        p_beh = next(g)
    except StopIteration:
        print(f'WARNING no *actions.mat file found')
        return
    
    try:
        next(g)
        print(f'WARNING multiple *actions.mat files found')
        return
    except StopIteration:
        return p_ball, p_beh
    

def load_behavior(p_mat, beh_keys):

    m = loadmat(p_mat, squeeze_me=True, struct_as_record=False)

    beh = { k: v for k, v in zip(m['behs'], m['bouts']) if k in beh_keys}

    return beh

def load_ball(p_mat):

    m = loadmat(p_mat, squeeze_me=True, struct_as_record=False)

    ball = vars(m['sensorData'])['bufferRotations']

    nans = np.all(np.isnan(ball), axis=1)
    ball = ball[~nans]

    return ball