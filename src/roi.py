
import numpy as np
from scipy.ndimage import gaussian_filter, percentile_filter

# imageJ ROIs
from read_roi import read_roi_zip

# images
from PIL import Image, ImageDraw
from skimage.draw import polygon, ellipse
from skimage.color import gray2rgb
from matplotlib import colors


def read_imagej_rois(p_roi, arr):

    n_y, n_x = arr.shape[-2:]
    d_roi = read_roi_zip(p_roi)

    # all-false array with shape: (n_roi, n_y,  n_x)
    r = np.zeros((len(d_roi), n_y, n_x)).astype(bool)

    # set rois mask to true
    for i, v in enumerate(d_roi.values()):

        if v['type'] == 'polygon':
            x, y = polygon(v['x'], v['y'])

        elif v['type'] == 'oval':
            r_x = v['width'] / 2
            r_y = v['height'] / 2
            c_x = v['left'] + r_x
            c_y = v['top'] + r_y
            x, y = ellipse(c_x, c_y, r_x, r_y)

        else:
            print(f'WARNING skipping ROI {i+1}, because it has type {v["type"]} not implemented')
            continue
        
        # ignore out of bounds
        m = ( y < n_y ) * ( x < n_x )
        x, y = x[m], y[m]
        
        r[i, y, x] = True  

    return r

def draw_rois(img, rois):

    img = gray2rgb(img).astype('uint8')

    for i, r in enumerate(rois):
        rgb = tuple([int(f*255) for f in colors.to_rgb(f'C{i}')])
        img[r] = rgb

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i, r in enumerate(rois):
        r_ = gaussian_filter(r.astype(float), sigma=1)
        x, y = np.unravel_index(r_.argmax(), r_.shape)
        draw.text((y, x), str(i + 1))

    img = np.array(img)

    return img


def get_mean_trace(rois, arr):

    # array with shape: (n_rois, n_samples)
    arr_m = np.zeros((len(rois), len(arr)))

    for i, r in enumerate(rois):
        arr_m[i] = np.mean(arr[:, r], axis=1)

    return arr_m

def subtract_background(arr):

    a = arr[:-1] # all but last ROI
    b = arr[-1] # last
    # b = gaussian_filter(arr[-1], sigma=2) # this would smooth background ROI before subtraction
    arr = a - b # subtract last 

    return arr

def subtract_baseline(arr, percentile, size):
        
    x = arr
    x0 = percentile_filter(x, percentile=percentile, size=(1, size))
    dxx = (x - x0) / x

    return dxx