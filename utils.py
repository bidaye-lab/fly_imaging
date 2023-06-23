import numpy as np
import pandas as pd


from scipy.ndimage import gaussian_filter
from tifffile import imwrite, imread
from pystackreg import StackReg
from read_roi import read_roi_zip
from skimage.draw import polygon, ellipse
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore


from joblib import Parallel, delayed, parallel_backend

import matplotlib.pylab as plt
import seaborn as sns
from skvideo.io import vwrite
from PIL import Image, ImageDraw
from skimage.color import gray2rgb
from matplotlib import colors



###############
# file handling 

def load_tiff(file):

    img = imread(file)
    print(f'INFO loaded tiff stack from {file} with shape {img.shape}')

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

def write_tif(file, arr):
    
    print(f'INFO writing images to {file}')
    imwrite(file, arr, photometric='minisblack')

###############
# data handling

def split_channels(stack, n_z, n_ch):

    # order is reversed in stack
    n_y, n_x = stack.shape[-2:]

    # reshape array channel, z, time, x, y
    arr = np.reshape(stack.T, (n_x, n_y, n_ch, n_z, -1), order='F')
    print(f'INFO reshaped transposed stack to {arr.shape}')

    # separate channels
    ch1, ch2 = arr[:, :, 0], arr[:, :, 1]
    ch1, ch2 = ch1.T, ch2.T
    print(f'INFO split into 2 channels and tranposed back: shape {ch1.shape}')

    return ch1, ch2

def maxproj_z(arr):

    # collapse z: max procection
    arr = np.max(arr, axis=1)
    print(f'INFO max proj along z dim: shape {arr.shape}')
    
    return arr

def smooth_xy(arr, sigma):
    
    # smooth only along x and y direction
    arr_smth = gaussian_filter(arr, sigma=(0, sigma, sigma))

    return arr_smth


##############
# registration

def get_tmats(arr, reg, n_bin=1):
    
    # bin time series
    n_y, n_x = arr.shape[-2:]
    arr_b = np.mean(np.reshape(arr, (-1, n_bin, n_y, n_x)), axis=1)

    # get mean image
    ref = np.mean(arr_b, axis=0)

    print(f'INFO getting transformation matrices using n_bin = {n_bin} and registration {reg}')

    with parallel_backend('loky', n_jobs=-1):

        tmats_b = Parallel()(
            delayed(StackReg(reg).register)(ref, img) for img in arr_b
            )

    # non-binned tmats
    tmats = [ i for i in tmats_b for _ in range(n_bin) ]

    return tmats

def align(arr, tmats, reg):

    print(f'INFO aligning array using tranformation matrices and registration {reg}')

    with parallel_backend('loky', n_jobs=-1):

        arr_a = Parallel()(
            delayed(StackReg(reg).transform)(i, j) for i, j in zip(arr, tmats)
        )

    arr_a = np.array(arr_a)

    # replace 0 by mean
    arr_a[arr_a == 0] = np.mean(arr_a)
        
    return arr_a

#####
# ROI

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

def get_mean_trace(rois, arr, subtract_background=False, sigma=0):

    # array with shape: (n_rois, n_samples)
    arr_m = np.zeros((len(rois), len(arr)))

    for i, r in enumerate(rois):
        arr_m[i] = np.mean(arr[:, r], axis=1)
    
    # subtract last ROI
    if subtract_background:
        r = arr_m[:-1]
        b = gaussian_filter(arr_m[-1], sigma)
        arr_m = r - b

    return arr_m

##########
# behavior
def load_behavior(p_mat, beh_keys=['pushing', 'hindLegRub', 'frontLegRub', 'headGrooming', 'abdomenGrooming', 'ObL1+R1']):

    m = loadmat(p_mat, squeeze_me=True, struct_as_record=False)

    beh = { k: v for k, v in zip(m['behs'], m['bouts']) if k in beh_keys}

    return beh

def load_ball(p_mat):

    m = loadmat(p_mat, squeeze_me=True, struct_as_record=False)

    ball = vars(m['sensorData'])['bufferRotations']

    nans = np.all(np.isnan(ball), axis=1)
    ball = ball[~nans]

    return ball

def resample(arr, sr, sr_new):

    l = len(arr)
    x = np.linspace(0, 1, l)

    l_new = int(l / sr * sr_new)
    x_new = np.linspace(0, 1, l_new)

    y_new = np.interp(x_new, x, arr)

    return y_new

def upsample_to_behavior(ca, beh, ball, f_ca, f_ball, f_beh):

    y = np.zeros_like(ca[0])
    y = resample(y, f_ca, f_beh)
    x = np.arange(len(y))


    df = pd.DataFrame(index=x)
    # df.loc[:, 't'] = df.index / f_beh

    for i, y in enumerate(ca):
        y = resample(y, f_ca, f_beh)
        # y = zscore(y)
        df.loc[:, f'roi_{i+1}'] = y

    for j, y in zip('xyz', ball.T):
        y = resample(y, f_ball, f_beh)
        df.loc[:, f'ball_{j}'] = y[x]

    for k, v in beh.items():

        y = np.zeros_like(x)

        arr = np.array(v, ndmin=2)
        if arr.any():
            for f_i, f_f, _ in arr:
                y[f_i:f_f+1] = 1

        df.loc[:, f'beh_{k}'] = y

    return df

def zscore_rois(df):

    cols = [ c for c in df.columns if c.startswith('roi_') ]
    cols_z = [ f'z_{c}' for c in cols ]

    df.loc[:, cols_z] = df.loc[:, cols].apply(zscore).values

    return df

def ca_kernel(tau_on, tau_off, f):

    x = np.arange(0, 10*tau_off, 1 / f)
    
    ca_trace = lambda a, b, c: - np.exp(-a / b) + np.exp(-a / c)
    y = ca_trace(x, tau_on, tau_off)

    i_max = np.argmax(y)
    y = np.pad(y, pad_width=(len(y) - i_max, 0))

    y = y / y.sum()

    return y

def convolute_ca_kernel(df, f):

    tau_on, tau_off = 0.13, 0.63 # https://elifesciences.org/articles/23496#s4

    kern = ca_kernel(tau_on, tau_off, f)

    cols = [ c for c in df.columns if c.startswith('ball_') or c.startswith('beh_')]
    cols_new = [ f'conv_{c}' for c in cols ]
    df.loc[df.index, cols_new] = df.loc[df.index, cols].apply(np.convolve, axis=0, **{'v': kern, 'mode': 'same'}).values

    return df

def calculate_pearson(df):

    c_roi = [ c for c in df.columns if c.startswith('z_roi_')]
    c_ball = [ c for c in df.columns if c.startswith('conv_ball_') ]
    c_beh = [ c for c in df.columns if c.startswith('conv_beh_')]
    cols = c_roi + c_ball + c_beh

    with np.errstate(divide='ignore', invalid='ignore'): # ignore division by 0 warning: TODO better check array before calling corrcoef
        c = np.corrcoef(df.loc[:, cols].T)

    c[np.diag_indices(len(c))] = np.nan

    c_roi = [ c.replace('z_', '') for c in c_roi ] 
    c_ball = [ c.replace('conv_', '') for c in c_ball ]
    c_beh = [ c.replace('conv_beh_', '') for c in c_beh ]
    cols = c_roi + c_ball + c_beh

    d = pd.DataFrame(data=c, index=cols, columns=cols)

    return d

def calculate_ccf(df, dt, f):

    n = 2* dt * f + 1
    t = np.linspace(-dt, dt, n)

    cols_1 = [ c for c in df.columns if c.startswith('roi_')]
    cols_2 = [ c for c in df.columns if c.startswith('conv_ball_') or c.startswith('conv_beh_')]

    l = []
    for k, df_k in df.groupby(['cond', 'fly', 'trial']):
        for c1 in cols_1:
            for c2 in cols_2:

                y1 = df_k.loc[:, c1].values
                y2 = df_k.loc[:, c2].values
                
                a = y1
                b = np.pad(y2, dt * f)
                c = np.correlate(a, b, mode='valid')

                # equivalent to matlab norm
                norm = np.sqrt( np.sum(a**2) * np.sum(b**2))
                with np.errstate(divide='ignore', invalid='ignore'): # ignore division by 0 warning: TODO better check array before calling corrcoef
                    c = c / norm

                # c = c / len(a)

                cond, fly, trial = k
                d = pd.DataFrame(data={
                    't': t,
                    'ccf': c,
                    'r': c1,
                    'b': c2,
                    'cond': cond,
                    'fly': fly,
                    'trial': trial,
                })

                l.append(d)

    df_c = pd.concat(l, ignore_index=True)
    
    return df_c




###############
# visualization

def save_dual_movie(file, arr1, arr2, cmap='viridis', fps=30):

    arr = np.concatenate([arr1, arr2], axis=-1) # along x dim

    cm = plt.get_cmap(cmap)
    arr = cm(arr.astype(int)) * 255

    print(f'INFO writing video to file {file}')
    vwrite(file, arr, inputdict={'-framerate': str(fps)})


def save_img(file, img):

    img = img.astype('float')
    img -= img.min()
    img /= img.max()
    img = (img * 255).astype('uint8')

    img = Image.fromarray(img)
    print(f'INFO saving normalized image to {file}')
    img.save(file)


def plot_data(df, f, path=''):

    c_roi = [ c for c in df.columns if c.startswith('roi_')]
    c_ball = [ c for c in df.columns if c.startswith('ball_')]
    c_beh = [ c for c in df.columns if c.startswith('beh_')]

    cols = c_roi + c_ball + c_beh

    # n = len(c_roi) + len(c_ball) + c_beh
    n = len(cols)
    fig, axarr = plt.subplots(nrows=n, figsize=(20, 2.5*n))
    x = df.index / f


    for ax, c in zip(axarr, cols):
        

        ax2 = ax.twinx()
        for i, c2 in enumerate(c_beh):
            y = df.loc[:, c2].values
            mask = y == 1
            offset = 0.1 * i
            y = y -  offset
            color = f'C{i}'
            ax2.scatter(x[mask], y[mask], color=color, marker='|', label=c2.replace('beh_', ''))
            ax2.axhline(1 - offset, color=color, lw=0.5, ls=':')
        ax2.set_ylim(-2, 1.1)
        ax2.set_yticklabels([])
        if c == c_roi[0]:
            ax2.legend()


        if c.startswith('ball') or c.startswith('beh'):
            y = df.loc[:, c]
            y_c = df.loc[:, f'conv_{c}']
            y = y / y.max() * y_c.max()
            ax.plot(x, y, color='gray', ls='--', lw=1)
            ax.plot(x, y_c)

        elif c.startswith('roi'):
            y = df.loc[:, f'z_{c}'].values
            ax.plot(x, y)
        

        ax.set_title(c)
        ax.set_xlabel('time [s]')
        ax.margins(x=0)

    fig.tight_layout()

    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_corr_heatmap(df, path=''):

    d = calculate_pearson(df)

    fig, ax = plt.subplots()

    sns.heatmap(ax=ax, data=d, square=True, cmap='coolwarm', center=0)

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)

def plot_ccf(df, f, pool_fly=True, path=''):

    d = calculate_ccf(df, dt=2, f=f)
    d = d.dropna(axis=0)


    kw_args = {
        'x': 't',
        'y': 'ccf',
        'col': 'r', 
        'hue': 'b',
    }
    if not pool_fly:
        kw_args['hue'] = 'fly'
        kw_args['style'] = 'trial'
        kw_args['row'] = 'b'

    g = sns.relplot(data=d, kind='line', facet_kws={'sharey': False}, errorbar='se', **kw_args)
    g.set_axis_labels('time lag [s]', 'norm CCF')
    
    fig = g.fig
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)