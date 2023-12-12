
import numpy as np
import pandas as pd

# signal processing
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d


#############
# tiff stacks

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


#############
# stimulation

def get_onsets(arr, r_stim, r_rec, w=1, thresh=0.1, n_max=None):

    # convert to binary array based on thresh
    arr_bin = arr.copy()
    arr_bin[ arr_bin < thresh] = 0
    arr_bin[ arr_bin != 0 ] = 1
    
    # moving average and binary again
    arr_avg = uniform_filter1d(arr_bin, int(w*r_stim))
    arr_avg[ arr_avg != 0] = 1

    # indices for onset in arr_avg
    idx_avg = np.where(np.diff(arr_avg) == 1)[0]

    # onsets in arr
    idx = idx_avg + int(w*r_stim / 2)

    # convert from sample to s
    fs = idx / r_stim * r_rec
    f_max = len(arr) / r_stim * r_rec

    fs = np.round(fs, 0).astype(int)
    f_max = np.round(f_max, 0).astype(int)

    # limit number of stims
    fs = fs[:n_max]

    return fs, f_max

##########
# behavior
def resample(arr, sr, sr_new):

    l = len(arr)
    x = np.linspace(0, 1, l)

    l_new = int(l / sr * sr_new)
    x_new = np.linspace(0, 1, l_new)

    y_new = np.interp(x_new, x, arr)

    return y_new

def upsample_to_behavior(ca, beh, ball, f_ca, f_ball, f_beh, ca2=None):

    y = np.zeros_like(ca[0])
    y = resample(y, f_ca, f_beh)
    x = np.arange(len(y))

    df = pd.DataFrame(index=x)

    for i, y in enumerate(ca):
        y = resample(y, f_ca, f_beh)
        if ca2 is not None:
            y2 = resample(ca2[i], f_ca, f_beh)
            y /= y2
        df.loc[:, f'roi_{i+1}'] = y

    for j, y in zip('xyz', ball.T):
        y = resample(y, f_ball, f_beh)
        df.loc[:, f'ball_{j}'] = y[x]

    for k, v in beh.items():

        # boxcar
        y = np.zeros_like(x)

        arr = np.array(v, ndmin=2)
        if arr.any():
            for f_i, f_f, _ in arr:
                y[f_i:f_f+1] = 1

        df.loc[:, f'beh_{k}'] = y

        # delta 
        yi = np.zeros_like(x)
        yf = np.zeros_like(x)

        arr = np.array(v, ndmin=2)
        if arr.any():
            for f_i, f_f, _ in arr:
                yi[f_i:f_i+1] = 1
                yf[f_f:f_f+1] = 1

        df.loc[:, f'behi_{k}'] = yi
        df.loc[:, f'behf_{k}'] = yf

    return df

def align2events(df, beh, f, dt, use_z_roi=True):

    if use_z_roi:
        cols_roi = [ c for c in df.columns if c.startswith('z_roi_') ]
    else:
        cols_roi = [ c for c in df.columns if c.startswith('roi_') ]

    cols_ball = [ c for c in df.columns if c.startswith('ball_') ]
    cols = cols_roi + cols_ball
    
    dn = int(dt * f)
    s = np.arange(-dn, dn + 1)
    l = []
    n_good, n_bad = 0, 0

    for (fly, trial), d in df.groupby(['fly', 'trial']):
        ds = d.loc[:, beh]
        i_ons = np.flatnonzero(np.diff(ds) == 1) + 1
        idx_ons = ds.iloc[i_ons].index

        for idx in idx_ons:
            try: 
                for c in cols:
                    y = d.loc[idx - dn : idx + dn, c]
                    df_new = pd.DataFrame({
                        's': s,
                        't': s / f,
                        'fly': fly,
                        'trial': trial,
                        'y': y,
                        'match': c,
                        'onset': idx,
                    })
                    l.append(df_new)
                n_good += 1

            except ValueError:
                print(f'WARNING skipping fly {fly} trial {trial} event {idx}')
                n_bad += 1

    data = pd.concat(l)
    data.attrs['n_bad'] = n_bad
    data.attrs['n_good'] = n_good
    data.attrs['beh'] = beh
    data.attrs['n_fly'] = len(data.groupby(['fly']).groups.keys())
    data.attrs['n_trial'] = len(data.groupby(['fly', 'trial']).groups.keys())

    return data

def zscore_cols(df, col_start):

    cols = [ c for c in df.columns if c.startswith(col_start) ]
    cols_z = [ f'z_{c}' for c in cols ]

    df.loc[:, cols_z] = df.loc[:, cols].apply(zscore).values

    return df

def ca_kernel(tau_on, tau_off, f):

    x = np.arange(0, 10*tau_off, 1 / f)
    
    ca_trace = lambda a, b, c: - np.exp(-a / b) + np.exp(-a / c)
    y = ca_trace(x, tau_on, tau_off)

    i_max = np.argmax(y)
    y = np.pad(y, pad_width=(len(y) - 2 * i_max, 0))

    y = y / y.sum()

    return y

def convolute_ca_kernel(df, f):

    tau_on, tau_off = 0.13, 0.63 # https://elifesciences.org/articles/23496#s4

    kern = ca_kernel(tau_on, tau_off, f)

    cols = [ c for c in df.columns if c.startswith('ball_') or c.startswith('beh')]
    cols_new = [ f'conv_{c}' for c in cols ]
    df.loc[df.index, cols_new] = df.loc[df.index, cols].apply(np.convolve, axis=0, **{'v': kern, 'mode': 'same'}).values

    return df

def calculate_pearson(df, beh):

    c_roi = [ c for c in df.columns if c.startswith('z_roi_')]
    c_ball = [ c for c in df.columns if c.startswith('z_conv_ball_') ]
    c_beh = [ c for c in df.columns if c.startswith(f'conv_{beh}_')]
    cols = c_roi + c_ball + c_beh

    with np.errstate(divide='ignore', invalid='ignore'): # ignore division by 0 warning: TODO better check array before calling corrcoef
        c = np.corrcoef(df.loc[:, cols].T.values)
        # c = df.loc[:, cols].corr(method='pearson').values

    c[np.diag_indices(len(c))] = np.nan

    c_roi = [ c.replace('z_', '') for c in c_roi ] 
    c_ball = [ c.replace('z_conv_', '') for c in c_ball ]
    c_beh = [ c.replace(f'conv_{beh}_', '') for c in c_beh ]
    cols = c_roi + c_ball + c_beh

    d = pd.DataFrame(data=c, index=cols, columns=cols)

    return d

def calculate_ccf(df, dt, f, col1, col2, col2_):

    n = 2* dt * f + 1
    t = np.linspace(-dt, dt, n)

    cols_1 = [ c for c in df.columns if c.startswith(col1)]
    cols_2 = [ c for c in df.columns if c.startswith(col2) or c.startswith(col2_)]

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

def select_event(df, col, f, dt):

    dn = int(dt * f)

    l = []
    for k, d in df.groupby(['fly', 'trial']):
        ds = d.loc[:, col]
        i_ons = np.flatnonzero(np.diff(ds) == 1) + 1
        idx_ons = ds.iloc[i_ons].index
        for idx in idx_ons:
            try: 
                d_ = d.loc[idx - dn : idx + dn, :]
                l.append(d_)

            except ValueError:
                print(f'WARNING skipping fly {k[0]} trial {k[1]} event {idx}')

    df = pd.concat(l, ignore_index=True)

    # drop other behavior columns
    c_beh = [ c for c in df.columns if 'beh' in c]
    c_drop = [ c for c in c_beh if col.split('_')[-1] not in c]
    df = df.drop(columns=c_drop)

    return df