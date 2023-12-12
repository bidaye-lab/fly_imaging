# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: everything
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
from pathlib import Path
from pystackreg import StackReg
from scipy.ndimage import gaussian_filter
import pickle

import utils as utl

# %% [markdown]
# # User settings

# %%
# number of channels and z slices
n_ch, n_z = 2, 15

# smoothing in xy before registration (sigma 2D Gaussian)
xy_smth = 4

# settings for background subtraction: percentile and window size (winsize / f_ca = window in seconds)
perc, winsize = 10, 50

# frequencies used for imaging, ball velocities, and behavior
f_ca, f_ball, f_beh = 2, 50, 200

# Behaviors
beh_keys = ['pushing', 'hindLegRub', 'frontLegRub', 'headGrooming', 'abdomenGrooming', 'PER', 'midLeft+hindLegRub', 'midRight+hindLegRub'] # 'ObL1+R1'

# transformation used for registration https://pystackreg.readthedocs.io/en/latest/readme.html#usage
reg = StackReg.SCALED_ROTATION 

# path to folder
parent_dir = Path(r'Y:\Nino_2P_for_Salil\for_Nico\stop1_filteredData_2\stop1-GCaMP6f-tdTomato_VNC_smth4')

# selection rule for tif files
p_tifs = [ *parent_dir.glob('**/trials_to_register/*/trial*_00???.tif') ]

# force overwriting files
overwrite = True

# folder to save output
p_out = parent_dir / 'smth4_perc10_winsize50'
p_out.mkdir(exist_ok=True)

# %% [markdown]
# # general processing pipeline
# ## Step 1: Registration

# %%
for p_tif in p_tifs:
    print()

    # define plot folder
    p_plot = utl.fname(p_tif, '', new_root=p_out).parent / 'plots'
    p_plot.mkdir(exist_ok=True)

    if utl.fname(p_tif, 'ch1.tif', new_root=p_out).is_file() and not overwrite:
        print(f'INFO output file exists, skipping registration for {p_tif.parent}')
        continue
    else:
        print(f'INFO now registering {p_tif}')

    # load and split
    stack = utl.load_tiff(p_tif)
    ch1, ch2 = utl.split_channels(stack, n_z=15, n_ch=2)
    ch1 = utl.maxproj_z(ch1)
    ch2 = utl.maxproj_z(ch2)

    ch1 = utl.smooth_xy(ch1, xy_smth)
    ch2 = utl.smooth_xy(ch2, xy_smth)

    # register
    tmats = utl.get_tmats(ch2, reg)
    ch1_a = utl.align(ch1, tmats, reg)
    ch2_a = utl.align(ch2, tmats, reg)

    # mean image
    ch1_am = np.mean(ch1_a, axis=0)
    ch2_am = np.mean(ch2_a, axis=0)

    # save to disk
    utl.write_tif(utl.fname(p_tif, 'ch1.tif', new_root=p_out), ch1_a.astype('int16'))
    utl.write_tif(utl.fname(p_tif, 'ch2.tif', new_root=p_out), ch2_a.astype('int16'))

    utl.write_tif(utl.fname(p_tif, 'ch1reg.tif', new_root=p_out), ch1_a.astype('int16'))
    utl.write_tif(utl.fname(p_tif, 'ch2reg.tif', new_root=p_out), ch2_a.astype('int16'))

    utl.save_img(p_plot / 'ch1mean.bmp', ch1_am)
    utl.save_img(p_plot / 'ch2mean.bmp', ch2_am)

    utl.save_dual_movie(p_plot / 'ch1ch2.mp4', ch1, ch2)
    utl.save_dual_movie(p_plot / 'ch1reg.mp4', ch1, ch1_a)
    utl.save_dual_movie(p_plot / 'ch2reg.mp4', ch2, ch2_a)
    


# %% [markdown]
# ## Step 2: ROI extraction

# %%
for p_tif in p_tifs:
    print()

    # define plot folder
    p_plot = utl.fname(p_tif, '', new_root=p_out).parent / 'plots'
    p_plot.mkdir(exist_ok=True)
    
    # check if ROI traces have already been extracted
    p_roi = utl.fname(p_tif, 'roi_traces.pickle', new_root=p_out)
    if p_roi.is_file() and not overwrite:
        print(f'INFO output files exists, skipping ROI extraction for {p_tif.parent}')
        continue

    # load Roi.zip
    p_zip = utl.get_roi_zip_file(p_tif)
    if not p_zip:
        print(f'WARNING Skipping {p_tif.parent}')
        continue
    else:
        print(f'INFO loading ROIs from {p_zip}')

    # load aligned data
    ch1_a = utl.load_tiff(utl.fname(p_tif, 'ch1reg.tif', new_root=p_out))
    ch2_a = utl.load_tiff(utl.fname(p_tif, 'ch2reg.tif', new_root=p_out))

    # load ROIs
    img = np.mean(ch1_a, axis=0)
    rois = utl.read_imagej_rois(p_zip, img)
    img_rois = utl.draw_rois(img, rois)
    utl.save_img(p_plot / 'ch1mean_rois.bmp', img_rois)

    # extract traces
    d_roi = dict() # collect traces extracted with different methods here

    ca1 = utl.get_mean_trace(rois, ch1_a)
    ca2 = utl.get_mean_trace(rois, ch2_a)
    d_roi['ch1raw'] = ca1
    d_roi['ch2raw'] = ca2

    # channel 1 to 2 ratio
    r12 = ca1 / ca2
    d_roi['r12'] = r12

    # subtract baseline
    dca1 = utl.subtract_baseline(ca1, percentile=perc, size=winsize)
    dca2 = utl.subtract_baseline(ca2, percentile=perc, size=winsize)
    dr12 = utl.subtract_baseline(r12, percentile=perc, size=winsize)
    d_roi['dch1'] = dca1
    d_roi['dch2'] = dca2
    d_roi['dr12'] = dr12

    # subract background ROI
    ca1 = utl.subtract_background(ca1)
    ca2 = utl.subtract_background(ca2)
    d_roi['ch1'] = ca1
    d_roi['ch2'] = ca2  

    # save to disk
    with open(p_roi, 'wb') as f:
        pickle.dump(d_roi, f)
    print(f'INFO saving ROI traces to {p_roi}')

# %% [markdown]
# ## Step 3: Combine imaging and behavior data

# %%
for p_tif in p_tifs:
    print()

    # define plot folder
    p_plot = utl.fname(p_tif, '', new_root=p_out).parent / 'plots'
    p_plot.mkdir(exist_ok=True)

    # load ROI traces
    p_roi = utl.fname(p_tif, 'roi_traces.pickle', new_root=p_out)
    if not p_roi.is_file():
        print(f'WARNING file with ROI traces not found, skipping {p_tif.parent}')
        continue
    else:
        with open(p_roi, 'rb') as f:
            d_roi = pickle.load(f)

    # load behavior data and ball velocities
    p_mats = utl.get_matlab_files(p_tif)
    if not p_mats:
        print(f'WARNING skipping {p_tif.parent}')
        continue
    else:
        p_ball, p_beh =  p_mats
        
    ball = utl.load_ball(p_ball)
    beh = utl.load_behavior(p_beh, beh_keys)
    
    for method, traces in d_roi.items():

        # check if already been processed
        p_df = utl.fname(p_tif, f'data_{method}.parquet', new_root=p_out)
        if p_df.is_file() and not overwrite:
            print(f'INFO output files exists, skipping data merging for method {method} in {p_tif.parent}')
            continue

        # match sample rates
        df = utl.upsample_to_behavior(traces, beh, ball, f_ca, f_ball, f_beh)
        # zscore ROIs
        df = utl.zscore_cols(df, col_start='roi_')
        # convolute ball velocities and behavior with Ca kernel
        df = utl.convolute_ca_kernel(df, f=f_beh)
        # zscore ball velocities
        df = utl.zscore_cols(df, col_start='conv_ball_')

        # add additional data based on file and folder names
        pt = p_tif.parts
        cond, fly, trial = pt[-5], pt[-4], pt[-2]
        df.loc[:, 'cond'] = cond # e.g. fed/starved
        df.loc[:, 'fly'] = fly # fly number
        df.loc[:, 'trial'] = trial # trial number
        print(f'INFO parsing folder names: fly {fly} | trial {trial} | condition {cond}')

        # plots for quality control
        utl.plot_data(df, f_beh, path=p_plot / f'data_{method}.png')
        # pearson r heatmap
        utl.plot_corr_heatmap(df, beh='behi', path=p_plot / f'heatmap_{method}.png')
        # ccf
        utl.plot_ccf(df, f=f_beh, pool_fly=True, path=p_plot / f'ccf_{method}.png')

        # save to disk
        print(f'INFO writing merged data to {p_df}')
        df.to_parquet(p_df)

        # optional (will be big files): save also as CSV
        # df.to_csv(p_df.with_suffix('.csv')) 

# %% [markdown]
# ## Step 4: merge all trials

# %%
# merge all trials and flies

# get methods
all_pars = []
for p in p_tifs:
    l = [ *utl.fname(p, '', new_root=p_out).parent.glob('*data_*.parquet') ]
    all_pars.extend(l)
methods = { p.stem.split('_')[-1] for p in all_pars }

for method in methods:
    # list of all *_data_{method}.parquet files
    p_pars = [ utl.fname(p, f'data_{method}.parquet', new_root=p_out) for p in p_tifs ]

    l = []
    for p_par in p_pars:
        if not p_par.is_file():
            print(f'WARNING skipping {p_par.parent}')
            continue
        else:
            print(f'INFO loading file {p_par}')
            df = pd.read_parquet(p_par)
            l.append(df)

    if l:
        # combine dataframes and save
        df = pd.concat(l, ignore_index=True)
        p_df = p_out / f'all_data_{method}.parquet'
        df.to_parquet(p_df)

        print(f'INFO contents of {p_df}')
        for f, d in df.groupby('fly'):
            print(f'     {f}', end=': ')
            for t, _ in d.groupby('trial'):
                print(f'{t}', end=' ')
            print()
    else:
        # this check should not be necessary (?)
        print(f'WARNING no data files found, skipping {method}')

# %% [markdown]
# # Analysis

# %% [markdown]
# ## All recordings

# %%
# merge all trials and flies
for p_df in p_out.glob('all_data_*.parquet'):
    method = p_df.stem.split('_')[-1]

    # create plot folder
    p_plot = p_out / method
    p_plot.mkdir(exist_ok=True)

    # read data from disk
    df = pd.read_parquet(p_df)
    df = df.fillna(0) # TODO workaround because of missing behavior

    # optional: remove ROI 7, 8, 9
    df = df.drop(columns=['z_roi_7', 'z_roi_8', 'z_roi_9'])

    # pearson correlation heatmap (selection of columns: see utl.calculate_pearson)
    utl.plot_corr_heatmap(df, beh='behi', path=p_plot / 'heatmap.svg')

    # pearson heatmaps around behavior events
    dt = 5 # time in s to keep before and after behavior event

    # loop through all behavior
    cols = [ c for c in df.columns if c.startswith('beh_') ]
    for col in cols:

        # select df around behavoir
        d = utl.select_event(df, col, f_beh, dt)

        # generate plot
        utl.plot_corr_heatmap(d, beh='behi', path=p_plot / f'heatmap_{col}.svg')

    # cross-correlation functions behavior/ball and ROIs (averaged)
    utl.plot_ccf(df, f=f_beh, pool_fly=True,  path=p_plot / 'ccf.svg')

    # same (not averaged)
    utl.plot_ccf(df, f=f_beh, pool_fly=False, path=p_plot / 'ccf_indv.svg')

    # plot aligned data
    dt = 5 # time in s before and after behavior event
    s = 0.25 # smoothing window for velocity [in s] 

    # smooth velocity
    df_ = df.copy()
    df_.loc[:, ['ball_x', 'ball_y', 'ball_z']] = gaussian_filter(df_.loc[:, ['ball_x', 'ball_y', 'ball_z']].values, (s * f_beh, 0))

    # cycle through all behavoirs
    cols = [c for c in df_.columns if c.startswith('beh_')]
    for col in cols:

        # align to behavior
        df_al = utl.align2events(df_, col, f_beh, dt)

        utl.plot_aligned(df_al, path=p_plot / f'aligned_to_{col}.svg')

# %% [markdown]
# # spatial correlation maps (long)

# %%
# define tif file for session of interest
for p_tif in p_tifs:

    # define and create output folder
    p_plot = utl.fname(p_tif, '', new_root=p_out).parent / 'corrmaps'
    p_plot.mkdir(exist_ok=True)
    print(p_plot)

    # load registered tif files
    ch1 = utl.load_tiff(utl.fname(p_tif, 'ch1reg.tif', new_root=p_out))
    ch2 = utl.load_tiff(utl.fname(p_tif, 'ch2reg.tif', new_root=p_out))

    # load preprocessed behavior data
    df = pd.read_parquet(utl.fname(p_tif, 'data_ch1.parquet', new_root=p_out)) #either of the data files work

    # loop through all conv behi and conv ball columns
    cols = [ c for c in df.columns if c.startswith('conv_behi') or c.startswith('conv_ball') ]
    for col in cols:
        utl.plot_corrmap(ch1, ch2, df, col, f_ca=f_ca, f_beh=f_beh, path=p_plot / f'{col}_1xy.svg')

# %% [markdown]
# ## pooled

# %%
# path to folder
ps = [ *parent_dir.glob('female13/trials_to_register/*/*0000?.tif') ]

p_plot = p_out / 'pooled_corrmap'
p_plot.mkdir(exist_ok=True)

# combine tifs
l1, l2 = [], []
for p_tif in ps:
    ch1 = utl.load_tiff(utl.fname(p_tif, 'ch1.tif', new_root=p_out))
    ch2 = utl.load_tiff(utl.fname(p_tif, 'ch2.tif', new_root=p_out))
    l1.append(ch1)
    l2.append(ch2)

ch1 = np.concatenate(l1, axis=0)
ch2 = np.concatenate(l2, axis=0)

# register
tmats = utl.get_tmats(ch2, reg)
ch1_a = utl.align(ch1, tmats, reg)
ch2_a = utl.align(ch2, tmats, reg)

# mean image
ch1_am = np.mean(ch1_a, axis=0)
ch2_am = np.mean(ch2_a, axis=0)

# save to disk
utl.write_tif(p_plot / 'ch1.tif', ch1_a.astype('int16'))
utl.write_tif(p_plot / 'ch2.tif', ch2_a.astype('int16'))

utl.write_tif(p_plot / 'ch1reg.tif', ch1_a.astype('int16'))
utl.write_tif(p_plot / 'ch2reg.tif', ch2_a.astype('int16'))

utl.save_img(p_plot / 'ch1mean.bmp', ch1_am)
utl.save_img(p_plot / 'ch2mean.bmp', ch2_am)

utl.save_dual_movie(p_plot / 'ch1ch2.mp4', ch1, ch2)
utl.save_dual_movie(p_plot / 'ch1reg.mp4', ch1, ch1_a)
utl.save_dual_movie(p_plot / 'ch2reg.mp4', ch2, ch2_a)

# load preprocessed behavior data
l = []
for p_tif in ps:
    df = pd.read_parquet(utl.fname(p_tif, 'data_ch1.parquet', new_root=p_out)) #either of the data works
    l.append(df)
df = pd.concat(l, ignore_index=True)

# loop through all conv behi and conv ball columns
cols = [ c for c in df.columns if c.startswith('conv_behi') or c.startswith('conv_ball') ]
for col in cols:
    utl.plot_corrmap(ch1, ch2, df, col, f_ca=f_ca, f_beh=f_beh, path=p_plot / f'{col}_1xy.svg')
