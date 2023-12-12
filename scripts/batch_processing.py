import numpy as np
import pandas as pd
import pickle

from src.file_handling import (
    fname,
    load_tiff,
    write_tiff,
    save_img,
    save_dual_movie,
    get_roi_zip_file,
    get_matlab_files,
    load_ball,
    load_behavior,
)

from src.processing import (
    split_channels,
    maxproj_z,
    smooth_xy,
    zscore_cols,
    upsample_to_behavior,
    convolute_ca_kernel,
)

from src.registration import get_tmats, align
from src.roi import (
    read_imagej_rois,
    draw_rois,
    get_mean_trace,
    subtract_baseline,
    subtract_background,
)

from src.visualization import plot_data, plot_corr_heatmap, plot_ccf


def motion_correction(params):
    # load parameters
    p_tifs = params["p_tifs"]
    p_out = params["p_out"]
    overwrite = params["overwrite"]
    xy_smth = params["xy_smth"]
    reg = params["reg"]

    for p_tif in p_tifs:
        print()

        # define plot folder
        p_plot = fname(p_tif, "", new_root=p_out).parent / "plots"
        p_plot.mkdir(exist_ok=True)

        if fname(p_tif, "ch1.tif", new_root=p_out).is_file() and not overwrite:
            print(f"INFO output file exists, skipping registration for {p_tif.parent}")
            continue
        else:
            print(f"INFO now registering {p_tif}")

        # load and split
        stack = load_tiff(p_tif)
        ch1, ch2 = split_channels(stack, n_z=15, n_ch=2)
        ch1 = maxproj_z(ch1)
        ch2 = maxproj_z(ch2)

        ch1 = smooth_xy(ch1, xy_smth)
        ch2 = smooth_xy(ch2, xy_smth)

        # register
        tmats = get_tmats(ch2, reg)
        ch1_a = align(ch1, tmats, reg)
        ch2_a = align(ch2, tmats, reg)

        # mean image
        ch1_am = np.mean(ch1_a, axis=0)
        ch2_am = np.mean(ch2_a, axis=0)

        # save to disk
        write_tiff(fname(p_tif, "ch1.tif", new_root=p_out), ch1_a.astype("int16"))
        write_tiff(fname(p_tif, "ch2.tif", new_root=p_out), ch2_a.astype("int16"))

        write_tiff(fname(p_tif, "ch1reg.tif", new_root=p_out), ch1_a.astype("int16"))
        write_tiff(fname(p_tif, "ch2reg.tif", new_root=p_out), ch2_a.astype("int16"))

        save_img(p_plot / "ch1mean.bmp", ch1_am)
        save_img(p_plot / "ch2mean.bmp", ch2_am)

        save_dual_movie(p_plot / "ch1ch2.mp4", ch1, ch2)
        save_dual_movie(p_plot / "ch1reg.mp4", ch1, ch1_a)
        save_dual_movie(p_plot / "ch2reg.mp4", ch2, ch2_a)


def extract_trace(params):
    # load parameters
    p_tifs = params["p_tifs"]
    p_out = params["p_out"]
    overwrite = params["overwrite"]
    perc = params["perc"]
    winsize = params["winsize"]

    for p_tif in p_tifs:
        print()

        # define plot folder
        p_plot = fname(p_tif, "", new_root=p_out).parent / "plots"
        p_plot.mkdir(exist_ok=True)

        # check if ROI traces have already been extracted
        p_roi = fname(p_tif, "roi_traces.pickle", new_root=p_out)
        if p_roi.is_file() and not overwrite:
            print(
                f"INFO output files exists, skipping ROI extraction for {p_tif.parent}"
            )
            continue

        # load Roi.zip
        p_zip = get_roi_zip_file(p_tif)
        if not p_zip:
            print(f"WARNING Skipping {p_tif.parent}")
            continue
        else:
            print(f"INFO loading ROIs from {p_zip}")

        # load aligned data
        ch1_a = load_tiff(fname(p_tif, "ch1reg.tif", new_root=p_out))
        ch2_a = load_tiff(fname(p_tif, "ch2reg.tif", new_root=p_out))

        # load ROIs
        img = np.mean(ch1_a, axis=0)
        rois = read_imagej_rois(p_zip, img)
        img_rois = draw_rois(img, rois)
        save_img(p_plot / "ch1mean_rois.bmp", img_rois)

        # extract traces
        d_roi = dict()  # collect traces extracted with different methods here

        ca1 = get_mean_trace(rois, ch1_a)
        ca2 = get_mean_trace(rois, ch2_a)
        d_roi["ch1raw"] = ca1
        d_roi["ch2raw"] = ca2

        # channel 1 to 2 ratio
        r12 = ca1 / ca2
        d_roi["r12"] = r12

        # subtract baseline
        dca1 = subtract_baseline(ca1, percentile=perc, size=winsize)
        dca2 = subtract_baseline(ca2, percentile=perc, size=winsize)
        dr12 = subtract_baseline(r12, percentile=perc, size=winsize)
        d_roi["dch1"] = dca1
        d_roi["dch2"] = dca2
        d_roi["dr12"] = dr12

        # subract background ROI
        ca1 = subtract_background(ca1)
        ca2 = subtract_background(ca2)
        d_roi["ch1"] = ca1
        d_roi["ch2"] = ca2

        # save to disk
        with open(p_roi, "wb") as f:
            pickle.dump(d_roi, f)
        print(f"INFO saving ROI traces to {p_roi}")


def merge_imaging_and_behavior(params):
    # load parameters
    p_tifs = params["p_tifs"]
    p_out = params["p_out"]
    overwrite = params["overwrite"]

    beh_keys = params["beh_keys"]
    f_ca, f_ball, f_beh = params["f_ca"], params["f_ball"], params["f_beh"]

    for p_tif in p_tifs:
        print()

        # define plot folder
        p_plot = fname(p_tif, "", new_root=p_out).parent / "plots"
        p_plot.mkdir(exist_ok=True)

        # load ROI traces
        p_roi = fname(p_tif, "roi_traces.pickle", new_root=p_out)
        if not p_roi.is_file():
            print(f"WARNING file with ROI traces not found, skipping {p_tif.parent}")
            continue
        else:
            with open(p_roi, "rb") as f:
                d_roi = pickle.load(f)

        # load behavior data and ball velocities
        p_mats = get_matlab_files(p_tif)
        if not p_mats:
            print(f"WARNING skipping {p_tif.parent}")
            continue
        else:
            p_ball, p_beh = p_mats

        ball = load_ball(p_ball)
        beh = load_behavior(p_beh, beh_keys)

        for method, traces in d_roi.items():
            # check if already been processed
            p_df = fname(p_tif, f"data_{method}.parquet", new_root=p_out)
            if p_df.is_file() and not overwrite:
                print(
                    f"INFO output files exists, skipping data merging for method {method} in {p_tif.parent}"
                )
                continue

            # match sample rates
            df = upsample_to_behavior(traces, beh, ball, f_ca, f_ball, f_beh)
            # zscore ROIs
            df = zscore_cols(df, col_start="roi_")
            # convolute ball velocities and behavior with Ca kernel
            df = convolute_ca_kernel(df, f=f_beh)
            # zscore ball velocities
            df = zscore_cols(df, col_start="conv_ball_")

            # add additional data based on file and folder names
            pt = p_tif.parts
            cond, fly, trial = pt[-5], pt[-4], pt[-2]
            df.loc[:, "cond"] = cond  # e.g. fed/starved
            df.loc[:, "fly"] = fly  # fly number
            df.loc[:, "trial"] = trial  # trial number
            print(
                f"INFO parsing folder names: fly {fly} | trial {trial} | condition {cond}"
            )

            # plots for quality control
            plot_data(df, f_beh, path=p_plot / f"data_{method}.png")
            # pearson r heatmap
            plot_corr_heatmap(df, beh="behi", path=p_plot / f"heatmap_{method}.png")
            # ccf
            plot_ccf(df, f=f_beh, pool_fly=True, path=p_plot / f"ccf_{method}.png")

            # save to disk
            print(f"INFO writing merged data to {p_df}")
            df.to_parquet(p_df)

            # optional (will be big files): save also as CSV
            # df.to_csv(p_df.with_suffix('.csv'))


def merge_sessions(params):
    # load parameters
    p_tifs = params["p_tifs"]
    p_out = params["p_out"]
    overwrite = params["overwrite"]

    # get methods
    all_pars = []
    for p in p_tifs:
        l = [*fname(p, "", new_root=p_out).parent.glob("*data_*.parquet")]
        all_pars.extend(l)
    methods = {p.stem.split("_")[-1] for p in all_pars}

    for method in methods:
        # list of all *_data_{method}.parquet files
        p_pars = [fname(p, f"data_{method}.parquet", new_root=p_out) for p in p_tifs]

        l = []
        for p_par in p_pars:
            if not p_par.is_file():
                print(f"WARNING skipping {p_par.parent}")
                continue
            else:
                print(f"INFO loading file {p_par}")
                df = pd.read_parquet(p_par)
                l.append(df)

        if l:
            # combine dataframes and save
            df = pd.concat(l, ignore_index=True)
            p_df = p_out / f"all_data_{method}.parquet"
            if overwrite:
                df.to_parquet(p_df)

            print(f"INFO contents of {p_df}")
            for f, d in df.groupby("fly"):
                print(f"     {f}", end=": ")
                for t, _ in d.groupby("trial"):
                    print(f"{t}", end=" ")
                print()
        else:
            # this check should not be necessary (?)
            print(f"WARNING no data files found, skipping {method}")
