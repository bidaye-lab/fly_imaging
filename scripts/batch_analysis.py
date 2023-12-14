import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.file_handling import fname, load_tiff, write_tiff
from src.visualization import (
    plot_corr_heatmap,
    plot_ccf,
    plot_aligned,
    plot_corrmap,
    save_dual_movie,
    save_img,
)
from src.processing import align2events, select_event
from src.registration import align, get_tmats


def generate_plots(params):
    # load parameters
    p_out = params["p_out"]

    f_beh = params["f_beh"]

    # merge all trials and flies
    for p_df in p_out.glob("all_data_*.parquet"):
        method = p_df.stem.split("_")[-1]

        # create plot folder
        p_plot = p_out / method
        p_plot.mkdir(exist_ok=True)

        # read data from disk
        df = pd.read_parquet(p_df)
        df = df.fillna(0)  # TODO workaround because of missing behavior

        # optional: remove ROI 7, 8, 9
        df = df.drop(columns=["z_roi_7", "z_roi_8", "z_roi_9"])

        # pearson correlation heatmap (selection of columns: see utl.calculate_pearson)
        plot_corr_heatmap(df, beh="behi", path=p_plot / "heatmap.svg")

        # pearson heatmaps around behavior events
        dt = 5  # time in s to keep before and after behavior event

        # loop through all behavior
        cols = [c for c in df.columns if c.startswith("beh_")]
        for col in cols:
            # select df around behavoir
            d = select_event(df, col, f_beh, dt)

            # generate plot
            plot_corr_heatmap(d, beh="behi", path=p_plot / f"heatmap_{col}.svg")

        # cross-correlation functions behavior/ball and ROIs (averaged)
        plot_ccf(df, f=f_beh, pool_fly=True, path=p_plot / "ccf.svg")

        # same (not averaged)
        plot_ccf(df, f=f_beh, pool_fly=False, path=p_plot / "ccf_indv.svg")

        # plot aligned data
        dt = 5  # time in s before and after behavior event
        s = 0.25  # smoothing window for velocity [in s]

        # smooth velocity
        df_ = df.copy()
        df_.loc[:, ["ball_x", "ball_y", "ball_z"]] = gaussian_filter(
            df_.loc[:, ["ball_x", "ball_y", "ball_z"]].values, (s * f_beh, 0)
        )

        # cycle through all behavoirs
        cols = [c for c in df_.columns if c.startswith("beh_")]
        for col in cols:
            # align to behavior
            df_al = align2events(df_, col, f_beh, dt)

            plot_aligned(df_al, path=p_plot / f"aligned_to_{col}.svg")


def spatial_corrmaps(params):
    # load parameters
    p_tifs = params["p_tifs"]
    p_out = params["p_out"]
    f_ca, f_beh = params["f_ca"], params["f_beh"]

    # define tif file for session of interest
    for p_tif in p_tifs:
        # define and create output folder
        p_plot = fname(p_tif, "", new_root=p_out).parent / "corrmaps"
        p_plot.mkdir(exist_ok=True)
        print(p_plot)

        # load registered tif files
        ch1 = load_tiff(fname(p_tif, "ch1reg.tif", new_root=p_out))
        ch2 = load_tiff(fname(p_tif, "ch2reg.tif", new_root=p_out))

        # load preprocessed behavior data
        df = pd.read_parquet(
            fname(p_tif, "data_ch1.parquet", new_root=p_out)
        )  # either of the data files work

        # loop through all conv behi and conv ball columns
        cols = [
            c
            for c in df.columns
            if c.startswith("conv_behi") or c.startswith("conv_ball")
        ]
        for col in cols:
            plot_corrmap(
                ch1,
                ch2,
                df,
                col,
                f_ca=f_ca,
                f_beh=f_beh,
                path=p_plot / f"{col}_1xy.svg",
            )


def pooled(params, p_tifs):
    # load parameters
    p_out = params["p_out"]
    reg = params["reg"]
    f_ca, f_beh = params["f_ca"], params["f_beh"]

    p_plot = p_out / "pooled_corrmap"
    p_plot.mkdir(exist_ok=True)

    # combine tifs
    l1, l2 = [], []
    for p_tif in p_tifs:
        ch1 = load_tiff(fname(p_tif, "ch1.tif", new_root=p_out))
        ch2 = load_tiff(fname(p_tif, "ch2.tif", new_root=p_out))
        l1.append(ch1)
        l2.append(ch2)

    ch1 = np.concatenate(l1, axis=0)
    ch2 = np.concatenate(l2, axis=0)

    # register
    tmats = get_tmats(ch2, reg)
    ch1_a = align(ch1, tmats, reg)
    ch2_a = align(ch2, tmats, reg)

    # mean image
    ch1_am = np.mean(ch1_a, axis=0)
    ch2_am = np.mean(ch2_a, axis=0)

    # save to disk
    write_tiff(p_plot / "ch1.tif", ch1_a.astype("int16"))
    write_tiff(p_plot / "ch2.tif", ch2_a.astype("int16"))

    write_tiff(p_plot / "ch1reg.tif", ch1_a.astype("int16"))
    write_tiff(p_plot / "ch2reg.tif", ch2_a.astype("int16"))

    save_img(p_plot / "ch1mean.bmp", ch1_am)
    save_img(p_plot / "ch2mean.bmp", ch2_am)

    save_dual_movie(p_plot / "ch1ch2.mp4", ch1, ch2)
    save_dual_movie(p_plot / "ch1reg.mp4", ch1, ch1_a)
    save_dual_movie(p_plot / "ch2reg.mp4", ch2, ch2_a)

    # load preprocessed behavior data
    l = []
    for p_tif in ps:
        df = pd.read_parquet(
            fname(p_tif, "data_ch1.parquet", new_root=p_out)
        )  # either of the data works
        l.append(df)
    df = pd.concat(l, ignore_index=True)

    # loop through all conv behi and conv ball columns
    cols = [
        c for c in df.columns if c.startswith("conv_behi") or c.startswith("conv_ball")
    ]
    for col in cols:
        plot_corrmap(
            ch1, ch2, df, col, f_ca=f_ca, f_beh=f_beh, path=p_plot / f"{col}_1xy.svg"
        )
