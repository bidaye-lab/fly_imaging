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

from pathlib import Path
from pystackreg import StackReg

from scripts.batch_processing import motion_correction_based_on_ch2, extract_traces, merge_imaging_and_behavior, merge_sessions
from scripts.batch_analysis import generate_plots, spatial_corrmaps, pooled

# %% [markdown]
# # User settings

# %%
params = {
    "n_ch": 2,  # number of channels
    "n_z": 15,  # number of z slices
    "xy_smth": 4,  # smoothing in xy before registration (sigma 2D Gaussian)
    "perc": 10,  # settings for background subtraction: percentile
    "winsize": 50,  # and window size (winsize / f_ca = window in seconds)
    "f_ca": 2,  # sample rate imaging
    "f_ball": 50,  # sample rate ball velocity
    "f_beh": 200,  # sample rate behavior
    "beh_keys": [  # keys in behavior file to consider
        "pushing",
        "hindLegRub",
        "frontLegRub",
        "headGrooming",
        "abdomenGrooming",
        "PER",
        "midLeft+hindLegRub",
        "midRight+hindLegRub",
        # 'ObL1+R1'
    ],
    # transformation used for registration https://pystackreg.readthedocs.io/en/latest/readme.html#usage
    "reg": StackReg.SCALED_ROTATION,
    # path to folder
    "parent_dir": r"Y:\Nino_2P_for_Salil\for_Nico\stop1_filteredData_2\stop1-GCaMP6f-tdTomato_VNC_smth4",
    "overwrite": True,  # force overwriting files
}

# ensure `parent_dir` is Path object
params["parent_dir"] = Path(params["parent_dir"])

# selection rule for tif files
params["p_tifs"] = [
    *params["parent_dir"].glob("**/trials_to_register/*/trial*_00???.tif")
]

# set and create output folders
params["p_out"] = Path(r'C:\temp\imaging_pipeline\test_run')
params["p_out_all"] = params["p_out"] / "all_data"
params["p_out_all"].mkdir(exist_ok=True, parents=True)

# print info
print("INFO: Saving files to {}".format(params["p_out"]))
print("INFO: Found {} files:".format(len(params["p_tifs"])))
for p in params["p_tifs"]:
    print(p)

# %% [markdown]
# # Processing
# The processing part contains
# - motion correction
# - fluorescence trace extraction
# - merging behavior and imaging data
#
# All output is stored in the `params["p_out"]` folder.
# Here, the folder structure of all `params["p_tifs"]` files is regenerated relative to `params["parent_dir"]`.
# These folders contain all intermediate files of the processing pipeline, 
# as well as a `plots` folder with analysis for that specific TIF file.
#
# The output folder further contains an `all_data` folder,
# which contains analysis pooled over all TIF files.

# %% [markdown]
# ## Motion correction
# The motion correction is done with [pystackreg](https://pystackreg.readthedocs.io/en/latest/readme.html#summary), which is a python port of the TurboReg/StackReg extension in ImageJ.
#
# The processing steps are
# - maximum projection along z
# - smoothing with 2D Gaussian along xy (`params['xy_smth']`)
# - alignment based on channel 2 (`params['reg']`)
#
# Following files are generated for each TIF file, split by channel:
# - uncorrected maxproj data (`*ch1.tif` and `ch2.tif`)
# - motion-corrected maxproj data (`*ch1reg.tif` and `ch2reg.tif`)
# - mean maxproj image (`*ch1mean.bpm` and `*ch2mean.bpm`)
# - side-by-side movies (quick quality control)
#     - maxproj channel 1 and 2 (`*ch1ch2.mp4`)
#     - uncorrected and motion-corrected maxproj channel 1 (`*ch1reg.mp4`)
#     - uncorrected and motion-corrected maxproj channel 2 (`*ch2reg.mp4`)
#
#
# For more details, look into the `scripts.batch_processing.motion_correction` function.
#
#

# %%
motion_correction_based_on_ch2(params)

# %% [markdown]
# ## Trace extraction
# This part loads the motion-corrected TIF files saved to disk
# and extracts the fluorescence traces for ROIs defined in the ImageJ Roi.zip file.
#
# Check `ch1mean_rois.bmp` if ROIs are parsed correctly.
#
# Following files are generated for each TIF file, split by channel.
# The name indicates for which data the trace was extracted:
# - `ch1raw/ch2raw`: raw data
# - `r12`: ratio of ch1 to ch2
# - `dch1/dch2`: baseline subtracted from raw data
# - `dr12`: baseline subtracted from ratio
# - `ch1/ch2`: background (last) ROI subtracted from raw data
#

# %%
extract_traces(params)

# %% [markdown]
# ## Merge imaging and behavior data
# Imaging data, behavior data, and ball velocity data is recorded at different sampling rates.
# Here, they are resampled to the sampling rate of the behavior data and merged into one pandas.DataFrame.
#
# The ball velocity file is expected to be called `trial16.mat` for the TIF file `trial16_00001.tif`.
#
# The behavior data file is expteced to be called `*-actions.mat`

# %%
merge_imaging_and_behavior(params)

# %% [markdown]
# As a final processing step,
# the data from all TIF files is merged into one pandas.DataFrame and 
# saved to `params["all_data"] / all_data_{method}.parquet` (see _Trace extraction_).

# %%
merge_sessions(params)

# %% [markdown]
# # Analysis

# %% [markdown]
# ## All recordings

# %%
generate_plots(params)

# %% [markdown]
# # spatial correlation maps (long)

# %%
spatial_corrmaps(params)

# %% [markdown]
# ## pooled

# %%
# path to folder
ps = [ *params['parent_dir'].glob('female13/trials_to_register/*/*0000?.tif') ]
pooled(params, ps)
