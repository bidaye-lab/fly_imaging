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

from scripts.batch_processing import motion_correction, extract_traces, merge_imaging_and_behavior, merge_sessions

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

# set and create output folder
params["p_out"] = params["parent_dir"] / "smth4_perc10_winsize50"
params["p_out"].mkdir(exist_ok=True)

# print info
print("INFO: Saving files to {}".format(params["p_out"]))
print("INFO: Found {} files:".format(params["p_tifs"]))
print(params["p_tifs"])

# %% [markdown]
# # full processing pipeline

# %%
# step 1
motion_correction(params)

# %%
# step 2
extract_traces(params)

# %%
# step 3
merge_imaging_and_behavior(params)

# %%
# step 4
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
