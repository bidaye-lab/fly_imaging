!!! Work in progress !!!
Code in this repos is currently being reorganized

# old scripts
- scripts/old_notebooks/ca_imaging_stim.ipynb
- scripts/old_notebooks/ca_imaging.ipynb
- scripts/old_notebooks/focused_heatmaps.ipynb
- scripts/old_notebooks/hacky_aligned_behavior.ipynb
- scripts/old_notebooks/pretty_plot.ipynb

# new scripts
- scripts/full_pipeline.py


# Installation

```
# create conda environment with necessary dependencies
conda create -n imaging_analysis -f environment.yml

# get source code
git clone https://github.com/bidaye-lab/imaging_analysis

# install code as local local python module
cd imaging_analysis
pip install -e .
```

# How to use this repo
## Analysis pipelines
The script files `scripts/*.py` are workflows to analyse imaging data.

## Code structure
The analysis pipelines in the `scripts/` folder
call the code stored in the `src/` folder.
The [installation](#installation) method makes the code in the `src/` folder
available as a local python package.
This setup loosely follows the [The Good Research Code Handbook](https://goodresearch.dev/s).
