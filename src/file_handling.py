import numpy as np
from pathlib import Path

from tifffile import imwrite, imread, TiffFile
from scipy.io import loadmat


def fname(root_file, new_ending, new_root=""):
    """Helper function to generate path for output files

    This will strip the suffix from `root_file` and attach a new
    file ending to it.

    Examples
    -------
    Without `new_root`
    >>> fname(root_file='data/subfolder/image.tif', new_ending='mean_ch1.png')
    >>> 'data/subfolder/image_mean_ch1.png'

    With `new_root`
    >>> fname(root_file='data/subfolder/image.tif', new_ending='mean_ch1.png', new_root='data/output')
    >>> 'data/output/subfolder/image_mean_ch1.png'


    Parameters
    ----------
    root_file : pathlike
        Path to original file (e.g. original TIF data file)
    new_ending : str
        New ending to be placed after suffix is removed
    new_root : str, pathlike
        If not '', place output file in subfolders in `new_root`
        relative to `root_file.parent`. If '', place in folder of
        `root_file`. By default ''

    Returns
    -------
    new_file : Path
        Path to the output file
    """

    root_file = Path(root_file)  # ensure that Path

    new_file = root_file.parent / "{}_{}".format(
        root_file.with_suffix("").name, new_ending
    )

    if new_root:
        new_root = Path(new_root)
        new_file = new_root / new_file.relative_to(new_root.parent)
        new_file.parent.mkdir(exist_ok=True, parents=True)

    return new_file


def load_tiff(file, correct_offset=False, subtract_min=False):
    """Load TIF file with optional processing.

    Parameters
    ----------
    file : pathlike
        Path to TIF file
    correct_offset : bool, optional
        If True, correct offset per channel according to 'SI.hScan2D.channelOffsets'
        scanimage metadata, by default False
    subtract_min : bool, optional
        If True, subtract min value to ensure that all values are positive, by default False

    Returns
    -------
    img : np.array
        Data from TIF file
    """
    # load data
    img = imread(file)
    print(f"INFO loaded tiff stack from {file} with shape {img.shape}")

    # correct for scanimage offsets
    if correct_offset:
        tif = TiffFile(file)
        offsets = tif.scanimage_metadata["FrameData"]["SI.hScan2D.channelOffsets"]
        img[:, 0] -= offsets[0]
        img[:, 1] -= offsets[1]
        print(
            f"INFO added offsets: {offsets[0]} (channel 1) | {offsets[1]} (channel 2)"
        )

    # assure that all intensities are positive
    if subtract_min:
        min1, min2 = img[:, 0].min(), img[:, 1].min()
        img[:, 0] -= min1
        img[:, 1] -= min2
        print(f"INFO subtracted minimum value: {min1} (channel 1) | {min2} (channel 2)")

    return img


def load_tiff_files(l_tif):
    """Load a list of TIF files and return as single array

    Calls `scr.file_handling.load_tiff`, so defaults of that function are used.

    Parameters
    ----------
    l_tif : list
        List of pathlike objects pointing to TIF files

    Returns
    -------
    stack : np.array
        Array with concatenated data from all TIF files
    """
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
    """Write array as TIF to disk

    Parameters
    ----------
    file : pathlike
        Path to output TIF file
    arr : np.array
        Array with data
    """
    print(f"INFO writing images to {file}")
    imwrite(file, arr, photometric="minisblack")


def get_roi_zip_file(p_tif):
    """Returns the path of the *RoiSet.zip file

    Since file names for the ImageJ-generated ROI file may differ,
    this helper function looks in the parent directory of `p_tif`
    to look for a file with the ending `*RoiSet.zip`. Returns None
    if zero or multiple files are matched.

    Parameters
    ----------
    p_tif : Path
        Path to file in same directory

    Returns
    -------
    p_zip : Path or None
        If exactly one file matches, returns path to that file,
        otherwise return None
    """
    g = p_tif.parent.glob("*RoiSet.zip")

    try:
        p_zip = next(g)
    except StopIteration:
        print(f"WARNING no *RoiSet.zip file found")
        return

    try:
        next(g)
        print(f"WARNING multiple *RoiSet.zip files found")
        return
    except StopIteration:
        return p_zip


def get_matlab_files(p_tif):
    """Search for ball and behavior file in parent directory

    Looks for the ball velocity file by taking the first part before
    `_` of the `p_tif` basename and appending `.mat`.
    Looks for the behavior file by looking for `*actions.mat` in the
    parent folder of `p_tif`. Returns None if zero and multiple files
    are found

    Parameters
    ----------
    p_tif : Path
        Path to file in same directory

    Returns
    -------
    p_ball, p_beh : (Path, Path) or None
        If both files found, return tuple of Path to them, otherwise return None
    """
    p_ball = p_tif.parent / (p_tif.name.split("_")[0] + ".mat")
    if not p_ball.is_file():
        print("WARNING ball velocity matlab file not found")
        return

    g = p_tif.parent.glob("*-actions.mat")

    try:
        p_beh = next(g)
    except StopIteration:
        print(f"WARNING no *actions.mat file found")
        return

    try:
        next(g)
        print(f"WARNING multiple *actions.mat files found")
        return
    except StopIteration:
        return p_ball, p_beh


def load_behavior(p_mat, beh_keys):
    """Load arrays with behavior from matlab file

    Parameters
    ----------
    p_mat : pathlike
        Path to matlab behavior file
    beh_keys : list of str
        Behavior keys to look for in `p_mat`

    Returns
    -------
    beh : dict
        Mapping between behavior key and data array
    """
    m = loadmat(p_mat, squeeze_me=True, struct_as_record=False)

    beh = {k: v for k, v in zip(m["behs"], m["bouts"]) if k in beh_keys}

    return beh


def load_ball(p_mat):
    """Load ball velocity data

    Parameters
    ----------
    p_mat : pathlike
        Path to ball velocity matlab file

    Returns
    -------
    ball : np.array
        Ball velocity data in x, y, and z
    """
    m = loadmat(p_mat, squeeze_me=True, struct_as_record=False)

    ball = vars(m["sensorData"])["bufferRotations"]

    nans = np.all(np.isnan(ball), axis=1)
    ball = ball[~nans]

    return ball
