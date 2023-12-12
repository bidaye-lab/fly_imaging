import numpy as np

# registration
from pystackreg import StackReg

# parallel processing
from joblib import Parallel, delayed, parallel_backend



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