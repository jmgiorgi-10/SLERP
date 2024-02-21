#!/usr/bin/env python3

import glob
import numpy as np
import os
import torch
from mat_sci_torch_quats.quats import quat_upsampling_symm2, quat_upsampling_symm3
from mat_sci_torch_quats.quats import fz_reduce, scalar_last2first, scalar_first2last
from mat_sci_torch_quats.symmetries import hcp_syms
from mat_sci_torch_quats.symmetries import fcc_syms
import random

# The concept for this file is to implement slerp interpolation between quaternions, that never leaves the fundamental region.
# In order to achieve this goal, at each individual interpolation between quaterions we should apply the symmetry operator, and output the disorientation (minimum misorientation) angle.
# Hence, we are naming this interpolation symmetry-aware-slerp.

# files with deformed grains (high intra-grain misorientation)

def normalize(x):
    x_norm = torch.norm(x, dim=2, keepdim=True)
    # make ||q|| = 1
    y_norm = torch.div(x, x_norm) 

    return y_norm

file_array = sorted(glob.glob(f'/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/LR_Images/X4/preprocessed_imgs_all_Blocks/*.npy'))

# for file_loc in file_array:

scale = 4

quat_array = np.load(file_array[0])
X = torch.from_numpy(quat_array)
X = scalar_last2first(X) # switch quaternion convention


quat_array_upsampled = quat_upsampling_symm3(X, scale)
quat_array_upsampled.to(torch.device('cpu'))

# import pdb; pdb.set_trace()
basename = os.path.basename(file_array[0])
filename = os.path.splitext(basename)[0]

quat_array_upsampled = normalize(quat_array_upsampled)
basename = basename.replace('.npy', '')
np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_SLERPED.npy', quat_array_upsampled)

import pdb; pdb.set_trace()
quat_array_upsampled_fz = fz_reduce(quat_array_upsampled, fcc_syms) # reduces angles represented by quaternions to the fundamental zone of the specific crystal element 
quat_array_upsampled_fz = scalar_first2last(quat_array_upsampled_fz)

quat_array_upsampled = scalar_first2last(quat_array_upsampled)

# save the x_block_0 slerp, AFTER fz reduction.
np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_SLERPED_FZ.npy', quat_array_upsampled_fz)

