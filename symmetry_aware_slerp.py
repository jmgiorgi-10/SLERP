#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from mat_sci_torch_quats.quats import quat_upsampling_symm, quat_upsampling_symm2, quat_upsampling_symm3, misorientation, outer_prod, slerp_calc2
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

import pdb; pdb.set_trace()
file_array = sorted(glob.glob(f'/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/LR_Images/X4/preprocessed_imgs_all_Blocks/*.npy'))
hr_array = sorted(glob.glob(f'/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Blocks/*.npy'))

# for file_loc in file_array:

scale = 4

quat_array = np.load(file_array[0])
hr_quat_array = np.load(hr_array[0])
# lr points are at multiples of 4
hr_quat_array = hr_quat_array[200:200 + 4 +1, 156:156+4+1, :]
hr_quat_array = scalar_last2first(torch.from_numpy(hr_quat_array))

# # taking just a single interpolation:
# quat_array = quat_array[201, 157:162, :]

X = torch.from_numpy(quat_array)
X = scalar_last2first(X) # switch quaternion convention

# quat_array_upsampled = quat_upsampling_symm3(X, scale)
quat_array_upsampled = quat_upsampling_symm3(X, scale)
# quat_array_upsampled.to(torch.device('cpu'))
# quat_array_upsampled = torch.from_numpy(quat_array_upsampled)
# quat_array_upsampled = normalize(quat_array_upsampled)

# quat_array_upsampled_fz = fz_reduce(quat_array_upsampled, fcc_syms) # reduces angles represented by quaternions to the fundamental zone of the specific crystal element 
# quat_array_upsampled_fz = scalar_first2last(quat_array_upsampled_fz)

# import pdb; pdb.set_trace()
basename = os.path.basename(file_array[0])
basename = basename.replace('.npy', '')
filename = os.path.splitext(basename)[0] 

##
# Will try converting to scalar_first2last before fz_reduction:
##

import pdb; pdb.set_trace()
quat_array_upsampled_fz = fz_reduce(quat_array_upsampled, fcc_syms) # reduces angles represented by quaternions to the fundamental zone of the specific crystal element 

import pdb; pdb.set_trace()

# ### Selecting a Single 5x5 Window
# quat_array_upsampled_fz = quat_array_upsampled_fz[200:205, 156:161, :]

delta_t = 0.01
# quat_array_upsampled_fz = quat_array_upsampled_fz +delta*np.ones(quat_array_upsampled_fz.shape)

# ### Selecting a Single 5x5 Window
# quat_array_upsampled = quat_array_upsampled[200:205, 156:161, :]

quat_array_upsampled = scalar_first2last(quat_array_upsampled)

import pdb; pdb.set_trace()
# quat_diff = torch.sum(torch.abs(quat_array_upsampled_fz - hr_quat_array),-1)

quat_array_upsampled_fz = scalar_first2last(quat_array_upsampled_fz)


# plt.imsave(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/quat_diff.png', quat_diff)


np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_SLERPED.npy', quat_array_upsampled)

# save the x_block_0 slerp, AFTER fz reduction.
np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_SLERPED_FZ.npy', quat_array_upsampled_fz)

np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_HR_ARRAY.npy', scalar_first2last(hr_quat_array))

