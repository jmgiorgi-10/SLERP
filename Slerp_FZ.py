#!/usr/bin/python3

import glob
import numpy as np
import os
import torch
from slerp3 import quat_upsampling
# from mat_sci_torch_quats.quats_joaquin import fz_reduce, scalar_last2first, scalar_first2last
from mat_sci_torch_quats.quats import fz_reduce, scalar_last2first, scalar_first2last
from mat_sci_torch_quats.symmetries import hcp_syms
from mat_sci_torch_quats.symmetries import fcc_syms

## This file will perform fundamental zone reduction

file_array = sorted(glob.glob(f'/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/LR_Images/X4/preprocessed_imgs_all_Blocks/*.npy'))

for file_loc in file_array:
    quat_array = np.load(file_loc)
    quat_array_upsampled = torch.from_numpy(quat_upsampling(quat_array, 4))
    quat_array_upsampled.to(torch.device('cpu'))

    # import pdb; pdb.set_trace()
    basename = os.path.basename(file_loc)
    filename = os.path.splitext(basename)[0]

    # # Save the x_block_0 slerp upsampled array, PRIOR to fz reduction.
    # np.save(f'{basename}', quat_array_upsampled)

    # Fundamental zone reduction
    # quat_array_upsampled_fz = scalar_last2first(quat_array_upsampled)
    quat_array_upsampled_fz = fz_reduce(quat_array_upsampled, fcc_syms) # reduces angles represented by quaternions to the fundamental zone of the specific crystal element 
    quat_array_upsampled_fz = scalar_first2last(quat_array_upsampled_fz)

    # save the x_block_0 slerp, AFTER fz reduction.
    basename = basename.replace('.npy', '')
    np.save(f'/media/hdd3/jmgiorgi/SLERP/Open_718/X4/{basename}_SLERPED.npy', quat_array_upsampled_fz)
