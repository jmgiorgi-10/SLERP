#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from mat_sci_torch_quats.quats import quat_upsampling_symm, quat_upsampling_symm2, quat_upsampling_symm3, misorientation, symm_misorientation, symm_double_misorientation, symm_complete_misorientation, outer_prod, slerp_calc2, matrix_hamilton_prod, inverse2, quat_exp, slerp_calc
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
hr_array = sorted(glob.glob(f'/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Blocks/*.npy'))

# for file_loc in file_array:

scale = 4

quat_array = np.load(file_array[0])
hr_quat_array = np.load(hr_array[0])
# lr points are at multiples of 4
hr_quat_array = hr_quat_array[200, 156:156+4+1, :]
hr_quat_array = scalar_last2first(torch.from_numpy(hr_quat_array))

import pdb; pdb.set_trace()

# # taking just a single interpolation:
# quat_array = quat_array[201, 157:162, :]

X = torch.from_numpy(quat_array)
X = scalar_last2first(X) # switch quaternion convention

# X_up = torch.from_numpy(quat_upsampling_symm(X,4,False))
# X_up_symm = torch.from_numpy(quat_upsampling_symm(X,4,True))

# X_up = quat_upsampling_symm(X)
# X_up_symm = quat_upsampling_symm(X)

# X_row = X_up[200,156:161] # selecting a row that's known to have large ipf-mapping devations
# X_row_s = X_up_symm[200,156:161]

X_row = X[50,39:39+2]

# np.save(f'/media/hdd3/jmgiorgi/SLERP/test_row/test_row.npy', X_row)
# np.save(f'/media/hdd3/jmgiorgi/SLERP/test_row/test_row_symm.npy', X_row_s)

import pdb; pdb.set_trace()
# try the adjusted slerp here:
q1 = X_row[0]
q2 = X_row[1]

A = matrix_hamilton_prod(q1, inverse2(q2))
A_syms = outer_prod(A, fcc_syms)
# A_syms = A_syms[None,...]
a_min_indices = torch.max(A_syms[...,0],-1)[1]
A_min = A_syms[a_min_indices]
qs = inverse2(matrix_hamilton_prod(inverse2(q1),A_min))

q31 = slerp_calc(q1,qs,.25)
q32 = slerp_calc(q1,qs,.5)
q33 = slerp_calc(q1,qs,.75)

# q_i = matrix_hamilton_prod(q1, quat_exp(A_min[0],0.5))

## Slerp interpolation, no-symm [why no-symm slerp doesn't work]
# 1: mutual misorientations (show that q1 and q2 are far in rotational distance)
# misor_X_row = misorientation(X_row, X_row)

import pdb; pdb.set_trace()
misor = torch.zeros([sum(list(X_row[...,-2].shape)), sum(list(X_row[...,-2].shape))])
for i in range(X_row.shape[0]):
    misor[i,:] = misorientation(X_row, X_row[i]) * 180 / np.pi # broadcasting
# import pdb; pdb.set_trace()

# 2: differences in rotational distance to the origin - i.e. Delta_Theta0 (interpolated points should have > 5 degree difference in rotational distance compared to datapoints; reaching max at middle interpolation)
delta_theta_0_misor = torch.zeros([sum(list(X_row[...,-2].shape)), sum(list(X_row[...,-2].shape))])
for i in range(X_row.shape[0]):
    delta_theta_0_misor[i,:] = (misorientation(X_row, torch.Tensor([1,0,0,0])) - misorientation(X_row[i], torch.Tensor([1,0,0,0]))) * 180 / np.pi # broadcasting
# import pdb; pdb.set_trace()



## 24x1 misorientation
# 1: mutual disorientation (show that q1 and q2 are close in rotational distance)
misor24 = torch.zeros([sum(list(X_row_s[...,-2].shape)), sum(list(X_row_s[...,-2].shape))])
for i in range(X_row.shape[0]):
    # import pdb; pdb.set_trace()
    misor24[i,:] = symm_misorientation(X_row_s, X_row_s[i]) * 180 / np.pi # broadcasting
# import pdb; pdb.set_trace()

## 24x24 misorientation
# 1: mutual disorientation (show that q1 and q2 are close in rotational distance)
misor24x24 = torch.zeros([sum(list(X_row_s[...,-2].shape)), sum(list(X_row_s[...,-2].shape))])
for i in range(X_row.shape[0]):
    misor24x24[i,:] = symm_double_misorientation(X_row_s, X_row_s[i]) * 180 / np.pi # broadcasting
# import pdb; pdb.set_trace()

## 24x24x2 misorientation
misor24x24x2 = torch.zeros([sum(list(X_row[...,-2].shape)), sum(list(X_row[...,-2].shape))])
for i in range(X_row.shape[0]):
    misor24x24x2[i,:] = symm_complete_misorientation(X_row, X_row[i]) * 180 / np.pi # broadcasting

import pdb; pdb.set_trace()

# Find differences between all symm_data1 and symm_data2 permutations
data1 = hr_quat_array[0]
data2 = hr_quat_array[4]

data1_w_syms = outer_prod(data1, fcc_syms)
data2_w_syms = outer_prod(data2, fcc_syms)

# force all symmetries to be in +q hemisphere of 4d hyper-sphere:
data1_w_syms *= torch.sign(data1_w_syms[...,:1])
data2_w_syms *= torch.sign(data2_w_syms[...,:1])

#(data1,data2)
# 24x24x2 symmetry operations (now includes switching symmetry)
syms_diff_matrix = torch.zeros(24,24)

# Check if this method allows us to properly choose the symmetry option with the lowest theta (at least < 3 degrees is what we expect)
for i in range(24):
    for j in range(24):
        syms_diff_matrix[i,j] = torch.linalg.vector_norm(data1_w_syms[i] + data2_w_syms[j],2,-1)

import pdb; pdb.set_trace()
print(min(syms_diff_matrix.view(-1)))


# proper use of misorientation
syms_diff_matrix = torch.zeros(24,24,2)
for i in range(24):
    syms_diff_matrix[i,:,0] = matrix_hamilton_prod(data2_w_syms, inverse2(data1_w_syms[i]))
    syms_diff_matrix[i,:,1] = matrix_hamilton_prod(data1_w_syms, inverse2(data2_w_syms[i]))

import pdb; pdb.set_trace()



# 2: differences in rotational distance to the orirign 

# basename = os.path.basename(file_array[0])
# basename = basename.replace('.npy', '')
# filename = os.path.splitext(basename)[0] 


# import pdb; pdb.set_trace()
# quat_array_upsampled_fz = fz_reduce(quat_array_upsampled, fcc_syms) # reduces angles represented by quaternions to the fundamental zone of the specific crystal element 

# import pdb; pdb.set_trace()

# # ### Selecting a Single 5x5 Window
# # quat_array_upsampled_fz = quat_array_upsampled_fz[200:205, 156:161, :]
# quat_array_upsampled_fz = quat_array_upsampled_fz[200, 156:161, :]

# delta_t = 0.01
# # quat_array_upsampled_fz = quat_array_upsampled_fz +delta*np.ones(quat_array_upsampled_fz.shape)

# # ### Selecting a Single 5x5 Window


# quat_array_upsampled = scalar_first2last(quat_array_upsampled)

# import pdb; pdb.set_trace()
# # quat_diff = torch.sum(torch.abs(quat_array_upsampled_fz - hr_quat_array),-1)

# quat_array_upsampled_fz = scalar_first2last(quat_array_upsampled_fz)


# # plt.imsave(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/quat_diff.png', quat_diff)


# np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_SLERPED.npy', quat_array_upsampled)

# # save the x_block_0 slerp, AFTER fz reduction.
# np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_SLERPED_FZ.npy', quat_array_upsampled_fz)

# np.save(f'/media/hdd3/jmgiorgi/SLERP-symm/Open_718/X4/{basename}_HR_ARRAY.npy', scalar_first2last(hr_quat_array))
