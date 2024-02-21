#!/usr/bin/env python3

import random
import torch
from mat_sci_torch_quats.quats import misorientation
from mat_sci_torch_quats.symmetries import fcc_syms



# forcibly globally restrict theta to symmetry limit (theta = 62 degrees for cubic), wrt point in LR Image.
# since LR is fz-reduced wrt [1,0,0,0], our output will be fz-reduced wrt to Theta=0, as well as wrt to a common point of reference
# we expect this will help problem areas, which are 
def restrict_theta(X, thresh_degree):

  num_rows = X.shape[0]
  num_cols = X.shape[1]

  X_new = np.zeros(X.shape)

  quat_ref = X[0,0,:] # reference quaternion (LR point; fz-reduced wrt [1,0,0,0])
  quat_ref_syms = outer_prod(quat_ref, fcc_syms)

  for i in range(num_rows):
    for j in range(num_cols):
      quat0 = X[i,j,:]
      dists = misorientation(quat_ref_syms, quat_ref)
      inds = torch.min(dists,-1)[1]
      q3 = q3_w_syms[torch.arange(len(q3_w_syms)), inds]


