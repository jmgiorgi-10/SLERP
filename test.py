#!/usr/bin/env python3

import glob
import numpy as np
import os
import torch
from mat_sci_torch_quats.quats import quat_upsampling_symm2
from mat_sci_torch_quats.quats import fz_reduce, scalar_last2first, scalar_first2last
from mat_sci_torch_quats.symmetries import hcp_syms
from mat_sci_torch_quats.symmetries import fcc_syms
import random

import pdb; pdb.set_trace()

X = torch.zeros((10,10,4))
X_reshaped = X.reshape((-1, 4))

new_columns = torch.rand((10, 4))

X_reshaped = torch.hstack([X_reshaped, new_columns.unsqueeze(1)])
X_result = X_reshaped.reshape(10, -1)