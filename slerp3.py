## Purpose: Bilinear slerp interpolation ##
## Updates from 'slerp1.py': Slerp after realizing that the data input convention is scalar last (permute the indices prior to upsampling), and keep scalar first convention for output. #

import numpy as np
import math

class Quat:
    def __init__(self, q):
        self.q = q

    def __getitem__(self, idx):
        return self.q[idx]

    def __setitem__(self, val, idx=None):
        if (idx == None):
            set_array(self, val)
        else:
            self.q[idx] = val

    def set_array(self, array):
        self.q = array

    def __pow__(self, t):
        new_q_array = np.zeros(4)
        mag = math.sqrt(self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)
        phi = math.acos(self.q[0] / mag)
        v = self.q[1:4] # quaternion versor
        v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if v_mag != 0:
            normal = v / v_mag
        else:
            # import pdb; pdb.set_trace()
            normal = v
        new_q_array[0] = mag**t * math.cos(t*phi)
        new_q_array[1:4] = (mag**t * math.sin(t*phi)) * v
        return(Quat(new_q_array))

    def __add__(self, q2): # addition overload
        self.q = self.q + q2

    def __mul__(self, q2): # multiplication overload (Hamilton product)
        if isinstance(q2, Quat): # perform hamilton product if q2 is a quaternion
            q_n = np.zeros(4)
            q_n[0] = self.q[0]*q2.q[0] - self.q[1]*q2.q[1] - self.q[2]*q2.q[2] - self.q[3]*q2.q[3]
            q_n[1] = self.q[0]*q2.q[1] + self.q[1]*q2.q[0] + self.q[2]*q2.q[3] - self.q[3]*q2.q[2]
            q_n[2] = self.q[0]*q2.q[2] - self.q[1]*q2.q[3] + self.q[2]*q2.q[0] + self.q[3]*q2.q[1]
            q_n[3] = self.q[0]*q2.q[3] + self.q[1]*q2.q[2] - self.q[2]*q2.q[1] + self.q[3]*q2.q[0]
            return(Quat(q_n))
        else: # if q2 is a scalar:
            return(Quat(q2 * self.q))

    def inverse(self):
        mag = self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2
        return(Quat(mag * np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])))

def permute_X(X):
    restore_shape = X.shape
    X_flat = X.flatten()
    X_length = X_flat.shape[0]
    for index in range(int(X_length / 4)):

        k = 4 * index
        temp = X_flat[k + 1]
        X_flat[k + 1] = X_flat[k]
        X_flat[k] = X_flat[k + 3]
        X_flat[k + 3] = X_flat[k + 2]
        X_flat[k + 2] = temp

    X = X_flat.reshape(restore_shape)
    return(X)



# spherical interpolation, used for quaternions
def slerp(q1, q2, t):

    # check if Cos(Omega) is negative, and if so, negate one end.
    q = q2*q1.inverse()
    if (q.q[0] < 0):
        # import pdb; pdb.set_trace()
        q2_neg = Quat(-q2.q)
        return(((q2_neg*q1.inverse())**t)*q1)
    else:
        return(((q2*q1.inverse())**t)*q1)



def quat_upsampling(X, scale=4): # X is low-resolution numpy-array
        # import pdb; pdb.set_trace()

        X = permute_X(X) # convert from quaternion scalar_last convention to scalar_first.

        X_rows = X.shape[0]; X_cols = X.shape[1]
        interp_factor = scale
        X_SR = np.zeros([scale * X_rows, scale * X_cols, 4], dtype=np.float32) # Super-resolved numpy-array

        for i in range(scale * X_rows):

            for j in range(scale * X_cols):

                x1 = math.floor(j/scale)
                x2 = math.floor(j/scale) + 1
                y1 = math.floor(i/scale)
                y2 = math.floor(i/scale) + 1

                # Check algorithm didn't exit bounds of original quat matrix
                if (y2 == X.shape[0]):
                    y2 = X.shape[0] - 1
                if (x2 == X.shape[1]):
                    x2 = X.shape[1] - 1

                q1 = Quat(X[y2][x1]) # quats from LR cell
                q2 = Quat(X[y2][x2])
                q3 = Quat(X[y1][x1])
                q4 = Quat(X[y1][x2])
                t_x = (j % scale) / scale # find mod, and normalize to obtain interpolation parameter, 't'.
                q_interp_x1 = slerp(q1, q2, t_x)
                q_interp_x2 = slerp(q3, q4, t_x)
                # y-interpolation:
                t_y = (i % scale) / scale
                X_SR[i][j] = slerp(q_interp_x1, q_interp_x2, t_y).q # slerp returns Quat object, so get its numpy array.
        return(X_SR)