import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd
import scipy
import cv2
path = r"c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\For sharing\20200911\20200911_v_0411_data_for_frans.csv"

with open(path,'r') as f:
    df = pd.read_csv(f,skiprows=2)

x = df[['x','y','x.1','y.1','x.2','y.2']].loc

x1 = x[8551].values.reshape(3,2).T # col1 : snout, col2: lear, col3: rear
x2 = x[33165].values.reshape(3,2).T

#hom = cv2.findHomography(x1,x2)
#cv2.decomposeHomographyMat(H=cv2.UMat(hom[0]),K=cv2.UMat(np.matrix(np.identity(3))))

X1 = np.zeros((3,3))
X1[:2,:] = x1

X2 = np.zeros((3,3))
X2[:2,:] = x2

X = np.zeros((3,3))
X[0,:] = X1[:,0]
X[1,:] = (X1[:,1] + X1[:,2]) / 2
X[2,:] = np.cross(X[0,:],X[1,:])
X = X.T

Y = np.zeros((3,3))
Y[0,:] = X2[:,0]
Y[1,:] = (X2[:,1] + X2[:,2]) / 2
Y[2,:] = np.cross(X[0,:],X[1,:])
Y = Y.T

X = np.array([[1,0,0],[0,1,0],[0,0,1]])
Y = np.array([[1,0,0],[0,0.5,0],[0,0,0.5]])

B_Y = scipy.linalg.orth(Y)
B_X = scipy.linalg.orth(X)

x_prime = np.linalg.solve(B_X,X)
y_prime = np.linalg.solve(B_Y,Y)

R = np.linalg.solve(X,Y)

# B @ y = X

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


B = np.zeros((3,3))
B[0,0] = X1[0,1]
B[0,1] = X1[1,0]
B[1,0] = X1[0,2]
B[1,1] = X1[1,2]
B[2,2] = 1

C = np.zeros((3,3))



B_norm = scipy.linalg.orth(B.T)

a = np.random.random(size=(3,3))
b = np.random.random(size=(3,3))


a = np.array(([1,0,0],[0,1,0],[0,0,0]))
b = np.array(([0,1,0],[-1,0,0],[0,0,0]))

r = R.from_matrix(a)

r.as_euler('xyz',degrees=True)


A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
o = scipy.linalg.orth(A)

y = np.linalg.solve(o,A)