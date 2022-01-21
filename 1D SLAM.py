import os
import math
import numpy as np
import matplotlib.pyplot as plt


# Robot states according to odometry alone
def motion_model(x, u):
    f = np.zeros(6)    
    f[0] = x[0] + u[0] - x[1]
    f[1] = x[1] + u[1] - x[2]
    f[2] = x[2] + u[2] - x[3]
    f[3] = x[3] + u[3] - x[4]
    f[4] = x[0] + u[4] - x[4]
    f[5] = x[0]
    return f

def plot_poses1d(x, x_gnd):
    font1 = {'family':'serif','color':'black','size':14}
    font2 = {'family':'serif','color':'darkred','size':15}
#     print(x)
    plt.plot(x_gnd, x_gnd, 'ro', label = "Ground Poses")

#     plt.scatter(x, x, 'g->', label = "Odometry constraints")
    plt.scatter(x, x, label = "Optimized poses")

#     plt.plot(x, x, 'c-', label = "odometry constraints")
    x_l = [x[0], x[4]]
    plt.plot(x_l, x_l, 'c-', label = "Loop constraints")
    plt.title("Plotting the poses", fontdict = font1)
    plt.legend()
    plt.show()
    
def oneD_SLAM(u, x_gnd, epochs):
    x = np.zeros(u.size) 
    for nn in range(u.size - 1):
        x[nn + 1] = x[nn] + u[nn]

    J = np.zeros((6,5))
    J[0,0] = 1
    J[0,1] = -1
    J[1,1] = 1
    J[1,2] = -1
    J[2,2] = 1
    J[2,3] = -1
    J[3,3] = 1
    J[3,4] = -1
    J[4,0] = 1
    J[4,4] = -1
    J[5,0] = 1
    
    info_mat = np.zeros((6,6))
    info_mat[0,0] = 100
    info_mat[1,1] = 100
    info_mat[2,2] = 100
    info_mat[3,3] = 100
    info_mat[4,4] = 100
    info_mat[5,5] = 1000

    plot_poses1d(x, x_gnd)
    H = J.T@info_mat@J
    for i in range(epochs):
        err = np.linalg.norm(x - x_gnd)
        print("The error in epoch number: %d is %f"%(i, err))
        f = motion_model(x, u)
        b = -J.T@info_mat.T@f
        dx = np.linalg.inv(H)@b
        x = x + dx
        plot_poses1d(x, x_gnd)
        
    return x

epochs = 10
# Measurements or observed values and the groound truth
u = np.array([1.1, 1.0, 1.1, -2.7, 0.0])
x_gnd = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
x_optim = oneD_SLAM(u, x_gnd, epochs)
print("Optimized poses:", x_optim)
