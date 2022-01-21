#!/usr/bin/env python

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import fileinput
import jax.numpy as jnp
import jax
from jax import jacfwd, jacrev



def draw(X, Y, THETA, title):
    ax = plt.subplot(111)
    ax.plot(X, Y, 'ro')
    plt.plot(X, Y, 'c-')

    for i in range(len(THETA)):
        x2 = 0.25*math.cos(THETA[i]) + X[i]
        y2 = 0.25*math.sin(THETA[i]) + Y[i]
        plt.plot([X[i], x2], [Y[i], y2], 'g->')
        plt.plot([X[i], x2], [Y[i], y2], 'g->')
        
    plt.title(title)
    plt.show()
    
def read_VERTEX_SE2(filename):
    odometry_edges = []
    for line in fileinput.input(files=filename):
        line_parts = line.split()
        if line_parts[0] == 'VERTEX_SE2':
            edge_no = int(line_parts[1])
            X = float(line_parts[2])
            Y = float(line_parts[3])
            THETA = float(line_parts[4])
            odometry_edges.append([edge_no, X, Y, THETA])
    odometry_edges = jnp.asarray(odometry_edges)       
    return odometry_edges
    
def read_EDGE_SE2(filename):    
    odometry_edges = []
    for line in fileinput.input(files=filename):
        line_parts = line.split()
        if line_parts[0] == 'EDGE_SE2':
            edge_from = int(line_parts[1])
            edge_to = int(line_parts[2])
            dx = float(line_parts[3])
            dy = float(line_parts[4])
            dtheta = float(line_parts[5]) 
            odometry_edges.append([edge_from, edge_to, dx, dy, dtheta])
    odometry_edges = jnp.asarray(odometry_edges)
    return odometry_edges

def write_g2o(filename, data, add_file):
    with open(filename, 'w') as filehandle:
        for nn in range(data.shape[0]):
            filehandle.writelines("VERTEX_SE2 " )
            for i in range(data.shape[1]):
                if i == 0:
                    d = data[nn,i].astype(int)
                    filehandle.writelines("%s " % d)
                else:
                    filehandle.writelines("%s " % data[nn,i])
            filehandle.writelines("\n")
        filehandle.writelines([l for l in open(add_file).readlines() if "VERTEX_SE2" not in l ])
    return



# Obtaining poses
poses = read_VERTEX_SE2("../data/edges.txt")
poses_gnd = read_VERTEX_SE2("../data/gt.txt")

# Obtaining odometry and loop closure constraints
edges = read_EDGE_SE2("../data/edges.txt")
edges_gnd = read_EDGE_SE2("../data/gt.txt")

initialization = []
vertex1 = jnp.array(poses[0,:])
initialization.append(vertex1)

vertex_x = poses[0,1]
vertex_y = poses[0,2]
vertex_theta = poses[0,3]

for nn in range(edges.shape[0]):
    if abs(edges[nn, 0] - edges[nn, 1]) == 1:
        vertex_x = vertex_x + edges[nn, 2]*math.cos(vertex_theta) - edges[nn, 3]*math.sin(vertex_theta)
        vertex_y = vertex_y + edges[nn, 3]*math.cos(vertex_theta) + edges[nn, 2]*math.sin(vertex_theta)
        vertex_theta += edges[nn, 4]
        initialization.append(jnp.array([edges[nn, 1], vertex_x, vertex_y, vertex_theta]))
initialization = jnp.asarray(initialization)
write_g2o("../data/edges-poses.g2o", initialization, "../data/edges.txt")
draw(initialization[:,1], initialization[:,2], initialization[:,3], "Noisy initial trajectory")
draw(poses_gnd[:,1], poses_gnd[:,2], poses_gnd[:,3], "Ground truth trajectory")


def frobNorm(P1, P2, str1="mat1", str2="mat2"):
    jnp.set_printoptions(suppress=True)
    val = jnp.linalg.norm(P1 - P2, 'fro')
    print(f"Frobenius norm between {str1} and {str2} is: {val}")
    
def draw3(X, Y, THETA, X1, Y1, THETA1, X2, Y2, THETA2, title, title1, title2):
    fig, ax = plt.subplots(1,3)
    ax[0].plot(X, Y, 'ro')
    ax[0].plot(X, Y, 'c-')
    ax[1].plot(X1, Y1, 'ro')
    ax[1].plot(X1, Y1, 'c-')
    ax[2].plot(X2, Y2, 'ro')
    ax[2].plot(X2, Y2, 'c-')
    fig.set_figheight(5)
    fig.set_figwidth(15)
    
    for i in range(len(THETA)):
        x2 = 0.25*math.cos(THETA[i]) + X[i]
        y2 = 0.25*math.sin(THETA[i]) + Y[i]
        ax[0].plot([X[i], x2], [Y[i], y2], 'g->')
        ax[0].plot([X[i], x2], [Y[i], y2], 'g->')
        ax[0].set_title(title)
        x2 = 0.25*math.cos(THETA1[i]) + X1[i]
        y2 = 0.25*math.sin(THETA1[i]) + Y1[i]
        ax[1].plot([X1[i], x2], [Y1[i], y2], 'g->')
        ax[1].plot([X1[i], x2], [Y1[i], y2], 'g->')
        ax[1].set_title(title1)
        x2 = 0.25*math.cos(THETA2[i]) + X2[i]
        y2 = 0.25*math.sin(THETA2[i]) + Y2[i]
        ax[2].plot([X2[i], x2], [Y2[i], y2], 'g->')
        ax[2].plot([X2[i], x2], [Y2[i], y2], 'g->')
        ax[2].set_title(title2)
    plt.show() 
def draw3_(X, Y, THETA, X1, Y1, THETA1, X2, Y2, THETA2, it, title, title1, title2):
    ax = plt.subplot(111)
    ax.plot(X, Y, 'ro')
    ax.plot(X, Y, 'c-')
    
    for i in range(len(THETA)):
        x2 = 0.25*math.cos(THETA[i]) + X[i]
        y2 = 0.25*math.sin(THETA[i]) + Y[i]
        ax.plot([X[i], x2], [Y[i], y2], 'g->',  label = title)
        ax.plot([X[i], x2], [Y[i], y2], 'g->')
#         ax.set_legend(title)
        x2 = 0.25*math.cos(THETA1[i]) + X1[i]
        y2 = 0.25*math.sin(THETA1[i]) + Y1[i]
        ax.plot([X1[i], x2], [Y1[i], y2], 'r->',  label = title1)
        ax.plot([X1[i], x2], [Y1[i], y2], 'r->')
#         ax.set_legend(title1)
        x2 = 0.25*math.cos(THETA2[i]) + X2[i]
        y2 = 0.25*math.sin(THETA2[i]) + Y2[i]
        ax.plot([X2[i], x2], [Y2[i], y2], 'b->',  label = title2)
        ax.plot([X2[i], x2], [Y2[i], y2], 'b->')
    plt.show() 
    
def t2v(T):
    x = T[0,2]
    y = T[1,2]
    theta = jnp.arctan2(T[1,0], T[0,0])
    return jnp.array([x, y, theta])
def v2t(X):
    T = jnp.zeros((3,3))
    T[0,0] = jnp.cos(X[2])
    T[0,1] = -jnp.sin(X[2])
    T[0,2] = X[0]
    T[1,0] = jnp.sin(X[2])
    T[1,1] = jnp.cos(X[2])
    T[1,2] = X[1]
    T[2,2] = 1
    return T

def calc_residue(flg, vertices1, edges, vertices0):
    ver = vertices1
    r = []
    r_x = []
    r_y = []
    r_theta = []
    if(flg == 1):
        ver = jnp.vstack((vertices0, ver))
        ver = ver.reshape((4,120))
        ver = ver.T
   
    for nn in range(edges.shape[0]):
        edge_from = edges[nn,0]
        edge_to = edges[nn,1]
        dx = edges[nn,2]
        dy = edges[nn,3]
        dtheta = edges[nn,4]
        theta = ver[int(edge_from),3]
        
        res_x = ver[int(edge_from),1] + dx*jnp.cos(theta) - dy*jnp.sin(theta) - ver[int(edge_to),1]  
        res_y = ver[int(edge_from),2] + dy*jnp.cos(theta) + dx*jnp.sin(theta) - ver[int(edge_to),2] 
        res_theta = ver[int(edge_from),3] + dtheta - ver[int(edge_to),3]
        r_x.append(res_x)
        r_y.append(res_y)
        r_theta.append(res_theta)
    r_x.append(ver[0,1]-(-5))  
    r_y.append(ver[0,2]-(-8)) 
    r_theta.append(ver[0,3]-(0)) 
    r_ = [*r_x, *r_y, *r_theta]
    r_ = jnp.asarray(r_)
    return r_   

def calc_J(vertices, edges):
    J_x = []
    J_y = []
    J_theta = []
    for nn in range(edges.shape[0]):
        edge_from = edges[nn,0]
        edge_to = edges[nn,1]
        dx = edges[nn,2]
        dy = edges[nn,3]
        dtheta = edges[nn,4]
        theta = vertices[int(edge_from),3]
        
        Jx = jnp.zeros(360)
        Jx = jax.ops.index_update(Jx, int(edge_from), 1)
        Jx = jax.ops.index_update(Jx, int(edge_to), -1)
        Jx = jax.ops.index_update(Jx, 240 + int(edge_from), -dx*jnp.sin(theta) - dy*jnp.cos(theta))
        Jy = jnp.zeros(360)
        Jy = jax.ops.index_update(Jy, int(edge_from) + 120, 1)
        Jy = jax.ops.index_update(Jy, int(edge_to) + 120, -1)
        Jy = jax.ops.index_update(Jy, 240 + int(edge_from), -dy*jnp.sin(theta) + dx*jnp.cos(theta))
        Jtheta = jnp.zeros(360)
        Jtheta = jax.ops.index_update(Jtheta, int(edge_from) + 240, 1)
        Jtheta = jax.ops.index_update(Jtheta, int(edge_to) + 240, -1)
        J_x.append(Jx)
        J_y.append(Jy)
        J_theta.append(Jtheta)
    Jx = jnp.zeros(360)
    Jx = jax.ops.index_update(Jx, 0, 1)  
    Jy = jnp.zeros(360)
    Jy = jax.ops.index_update(Jy, 120, 1)  
    Jtheta = jnp.zeros(360)
    Jtheta = jax.ops.index_update(Jtheta, 240, 1)  
    J_x.append(Jx)  
    J_y.append(Jy) 
    J_theta.append(Jtheta) 
    J = [*J_x, *J_y, *J_theta]
    J = jnp.asarray(J)
#     print("Size-jacobian: ", J.shape)
    return J    

def info_mat(infoVal):
    indices = jnp.array([0,140,280])
    info_mat = []
    for val in indices:
        for i in range(119):
            row = jnp.zeros(420)
            row = jax.ops.index_update(row, val + i, infoVal[0])
            info_mat.append(row)
        for i in range(119,139):
            row = jnp.zeros(420)
            row = jax.ops.index_update(row, val + i, infoVal[1])
            info_mat.append(row)
        row = jnp.zeros(420)
        row = jax.ops.index_update(row, val + 139, infoVal[2])
        info_mat.append(row)    
    info_mat = jnp.asarray(info_mat)
    return info_mat
def JAXSjacobian(vertices, edges):
    W = jnp.vstack((vertices[:,1], jnp.vstack((vertices[:,2], vertices[:,3]))))
    f = lambda W:calc_residue(1,W,edges,vertices[:,0])
    J = jacrev(f)(W)
    J = J.reshape((420,360))
    return J 


# In[9]:


def calculate_J_F_Levenberg_Marquardt(J, r, info_mat, ld):
    return jnp.linalg.inv(J.T@info_mat@J + ld*jnp.eye(J.shape[1]))@J.T@info_mat.T@r

def calculate_J_F_Levenberg_Marquardt_Modified(J, r, info_mat, ld):
    H = J.T@info_mat@J
    identit = jnp.eye(H.shape[0],H.shape[1])
    for j in range(identit.shape[0]):
            identit = jax.ops.index_update(identit,jax.ops.index[j,j],H[j,j])
    modified = H + ld*identit
#     modified = H + identit

    b = J.T@info_mat@r
    J = jnp.linalg.inv(modified)@b
    return J

def Levenberg_Marquardt(vertices, edges, ld, num_iter, tol, infoVal):
    k = vertices
    ini = vertices
    stop_iter = num_iter 
    scale_factor = 10
    sigma = info_mat(infoVal)
#     print("sigma: ", sigma.shape)
#     print(sigma)
    for i in range(num_iter):
        
#       Finding the residue/cost
        r = calc_residue(0, k, edges, vertices[:,0])
#         print(r)
#         print("r: ", r.shape)
        F = (0.5)*r.T@sigma@r
#         print("F: ", F)
        cost = F
        
#       Calculating the jacobian
        J_r = calc_J(k, edges)
#         print(J_r.shape)
        J_jaxjacob = JAXSjacobian(k, edges)
#         print(J_jaxjacob)
        frobNorm(J_r, J_jaxjacob, "jax jacobian", "analytically calculated jacobian")
#         print(J_r)
#         print("J_r: ", J_r.shape)
        J_F = calculate_J_F_Levenberg_Marquardt_Modified(J_r, r, sigma, ld)
#         print(J_F)
#         print("J_F: ", J_F.shape)
        dk = -J_F
#         print(dk)
#         dk = dk.reshape((3,120))
#       Performing the update  
#         print(k[:, 1:].shape)
#         k[:, 1:] = k[:, 1:] + dk.reshape((3,120)).T
        k = k.at[:, 1:].set(k[:, 1:] + dk.reshape((3,120)).T)
#         print(k)
        r_new = calc_residue(0, k, edges, vertices[:,0])
        cost_new = (0.5)*r_new.T@sigma@r_new
#         print(cost_new)
        if(cost_new < cost):
            print("The error after iteration :%d is %f"%(i+1,cost_new))
            ld = ld/scale_factor  
        else: 
            print("The error after iteration :%d is %f"%(i+1,cost))
            k = k.at[:, 1:].set(k[:, 1:] - dk.reshape((3,120)).T)
            ld = ld*scale_factor
        if((i + 1)%10 == 0):    
            draw3_(k[:,1], k[:,2], k[:,3], ini[:,1], ini[:,2], ini[:,3], poses_gnd[:,1], poses_gnd[:,2], poses_gnd[:,3], i+1, "Initial Noisy trajectory", "Optimized Trajectory", "Ground truth trajectory")    
        if(jnp.linalg.norm(dk) < tol):
            print("dk: ", jnp.linalg.norm(dk))
            stop_iter = i + 1
            break
            
    return k, cost, stop_iter 


ld  = 0.01
num_iter = 50
tol = 1e-15
infoVal = [25, 250, 500]
optim_poses, cost, stop_iter = Levenberg_Marquardt(initialization, edges, ld, num_iter, tol, infoVal)

write_g2o("../data/edges-poses_.g2o", optim_poses, "../data/edges.txt")
