# Pose-Graph-Optimization

## Objective
A robot is travelling in a oval trajectory. It is equipped with wheel odometry for odometry information and RGBD sensors for loop closure information. Due to noise in wheel odometry it generates a noisy estimate of the trajectory. Our task is to use loop closure pairs to correct the drift.

We pose this problem as a graph optimization problem. In our graph, poses are the vertices and constraints are the edges. 

In practical scenarios, we'd obtain the following from our sensors after some post-processing:

1. Initial position
2. Odometry Contraints/Edges: This "edge" information tells us relative transformation between two nodes. These two nodes are consecutive in the case of Odometry but not in the case of Loop Closure (next point).
3. Loop Closure Contraints/Edges: Remember that while optimizing, you have another kind of "anchor" edge as you've seen in 1. solved example.

### Method
1. Using the motion model, generate the "initialization" for all the poses/vertices using the "Given" information. Just like in the 1D case.
2. Calculate the residual and the Jacobian and update your parameters using LM.

### Observations
* As the error decreases with the incresing iterations, the plots that are plotted for every 10th iteration improves and resembles more with the ground truth trajectory.  
* Assigning hight weight to loop closure constriants may not be good even though the error is minimised in this case. However, the output path does not resemble the ground truth in the slightest.
* From the APE and RPE plots it is observed that the error increases as the information value of the odometry constraint decreases.
* Perhaps the most obvious observation is that of the increase in the error as we increase the weight of the odometry constraint. This is because the odometry is not as reliable as compared to loop closure and anchor points.
