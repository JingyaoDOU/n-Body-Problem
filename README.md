# n-Body-simulator
Here I represent some of the python code I write for my MSc dissertation - a project about simulating n-body system and finding Choreographies orbits. 

The code will basically have two big part: one part is using different integrator to numerically solve Newtonian motion eqution in 2D and simulate the trajactory of each body in the system. The simulator can take different number of bodies, masses, integrator, step size and interaction force. Another part is used to find Choreographies, which is a special kind periodic solution to n-body problem.

## Requiements 
The examples and package make use of the following commonplace python modules:
* Python 3
* numpy 1.6+
* Scipy 1.x

## Usage 
Once you have the initial postion and velocity of a n-body system you can try the following codes to simulate the trajectory of each body in the system. First you should
```python
import numpy
```
Next, you can simulate the path by using a call like
```python
q_traj,p_traj, e_traj, t_traj ,L_traj= motion_simulation(q0, p0, M, Nsteps, h, Verlet, G_force)
```
Output _q_traj_, _p_traj_ are 2-D array tracing the position and velocity of all the particles in the system . Each row in the _q_traj_ represents the particles' postion at center time and there are in total `Nsteps` rows in the array. The structure of each row is like `(x1,y1,x2,y2,...,xn,yn)`. _p_traj's_ rows has the same structure. _e_traj_, _t_traj_, L_traj_ are the 1-D array storing the total energy, time, angualr momentum along the time. One can use the following call to plot the trajectory of each body in the system.
```python
plt.plot(q_traj[:,::2],q_traj[:,1::2])
plt.show()
```
![alt text](https://github.com/JingyaoDOU/n-Body-simulator/blob/master/git01.png)  
The full list of options for the simulator are:
```python
q_traj,p_traj, e_traj, t_traj ,L_traj=motion_simulation(q0, p0, M, Nsteps, tlim=1e10, h, step_function, force_function, adaptive=False, lamda=50)
```
where:  
   `q0`: The initial postion of all the particles in the system, having structure like (x1,y1,x2,y2,...,xn,yn)  
   `p0`: The initial velocity of all the particles in the system, having structure like             (vx1,vy1,vx2,vy2,...,vxn,vyn)  
   `M`: The array of mass of all particles in the system, having a shpe like (m1,m2,...)  
   `Nsteps`: Number of total steps for the simulation.  
   `tlim`: Time limit for simulation to stop. This parameter is used when one uses variable step size integrator and want simulates the motion of particle for a certain time.    
   `h`: The step size.  
   `step_function`: The type of integrator: Euler, Verlet, Leapfrog, RK4.  
   `force_function`: The type of interactive force between each body, for now only one type: G_force.  
   `adaptive`: Whether use variable step size method. Note can use with Leapfrog method.   
   `lamda`: The parameter control the adaptive step size. The larger lamda, the smaller the time step.
   
   

