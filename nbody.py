# Import libraries
from tqdm import tqdm
import numpy as np 
from numpy import linalg as LA
from math import sqrt,pi
from scipy.optimize import minimize

# Define the gravitational potential energy function between two body with mass m1 and m2
global G
G = 1.0

def UG(r,m1,m2): 
    
    # UG(r) = -GMm/r
    gamma = G*m1*m2
    potential = -gamma*np.power(r,-1) 
    
    return potential


def derivatives_U(q1,q2,m1,m2):
    
    r = np.linalg.norm( q1-q2 )
    
    energy = UG(r,m1,m2)
    
    q1x = q1[0] 
    q1y = q1[1]
    q2x = q2[0] 
    q2y = q2[1]
    
    gamma=G*m1*m2
    
    dU_dr = gamma*r**(-2)
    
    dnorm_dq1x = (q1x-q2x)/r 
    dnorm_dq1y = (q1y-q2y)/r  
    
    dnorm_dq2x = -(q1x-q2x)/r 
    dnorm_dq2y = -(q1y-q2y)/r
    

    du_dq1x = dU_dr * dnorm_dq1x
    du_dq1y = dU_dr * dnorm_dq1y
    
    du_dq2x = dU_dr * dnorm_dq2x
    du_dq2y = dU_dr * dnorm_dq2y
    
    return energy, du_dq1x, du_dq1y, du_dq2x, du_dq2y

# the force between to bodies with mass m1 and m2
def gravi_force(q1,q2,m1,m2):
    
    F1 = np.zeros_like(q1) 
    F2 = np.zeros_like(q2)
    
    energy, du_dq1x, du_dq1y, du_dq2x, du_dq2y = derivatives_U(q1,q2,m1,m2)
    
    F1[0] = -du_dq1x
    F1[1] = -du_dq1y
    F2[0] = -du_dq2x
    F2[1] = -du_dq2y
    
    return energy, F1, F2 

# let q be the vector containing position information of all the particles. 
# Each row repesent a particle and columns represent corresponding to x or y cooridnates 
# the last column represents the mass of that particle 

def total_force(q,p,M,N):
    pe=0
    L=0
    F=np.zeros((N,N,2))
    for i in range(N-1):
        q1=q[2*i:2*i+2]
        p1=p[2*i:2*i+2]
        m1=M[i]
        for j in range (i+1,N):
            q2=q[2*j:2*j+2]
            m2=M[j]
            e12, f12, f21 =gravi_force(q1,q2,m1,m2)
            pe=pe+e12
            F[i,j,0]=f12[0]
            F[i,j,1]=f12[1]
            F[j,i,0]=f21[0]
            F[j,i,1]=f21[1]
            
    for i in range(N):
        q1=q[2*i:2*i+2]
        p1=p[2*i:2*i+2]
        L+=np.cross(q1,p1)
            
#    pe = np.sum(E)
    f=np.sum(F,axis=1)
    f=np.reshape(f,(1,2*N))
    f=np.squeeze(f)
    #f = np.hstack(f) 
    return pe, f ,L
    
def Verlet(q,p,M,N,h, force_function):
    
    old_pe,f,old_L=force_function(q,p,M,N)
    
    pt=p+(h/2)*f
    qt=q+h*pt
    new_pe,f,new_L=force_function(qt,pt,M,N)
    pt=pt+(h/2)*f
    
    pe, f_new,L = force_function(qt,pt,M,N)
    ke = np.sum( pt*pt / 2*np.repeat(M,2)) 
    
    # Total energy is kinetic + potential
    total_e = ke + pe 
    
    # Return values 
    return qt, pt, total_e, L
    
def run_check(q0, p0,M,N, Nsteps,tlim, h, step_function, force_function):

    # Set initial conditions
    q = np.copy(q0)
    p = np.copy(p0)
           
    t = 0

    q_traj = [] 
    D_traj = []

    h_copy=np.copy(h)
    
    # Main loop
    for n in range(Nsteps):
        
        t = t + h 
        if t>tlim:
            break
        # Step in time
        q,p,energy,L = step_function(q, p,M,N, h, force_function)

        q_tran=q
        p_tran=p
        
        r1=np.linalg.norm(q_tran[0:2]-q_tran[2:4])
        r2=np.linalg.norm(q_tran[0:2]-q_tran[4:6])
        r3=np.linalg.norm(q_tran[2:4]-q_tran[4:6])
        r_tol=r1+r2+r3
        
        D12=r1/r_tol
        D13=r2/r_tol
        D=np.array([D12,D13])
        
        dv1=np.linalg.norm(p_tran[0:2]-p_tran[2:4])
        dv2=np.linalg.norm(p_tran[0:2]-p_tran[4:6])
        dv3=np.linalg.norm(p_tran[2:4]-p_tran[4:6])
        
        h1=r1/dv1
        h2=r2/dv2
        h3=r3/dv3
    
        h=np.min(np.array([h1,h2,h3]))/50
                
        rmax=np.max(np.array([r1,r2,r3]))
        
        D_traj += [D]
        
        if rmax>=10.0:
            break
      
    D_traj = np.array(D_traj)
    
    countbox=countBox(D_traj)

    return t,countbox

def countBox(q_traj):
    
    box=np.floor(1000*q_traj[:,:2]).astype(int)
    A=np.zeros((1001,1001))
    A[box[:,0],box[:,1]]=1
    Nbox=np.sum(A)
    return Nbox

def findVV(q0,p0,dx,dy,e_ini):
    
    dxy=np.hstack([dx,dy])
    q1_ini=-q0[0:2]
    q2_ini=-q1_ini

    q1_xy=q1_ini+dxy

    r=np.linalg.norm(q1_xy)

    p1_ini=p0[0:2]
    p1_norm=np.linalg.norm(p1_ini)
    px1=p1_ini[0]
    py1=p1_ini[1]

    v_norm=np.sqrt((e_ini+5/(2*r))/3)
    vx=v_norm*(px1/p1_norm)
    vy=v_norm*(py1/p1_norm)

    q_final=np.hstack([-q1_xy,0.0,0.0,q1_xy])
    p_final=np.hstack([vx,vy,-2*vx,-2*vy,vx,vy])
    
    return q_final,p_final




p0=np.array([0.4662036850, 0.4323657300, -0.93240737, -0.86473146, 0.4662036850, 0.4323657300])
q0 =np.array([-0.97000436, 0.24308753,0.0,0.0, 0.97000436, -0.24308753])
M=np.array([1.0,1.0,1.0])
N=3

pe_ini,f_ini,L_ini=total_force(q0,p0,M,N)
ke_ini = np.sum( p0*p0 / 2*np.repeat(M,2)) 
e_ini=pe_ini+ke_ini

h = 0.002
Nsteps = 500000
tlim=2000

dx2=np.linspace(-0.2,0.2,101)
dy2=np.linspace(-0.2,0.2,101)
i=0
j=0
island1=np.zeros((101,101))
island2=np.zeros((101,101))

    
for dx in tqdm( dx2 ):
    for dy in dy2:
        q_final,p_final=findVV(q0,p0,dx,dy,e_ini)
        island1[i,j],island2[i,j]=run_check(q_final, p_final,M,N, Nsteps,tlim, h, Verlet, total_force)
        j+=1
    i+=1
    j=0
    

np.save('island1.npy',island1)
np.save('island2.npy',island2)