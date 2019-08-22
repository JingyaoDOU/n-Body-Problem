# Import libraries
import numpy as np 
from numpy import linalg as LA
from math import sqrt,pi
from scipy.optimize import minimize


global G
G = 1.0

# Define the gravitational potential energy function between two body with mass m1 and m2
def UG(r,m1,m2): 
    
    # UG(r) = -GMm/r
    gamma = G*m1*m2
    potential = -gamma*np.power(r,-1) 
    
    return potential

# The force and energy between two bodies with mass m1 and m2
# F is a (1,2) array, with F[0] represents force in x direction
# F[1] represents force in y direction
def gravi_force(q1,q2,m1,m2):
    
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
    
    F1 = np.zeros_like(q1) 
    F2 = np.zeros_like(q2)
    
    F1[0] = -du_dq1x
    F1[1] = -du_dq1y
    F2[0] = -du_dq2x
    F2[1] = -du_dq2y
    
    return energy, F1, F2 

# Given position and velocity of n bodies in the system
# Output the force on each body, potential energy and angular momentum of the system
# The structure of q is (x1,y1,x2,y2,...,xn,yn)
# q and f ara of the similar form (f1x,f1y,f2x,f2y,...,fnx,fny)

def G_force(q,p,M,N):
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


# Update the position and velocity with different integrator.

# Output new postion, velocity, total energy and angular momentum
# of the new system after time step h.

def Eulers_Method(q,p,M,N,h, force_function):
    
    # Compute the force
    pe_old, f,L_old = force_function(q,p,M,N)

    # Do the update
    qt = q + h * p 
    pt = p + h * f
    
    # Compute the new energies
    pe, f_new, L = force_function(qt,pt,M,N)
    ke = np.sum( pt*pt / 2*np.repeat(M,2)) 
    
    # Total energy is kinetic + potential
    total_e = ke + pe 
    
    # Return values 
    return qt, pt, total_e,L

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

def Leapfrog(q,p,M,N,h,force_function):
    
    qt=q+h*p
    old_pe,f,old_L=force_function(qt,p,M,N)
    pt=p+h*f
    
    #q_half=(q+qt)/2
    p_half=(p+pt)/2
    
    pe,f,L=force_function(qt,p_half,M,N)
    ke = np.sum( p_half*p_half / 2*np.repeat(M,2)) 
    
    total_e = ke + pe 
    return qt, pt, total_e,L

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
    
def RK4(q,p,M,N,h, force_function):
    
    p1=np.copy(p)
    q1=np.copy(q)
    pe1,f1,L1=force_function(q1,p1,M,N)
    
    p2=p+(h/2)*f1
    q2=q+(h/2)*p1
    pe2,f2,L2=force_function(q2,p2,M,N)
    
    p3=p+(h/2)*f2
    q3=q+(h/2)*p2
    pe3,f3,L3=force_function(q3,p3,M,N)    
    
    p4=p+h*f3
    q4=q+h*p3
    pe4,f4,L4=force_function(q4,p4,M,N)
    
    qt=q+(h/6)*(p1+2*p2+2*p3+p4)
    pt=p+(h/6)*(f1+2*f2+2*f3+f4)
    
    pe, f_new,L = force_function(qt,pt,M,N)
    ke = np.sum( pt*pt / 2*np.repeat(M,2)) 
    
    total_e = ke + pe 
    
    return qt, pt, total_e, L

def motion_simulation(q0, p0, M, Nsteps, tlim, h, step_function, force_function, adaptive=False, lamda=50):
    

    if (step_function=='Leapfrog'):
        h_ini=h/20

        N_ini=10
        for n in range(N_ini):
            Q,p0,energy,L = Eulers_Method(q0, p0,M,N, h_ini, G_force)

    N=len(M)
    q = np.copy(q0)
    p = np.copy(p0)

    # Find initial totoal energy and angular momentum
    pe_ini,f_ini,L_ini=G_force(q,p,M,N)
    ke_ini = np.sum( p*p / 2*np.repeat(M,2)) 
    e_ini=pe_ini+ke_ini
    
    # shift coordinates to the system where center of mass is fixed at the origin
    q_center=np.array([np.sum(M*q[::2]),np.sum(M*q[1::2])])/np.sum(M)
    q_tran_ini=q-np.tile(q_center,N)
    p_center=np.array([np.sum(p[::2]),np.sum(p[1::2])])/np.sum(M)        
    p_tran_ini=p-np.tile(p_center,N)
           
    t = 0

    q_traj = [q_tran_ini] 
    p_traj = [p_tran_ini] 
    e_traj = [e_ini]
    L_traj = [L_ini]
    t_traj = [0]
    
    # Main loop
    for n in range(Nsteps):
        
        t = t + h 
        if t>tlim:
            break
    
        q,p,energy,L = step_function(q, p,M,N, h, force_function)
        q_center=np.array([np.sum(M*q[::2]),np.sum(M*q[1::2])])/np.sum(M)
        q_tran=q-np.tile(q_center,N)
        p_center=np.array([np.sum(p[::2]),np.sum(p[1::2])])/np.sum(M)        
        p_tran=p-np.tile(p_center,N)

        if (adaptive):
            r=[]
            v=[]
            for i in range(N-1):
                q1=q[2*i:2*i+2]
                p1=q[2*i:2*i+2]
                for j in range (i+1,N):
                    q2=q[2*j:2*j+2]
                    p2=q[2*j:2*j+2]
                    r+=[np.linalg.norm(q1-q2)]
                    v+=[np.linalg.norm(p1-p2)]
            r=np.array(r)
            v=np.array(v)
            dh=r/v
            h=np.min(dh)/lamda

        # Save the system's data
        q_traj += [q_tran] 
        p_traj += [p_tran] 
        e_traj += [energy] 
        L_traj += [L] 
        t_traj += [t] 

    # Format into numpy arrays
    q_traj = np.array(q_traj)
    p_traj = np.array(p_traj) 
    L_traj = np.array(L_traj) 
    e_traj = np.array(e_traj) 

    # Return the trajectories
    return  q_traj, p_traj, e_traj, t_traj,L_traj

def stability_check(dx, dy, M, Nsteps,tlim, h, step_function, force_function, adaptive=True, lamda=50):

    q,p=findVV(dx,dy)

    N=len(M)

    t = 0

    q_traj = []
    D_traj = []
    
    # Main loop
    for n in range(Nsteps):
        
        t = t + h 
        if t>tlim:
            break
        # Step in time
        q,p,energy,L = step_function(q, p,M,N, h, force_function)
        
        r1=np.linalg.norm(q[0:2]-q[2:4])
        r2=np.linalg.norm(q[0:2]-q[4:6])
        r3=np.linalg.norm(q[2:4]-q[4:6])
        r_tol=r1+r2+r3
        
        D12=r1/r_tol
        D13=r2/r_tol
        D=np.array([D12,D13])
        
        dv1=np.linalg.norm(p[0:2]-p[2:4])
        dv2=np.linalg.norm(p[0:2]-p[4:6])
        dv3=np.linalg.norm(p[2:4]-p[4:6])
        
        h1=r1/dv1
        h2=r2/dv2
        h3=r3/dv3

        if (adaptive):
            h=np.min(np.array([h1,h2,h3]))/lamda
                
        rmax=np.max(np.array([r1,r2,r3]))
        if rmax>=10.0:
            break
        D_traj += [D]

    D_traj = np.array(D_traj)
    
    countbox=countBox(D_traj)

    return t,countbox

def countBox(q_traj):
    
    box=np.floor(1000*q_traj[:,:2]).astype(int)
    A=np.zeros((1001,1001))
    A[box[:,0],box[:,1]]=1
    Nbox=np.sum(A)
    return Nbox

def findVV(dx,dy):
    M=np.array([1.0,1.0,1.0])
    N=3
    q0=np.array([-0.97000436, 0.24308753,0.0,0.0, 0.97000436, -0.24308753])
    p0=np.array([0.4662036850, 0.4323657300, -0.93240737, -0.86473146, 0.4662036850, 0.4323657300])
    pe_ini,f_ini,L_ini=G_force(q0,p0,M,N)
    ke_ini = np.sum( p0*p0 / 2*np.repeat(M,2)) 
    e_ini=pe_ini+ke_ini

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

def Action_function(x,N,n,m):
    lagTime=m/N
    dt=2*pi/m
    x_traj=np.zeros((N,m))
    y_traj=np.zeros((N,m))
    vx_traj=np.zeros((N,m))
    vy_traj=np.zeros((N,m))
    for i in ( range(N)):
        for t in ( range(m)):
            x_traj[i,t]=x[0]+np.sum(x[1:2*n+1]*sincos_1(n,((t+i*lagTime)%m)*dt))
            y_traj[i,t]=x[2*n+1]+np.sum(x[2*n+2:]*sincos_1(n,((t+i*lagTime)%m)*dt))
            
    for i in ( range(N)):
        for t in ( range(m)):
            vx_traj[i,t]=x[0]+np.sum(x[1:2*n+1]*sincos_2(n,((t+i*lagTime)%m)*dt))
            vy_traj[i,t]=x[2*n+1]+np.sum(x[2*n+2:]*sincos_2(n,((t+i*lagTime)%m)*dt))
            
    K=0.5*np.sum(vx_traj**2+vy_traj**2,axis=0)
    U=0
    for i in range(N-1):
        for j in range(i+1,N):
            r=np.sqrt((x_traj[i,:]-x_traj[j,:])**2+(y_traj[i,:]-y_traj[j,:])**2)
            U=U-1/r
            
    A=np.sum(K-U)*dt
    
    return A

def sincos_1(n,t):
    return np.hstack([np.sin(np.arange(1,n+1)*t),np.cos(np.arange(1,n+1)*t)])

def sincos_2(n,t):
    return np.hstack([np.arange(1,n+1)*np.cos(np.arange(1,n+1)*t),
                      -np.arange(1,n+1)*np.sin(np.arange(1,n+1)*t)])

def FFT8(n):
    n=2*n
    q0=np.array([-0.07880227334416882, 0.5570371897354746,0.5940359608209828, 0.383319210563721,-0.5152336874768139, -0.9403564002991956])
    p0=np.array([0.15998292728488323, 1.1593418791674066,-0.5557289806160467, -0.9029539156799118,0.39574605333116347, -0.2563879634874948])
    M=np.array([1.0,1.0,1.0])

    h = 2*pi/n

    T = 2*pi
    tlim=1000
    Nsteps = n
    N=3
    
    q_traj, p_traj, e_traj, t_traj ,L_traj= motion_simulation(q0, p0,M, Nsteps,tlim, h, RK4, G_force)
    
    a2=np.real(np.fft.rfft(q_traj[:,0]))
    a1=np.imag(np.fft.rfft(q_traj[:,0]))
    
    b2=np.real(np.fft.rfft(q_traj[:,1]))
    b1=np.imag(np.fft.rfft(q_traj[:,1]))
    
    x0=np.hstack([a2[0],a1[1:],a2[1:],b2[0],b1[1:],b2[1:]])
    return x0/n

def FFT(q_traj):
    
    a2=np.real(np.fft.rfft(q_traj[:,0]))
    a1=np.imag(np.fft.rfft(q_traj[:,0]))
    b2=np.real(np.fft.rfft(q_traj[:,1]))
    b1=np.imag(np.fft.rfft(q_traj[:,1]))
    x0=np.hstack([a2[0],a1[1:],a2[1:],b2[0],b1[1:],b2[1:]])
    n=len(a1[1:])

    return x0/n

