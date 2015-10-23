# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:56:45 2015

@author: shterev
"""

import os, sys, inspect

#Add dispy to your path
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],cmd_folder+"/dispy-3.10/")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
#----------------------------------------#
#This function contains the differential equation to be simulated.    
def sim(ic,e,O): #ic=initial conditions; e=Epsiolon; O=Omega 
    from scipy.integrate import ode
    import numpy as np

    #Diff Eq.
    def sys(t,x,e,O,z,b,l):
        p = 2.*e*O*np.sin(O*t)*(1-e*np.cos(O*t))/(z+(1-e*np.cos(O*t))**2)
        q = (1+4.*b/l*np.cos(O*t))*(z+(1-e*np.cos(O*t)))/( z+(1-e*np.cos(O*t))**2 )
        dx=np.zeros(2)
        dx[0] = x[1]
        dx[1] = -q*x[0]-p*x[1]
        return dx
    #Simulation.    
    t0=0; tEnd=10000.; dt=0.1
    r = ode(sys).set_integrator('dop853', nsteps=10,max_step=dt) #Definition of the integrator
    Y=[];S=[];T=[]
    # - parameters - # 
    z=0.5; l=1.0; b=0.06;
    # -------------- #
    color=1
    r.set_initial_value(ic, t0).set_f_params(e,O,z,b,l) #Set the parameters, the initial condition and the initial time
    #Loop to integrate.
    while r.successful() and r.t +dt < tEnd:
        r.integrate(r.t+dt)
        Y.append(r.y)
        T.append(r.t)
        if r.y[0]>1.25*ic[0]: #Bound. This is due to my own requirements.
            color=0
            break
        #r.y contains the solutions and r.t contains the time vector.
    return e,O,color #For each pair e,O return e,O and a color (0,1) which correspond to the color of the point in the stability chart (0=unstable) (1=stable)
    # ------------------------------------ #

#MAIN PROGRAM where the parallel magic happens
import matplotlib.pyplot as plt
import dispy
import numpy as np
F=100 #Total files
#Range of the values of Epsilon and Omega
Epsilon = np.linspace(0,1,100)
Omega_intervals   = np.linspace(0,4,F)

ic=[0.1,0]

cluster = dispy.JobCluster(sim) #This function sets that the cluster (array of processors) will be assigned the job sim.
jobs = [] #Initialize the array of jobs

for i in range(F-1):
    Data_Array=[]
    jobs = []
    Omega=np.linspace(Omega_intervals[i], Omega_intervals[i+1],10)
    print Omega
    for e in Epsilon:
        for O in Omega:
            job = cluster.submit(ic,e,O) #Send to the cluster a job with the specified parameters
            jobs.append(job) #Join all the jobs specified above
        cluster.wait()
    #Do the jobs
    for job in jobs:
        e,O,color = job()
        Data_Array.append([e,O,color])

    #Save the results of the simulation.
    file_name='Data'+str(i)+'.txt'
    f=open(file_name, 'a')
    f.write(str(Data_Array))
    f.close()