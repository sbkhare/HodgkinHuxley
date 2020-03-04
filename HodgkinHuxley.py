# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:20:51 2020

@author: Sikander

HODGKIN AL, HUXLEY AF. A quantitative description of membrane current and its 
application to conduction and excitation in nerve. J Physiol. 
1952;117(4):500–544. doi:10.1113/jphysiol.1952.sp004764

C.A.S. Batista, R.L. Viana, S.R. Lopes, A.M. Batista,
Dynamic range in small-world networks of Hodgkin–Huxley neurons with chemical synapses,
Physica A: Statistical Mechanics and its Applications,
Volume 410,2014, Pages 628-640, ISSN 0378-4371, 
https://doi.org/10.1016/j.physa.2014.05.069.

https://neuronaldynamics.epfl.ch/online/Ch2.S2.html#Ch2.F3:
    -The equilibrium functions x0 for gating variable x=m,n,h given by 
        x0(V) = a_x(V)/[a_x(V) + B_x(V)]
    -Graphically shown in Fig 2.3 and equation shown in boc with title Example: Time Constants, Transition Rates, and Channel Kinetics


https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4816789/:
    -we determined the synaptic conductance per synaptic contact to be 0.77 ± 0.4 nS

https://en.wikipedia.org/wiki/Axon:
    -Most individual axons are microscopic in diameter (typically about one micrometer (µm) across).
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import find_peaks

dt = 0.001

#MODEL CONSTANTS
Cm = 1.0 #uF/cm^3 : Mean = 0.91, Range = 0.8-1.5
VNa = 55#-115 #mV : Mean = -109, Range = -95 to -119
VK = -77#-72#12 # mV : Mean = 11. Range = 9-14
VL = -65#-49#10.613 # mV : Mean = -11, Range = -4 to -22
gNa = 40#120 #mS/cm^2 : Mean = 80/160, Range = 65-90/120-260
gK = 35#36 #mS/cm^2 : Mean = 34, Range = 26-49
gL = 0.3 #mS/cm^2 : DEfault = 0.3, Range = 0-3, 0-26, 13-50
gC = 0.06 #mS/cm^2 : Default = 0.06, Range = 0-0.33
tau_r = 0.5 #ms : characteristic rise time
tau_d = 8 #ms : characteristic decay time
Vsyn = 20 #mV
V0 = -20 #mV

def alpha_n(Vi):
    #return 0.01*(Vi + 50)/(1 - np.exp(-0.1*(Vi + 50)))
    return 0.02*(Vi - 25)/(1 - np.exp(-(Vi - 25)/9))

def beta_n(Vi):
    #return 0.125*np.exp((Vi+60)/80)
    return -0.002*(Vi - 25)/(1 - np.exp((Vi - 25)/9))

def alpha_m(Vi):
    #return 0.1*(Vi + 35)/(1 - np.exp(-0.1*(Vi + 35)))
    return 0.182*(Vi + 35)/(1 - np.exp(-(Vi + 35)/9))

def beta_m(Vi):
    #return 4*np.exp((Vi + 60)/18)
    return -0.124*(Vi + 35)/(1 - np.exp((Vi + 35)/9))

def alpha_h(Vi):
    #return 0.07*np.exp((Vi + 60)/20)
    return 0.25*np.exp(-(Vi + 90)/12)

def beta_h(Vi):
    #return 1/(1 + np.exp(-0.1*(Vi +30)))
    return 0.25*np.exp((Vi + 62)/6)/np.exp((Vi + 90)/12)

def x0(V, gating): #The equilibrium functions x0 for gating variable x=m,n,h
    if gating=='n':
        return alpha_n(V)/(alpha_n(V) + beta_n(V))
    elif gating=='m':
        return alpha_m(V)/(alpha_m(V) + beta_m(V))
    elif gating=='h':
        return alpha_h(V)/(alpha_h(V) + beta_h(V))
    
def show_x0():
    Vi = np.arange(-100,100,0.1)
    n0 = x0(Vi, 'n')
    m0 = x0(Vi, 'm')
    h0 = x0(Vi, 'h')
    plt.figure()
    plt.plot(Vi, n0, label='n0')
    plt.plot(Vi, m0, label='m0')
    plt.plot(Vi, h0, label='h0')
    plt.axvline(x=-63, color='r', linestyle='--', label='Resting potential')
    plt.title("Equilibrium functions of the gating variable")
    plt.xlabel("Membrane voltage (mV)")
    plt.legend()
    plt.show()
    
def adjmat(size, p=1, q=1, r=2, dim=2, show=True): #default is p=1 and r=2
    directed_sw = nx.navigable_small_world_graph(size, p, q, r, dim) #A navigable small-world graph is a directed grid with additional long-range connections that are chosen randomly.
    nx.draw_circular(directed_sw)
    aspl = nx.average_shortest_path_length(directed_sw)
    acc = nx.average_clustering(directed_sw)
    print("Average shortest path length: " + str(aspl))
    print("Average clustering coefficient: " + str(acc))
    aij = nx.to_numpy_matrix(directed_sw)
    #ADDING INHIBTORY NEURONS
    ind = [j for j in range(size**2)]
    inh = random.sample(ind, k=math.ceil(0.15 * size**2)) #15% OF NEURONS ARE INHIBITORY
    for i in inh:
        aij[i, :] *= -1
    if show:
        plt.matshow(aij)
    return aij #Returns the graph adjacency matrix as a NumPy matrix.
    

class HodgkinHuxley():
    def __init__(self, size, time):
        self.size = size
        self.time = time
        self.I = np.zeros(size) #External input currentn
        self.V = np.zeros(size) #Membrane voltage
        self.n = np.zeros(size) #K activation var.
        self.m = np.zeros(size) #NA activation var.
        self.h = np.zeros(size) #Na inactivation var.
        self.r = np.zeros(size) #Fraction of bond receptors with presynaptic neurons
        self.a = np.zeros((size,size)) #Synaptic adjacency matrix
        self.output = np.zeros((size, int(time/dt)))
        self.rt = np.zeros((size, int(time/dt)))
        #ADD TIME RECORDINGS
        
    def initialize(self, p, q, r, dim):
        self.V = 100*np.random.rand(self.size) - 80  # -63*np.ones(self.size)
        self.n = x0(self.V, 'n')
        self.m = x0(self.V, 'm')
        self.h = x0(self.V, 'h')
        self.r = np.random.rand(self.size) # (self.V + 80)/100
        self.a = adjmat(int(np.sqrt(self.size)), p, q, r, dim)
    
    def step(self, i):
        #RECORD CURRENT STATE
        self.output[:, i] = self.V
        self.rt[:, i] = self.r
        
        INa = gNa*(self.m**3)*self.h*(self.V - VNa)
        IK = gK*(self.n**4)*(self.V - VK)
        IL = gL*(self.V - VL)
        
        drdt = (1/tau_r - 1/tau_d)*(1 - self.r)/(1 + np.exp(-self.V + V0)) - self.r/tau_d
        self.r = drdt*dt + self.r
#        weight = self.r*self.a
#        weight = np.asarray(weight)
#        weight = weight.reshape((self.size,))
#        Isyn = gC*weight*(Vsyn - self.V)
        
        Isyn = gC*(Vsyn - self.V)*self.r*self.a
        Isyn = np.asarray(Isyn)
        Isyn = Isyn.reshape((self.size,))
        
        dVdt = (-INa -IK -IL + Isyn)/Cm
        dndt = -(alpha_n(self.V) + beta_n(self.V))*self.n + alpha_n(self.V)
        dmdt = -(alpha_m(self.V) + beta_m(self.V))*self.m + alpha_m(self.V)
        dhdt = -(alpha_h(self.V) + beta_h(self.V))*self.h + alpha_h(self.V)
        
        self.V = dVdt*dt + self.V
        self.n = dndt*dt + self.n
        self.m = dmdt*dt + self.m
        self.h = dhdt*dt + self.h
          
    def simulate(self):
        for t in range(int(self.time/dt)):
            self.step(t)
        
    def trace(self):
        t = np.arange(0,self.time,dt)
        plt.figure()
        plt.subplot(411)
        plt.plot(t, self.output[0,:])
        plt.title("Neuron 1")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.subplot(412)
        plt.plot(t, self.output[1,:])
        plt.title("Neuron 2")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.subplot(413)
        plt.plot(t, self.output[2,:])
        plt.title("Neuron 3")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.subplot(414)
        plt.plot(t, self.output[3,:])
        plt.title("Neuron 4")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.suptitle("Neural traces")
        plt.tight_layout()
        plt.show()
    
    def show_r(self):
        plt.figure()
        t = np.arange(0,hh.time,dt)
        for r in hh.rt:
            plt.plot(t, r)
        plt.title("r_ij(t)")
        plt.xlabel("Time (ms)")
        plt.show()
    
    def showOutput(self):
        plt.figure()
        t = np.arange(0,hh.time,dt)
        for v in hh.output:
            plt.plot(t, v)
        plt.title("V_i(t)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.show()
            
#    def findPeaks(self):
#        
#    
#    def raster(self):
#        
    
duration = 200
netsize = 25 #MUST BE A SQUARE NUMBER because of nx.navigable_small_world function  

if __name__=='__main__':
    #My suspicions (proven wrong) are for getting complex patterns: Need clustering > 0.1, Inhibitory neurons must have out-degree > 0.15*netsize, randomize initial voltages?
    #CPG like structure?
    #Take away r_ij?
    #Add stimulating current?
    hh = HodgkinHuxley(netsize, duration)
    hh.initialize(1, 3, 2.5, 2)
    hh.simulate()
    hh.showOutput()
    hh.show_r()
    #show_x0()