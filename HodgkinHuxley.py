0# -*- coding: utf-8 -*-
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
    -Stimulating current: "In order to explore a more realistic input scenario, 
    we stimulate the Hodgkin-Huxley model by a time-dependent input current I(t) that is generated by the following procedure. 
    Every 2 ms, a random number is drawn from a Gaussian distribution with zero mean and standard deviation σ=34μA/cm2. 
    To get a continuous input current, a linear interpolation was used between the target values."
https://neuronaldynamics.epfl.ch/online/Ch3.S1.html:
    -Synapse model

Inhibitory neurons:
    -https://www.cell.com/current-biology/pdf/S0960-9822(09)00796-9.pdf: 15% of mammalian coritcal neurons are inhibitory
    -https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4405698/; %age measure in mammals is 10-30%, optimal is 30%
    -https://www.ncbi.nlm.nih.gov/pubmed/19965761: GABAergic (inhibitory) neurons have "widespread axonal arborizations" i.e. have high out-degree

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
gC = 0.04 #mS/cm^2 : Default = 0.06, Range = 0-0.33
tau_r = 0.5 #ms : characteristic rise time
tau_d = 8 #ms : characteristic decay time, Default=8
alpha = 6.25 #ms^-1
tau_syn = 100 #ms
Vsyn = 20 #mV
V0 = -20 #mV
Esyn = -75 #mV: use for inhibitory neurons
injection_interval = 2 #ms: time between random sample of current injections

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

def F(Vpre, theta=0):
    return 1/(1 + np.exp(-(Vpre - theta)/2))
    
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
    
def is_square(integer):
    root = math.sqrt(integer)
    return integer == int(root + 0.5) ** 2
    
def adjmat(size, p=1, q=1, r=2, dim=2, inhib=0.15, show=True): #CHANGE SO THAT HIGHEST OUT-DEGREE NEURONS ARE INHIBITORY 
    directed_sw = nx.navigable_small_world_graph(size, p, q, r, dim) #A navigable small-world graph is a directed grid with additional long-range connections that are chosen randomly. 
    nx.draw_circular(directed_sw)
    aspl = nx.average_shortest_path_length(directed_sw)
    acc = nx.average_clustering(directed_sw)
    print("Average shortest path length: " + str(aspl))
    print("Average clustering coefficient: " + str(acc))
    aij = nx.to_numpy_matrix(directed_sw)
    #ADDING INHIBTORY NEURONS
    if dim == 1:
        ind = [j for j in range(size)]
        inh = random.sample(ind, k=math.ceil(inhib * size)) #CHANGE SO THAT HIGHEST OUT-DEGREE NEURONS ARE INHIBITORY 
    elif dim == 2:
        ind = [j for j in range(size**2)]
        inh = random.sample(ind, k=math.ceil(inhib * size**2)) #CHANGE SO THAT HIGHEST OUT-DEGREE NEURONS ARE INHIBITORY 
    for i in inh:
        aij[i, :] *= -1
    if show:
        plt.matshow(aij)
    return aij #Returns the graph adjacency matrix as a NumPy matrix.
    

class HodgkinHuxley():
    def __init__(self, size, time):
        self.size = size
        self.time = time
        self.V = np.zeros(size) #Membrane voltage
        self.n = np.zeros(size) #K activation var.
        self.m = np.zeros(size) #NA activation var.
        self.h = np.zeros(size) #Na inactivation var.
        self.r = np.zeros(size) #Fraction of bond receptors with presynaptic neurons
        self.s = np.zeros(size) #Gating variable represents fraction of docked synaptic neurotransmitters
        self.a = np.zeros((size,size)) #Synaptic adjacency matrix
        self.Input = np.zeros((size, int(time/dt))) #External input currentn
        self.output = np.zeros((size, int(time/dt)))
        self.rt = np.zeros((size, int(time/dt)))
        self.st = np.zeros((size, int(time/dt)))
        self.It = np.zeros((size, int(time/dt)))
                
    def initializeRand(self, gating=False):
        self.V = 100*np.random.rand(self.size) - 80  # -63*np.ones(self.size)
        if gating:
            self.n = np.random.rand(self.size)
            self.m = np.random.rand(self.size)
            self.h = np.random.rand(self.size)
        else:
            self.n = x0(self.V, 'n')
            self.m = x0(self.V, 'm')
            self.h = x0(self.V, 'h')
    
    def initializeEquil(self):
        self.V = -63*np.ones(self.size)
        self.n = x0(self.V, 'n')
        self.m = x0(self.V, 'm')
        self.h = x0(self.V, 'h')
        
    def assign_r(self, r):
        self.r = r
        self.n = x0(self.V, 'n')
        self.m = x0(self.V, 'm')
        self.h = x0(self.V, 'h')
        
    def assignAdjMat(self, a):
        self.a = a
#        plt.figure()
        plt.matshow(a)
        plt.show()
    
    def initializeSWNet(self, p, q, r, dim, inhib):
        self.V = 100*np.random.rand(self.size) - 80  # -63*np.ones(self.size)
        self.n = x0(self.V, 'n')
        self.m = x0(self.V, 'm')
        self.h = x0(self.V, 'h')
        self.r = ((1/tau_r - 1/tau_d)/(1 + np.exp(-self.V + V0)))/((1/tau_r - 1/tau_d)/(1 + np.exp(-self.V + V0)) + 1/tau_d) #np.random.rand(self.size) 
        self.s = np.random.rand(self.size)
        if dim == 1:
            self.a = adjmat(self.size, p, q, r, dim, inhib)
        elif dim == 2:
            if is_square(self.size):
                self.a = adjmat(int(math.sqrt(self.size)), p, q, r, dim, inhib)
            else:
                raise Exception("Size must be square number if dim = 2")
            
        else:
            raise Exception("dim too high, choose dim < 2")
        
    def inputCurrent(self, injection_interval=2, num_neurons=2, show=False):
        num_injections = int(int(self.time/dt) / int(injection_interval/dt))
        t = np.arange(0, self.time, dt)
        ind = [j for j in range(self.size)]
        inp_ind = random.sample(ind, k=num_neurons)
        for i in range(self.Input.shape[0]):
#            if i in inp_ind:
            t_inj = np.arange(0, duration, injection_interval)
            injections = np.random.normal(0, 3.3, size=num_injections)
            interp = np.interp(t, t_inj, injections)
            self.Input[i, :] = interp
        if show:
            plt.figure()
            for I_i in self.Input:
                plt.plot(t, I_i)
            plt.title("Input currents")
            plt.xlabel("Time (ms)")
            plt.ylabel("Current density (μA/cm^2)")
            plt.show()
        return inp_ind
        
    def step(self, i):
        #RECORD CURRENT STATE
        self.output[:, i] = self.V
        self.rt[:, i] = self.r
        self.st[:, i] = self.s
        
        INa = gNa*(self.m**3)*self.h*(self.V - VNa)
        IK = gK*(self.n**4)*(self.V - VK)
        IL = gL*(self.V - VL)
        
        drdt = (1/tau_r - 1/tau_d)*(1 - self.r)/(1 + np.exp(-self.V + V0)) - self.r/tau_d
        self.r = drdt*dt + self.r
#        dsdt = alpha*F(self.V)*(1 - self.s) - self.s/tau_syn
#        self.s = dsdt*dt + self.s
        
#        weight = self.r*self.a
#        weight = np.asarray(weight)
#        weight = weight.reshape((self.size,))
#        Isyn = gC*weight*(Vsyn - self.V)
        
#        a = np.asarray(self.a)
#        w = (a.T * self.r).T
#        colsums = w.sum(axis=0)
#        w = w/colsums[np.newaxis, :]
#        w = np.asmatrix(w)
#        Isyn = gC*(Vsyn - self.V)*w
        
        Isyn = gC*(Vsyn - self.V)*self.r*self.a #gC*(self.V - Esyn)*self.r*self.a
        
#        Isyn = gC*(self.V - Esyn)*self.s*self.a
        
        Isyn = np.asarray(Isyn)
        Isyn = Isyn.reshape((self.size,))
        self.It[:, i] = -INa - IK - IL + Isyn + self.Input[:, i]
        
        dVdt = (-INa -IK -IL + Isyn + self.Input[:, i])/Cm
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
        
    def trace(self, indlst, thresh_lst):
        t = np.arange(0,self.time,dt)
        plt.figure()
        plt.subplot(411)
        plt.plot(t, self.output[indlst[0],:])
        plt.axhline(y=thresh_lst[indlst[0]], color='r', linestyle='--')
        plt.title("Neuron {0}".format(indlst[0]))
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.subplot(412)
        plt.plot(t, self.output[indlst[1],:])
        plt.axhline(y=thresh_lst[indlst[1]], color='r', linestyle='--')
        plt.title("Neuron {0}".format(indlst[1]))
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.subplot(413)
        plt.plot(t, self.output[indlst[2],:])
        plt.axhline(y=thresh_lst[indlst[2]], color='r', linestyle='--')
        plt.title("Neuron {0}".format(indlst[2]))
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.subplot(414)
        plt.plot(t, self.output[indlst[3],:])
        plt.axhline(y=thresh_lst[indlst[3]], color='r', linestyle='--')
        plt.title("Neuron {0}".format(indlst[3]))
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.tight_layout()
        plt.show()
        
#        if len(indlst) != row*col:
#            raise Exception("Number of neurons doesn't match rows and columns")
#        fig, ax = plt.subplots(row, col)
#        for i in range(len(indlst)):
#            ax[i//row, i%col].plot(t, self.output[i,:])
#            ax[i//row, i%col].set_title("Neuron {0}".format(i))
#            ax[i//row, i%col].set_xlabel("Time (ms)")
#            ax[i//row, i%col].set_ylabel("Voltage (mV)")
#        plt.tight_layout()
#        plt.show()
    
    def show_r(self):
        plt.figure()
        t = np.arange(0,self.time,dt)
        for r in self.rt:
            plt.plot(t, r)
        plt.title("r_ij(t)")
        plt.xlabel("Time (ms)")
        plt.show()
        
#    def show_s(self):
#        plt.figure()
#        t = np.arange(0,self.time,dt)
#        for r in self.rt:
#            plt.plot(t, r)
#        plt.title("s_ij(t)")
#        plt.xlabel("Time (ms)")
#        plt.show()
    
    def showOutput(self):
        #Display spikes as well
        plt.figure()
        t = np.arange(0,self.time,dt)
        for v in self.output:
            plt.plot(t, v)
        plt.title("V_i(t)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.show()
        
    def showCurrent(self):
        #Display spikes as well
        plt.figure()
        t = np.arange(0,self.time,dt)
        for Ii in self.It:
            plt.plot(t[500:], Ii[500:])
        plt.title("I_i(t)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (nA)")
        plt.show()
            
    def findSpikes(self, hertz=5000, save=False, show=False):
        peaks = {}
        neuron = 0
        threshlst = []
        for o in self.output:
            thresh = np.mean(o) + 2*np.std(o)
            threshlst.append(thresh)
            pks_ni = find_peaks(o, height=thresh) #Peaks of neuron i
            peaks[neuron] = pks_ni[0]
            neuron += 1
        binsize = int(1000/hertz/dt)
        samples = int(self.time/dt/binsize)
        boolean_output = np.zeros((self.size, samples))
        for node in peaks:
            for spike in peaks[node]:
                boolean_output[node, spike//binsize] = 1
        if show:
            p = []
            for nd in peaks:
                pk_times = peaks[nd]*dt
                p.append(pk_times)
            plt.figure()
            plt.eventplot(p, color=[0,0,0])
            plt.title("Raster Plot")
            plt.xlabel("Time (ms)")
            plt.ylabel("Neuron")
            plt.show()
        if save:
            np.save('boolean_output.npy', boolean_output)
        return peaks, boolean_output, threshlst
#    
#    def raster(self):
#        
    
duration = 3000
netsize = 20 

if __name__=='__main__':
    hh = HodgkinHuxley(netsize, duration)
    hh.initializeSWNet(p=2, q=1, r=0.5, dim=1, inhib=0.15)
    inp_ind = hh.inputCurrent()
    hh.simulate()
#    hh.showOutput()
#    hh.show_r()
#    hh.showCurrent()
    peaks, out, thresh_lst = hh.findSpikes(show=True)
    #show_x0()