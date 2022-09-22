#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')
plt.rc('text.latex')

def haar_v2(t,x,a,epsilon,res):
    # roughly following Lovejoy (2015)
    # generate data points

    # remove nan
    not_nan = ~np.isnan(x)
    t = t[not_nan]
    x = x[not_nan]
    nmax = len(t)
    x_cs = np.cumsum(x)

    Dx = []
    Dt = []
    width = np.arange(2,(nmax//2)*2,2)
    
    for iw in range(0,len(width)):
        dint = int(a*width[iw]) # spacing between successive haar fluctuations
        if dint==0:
            dint=1
    
        for i in range(0,(nmax-width[iw])//dint):
            i0 = i*dint
            tstep = t[i0+width[iw]]-t[i0]
            e = (t[i0+width[iw]//2]-t[i0])/tstep
            if e>epsilon and e<(1-epsilon): 
                Ds1 = x_cs[i0+width[iw]//2]-x_cs[i0]
                Ds2 = x_cs[i0+width[iw]]-x_cs[i0+width[iw]//2]
                Dx.append(np.abs((Ds1-Ds2)*2/width[iw])**2)
                Dt.append(tstep)
    
    
    # average over uniform log-spaced bins
    bin_edges = np.arange(np.floor(np.log10(np.min(Dt))),np.ceil(np.log10(np.max(Dt))),1/res)
    bin_center = bin_edges[1:]-np.diff(bin_edges)
    n = np.histogram(Dt,10**bin_edges)[0]
    vals = np.histogram(Dt,10**bin_edges,weights=Dx)[0]/n

    # truncate when n is a factor of 5 smaller than maximum
    inds = n>-np.max(n)/5
    
    return np.sqrt(vals[inds]),10**bin_center[inds],n[inds]

# colors
col1 = np.array([0.7,0,0])
col2 = np.array([0,0,0.7])
col3 = (0.95,0.52,0.1)
col4 = (0.0,0.7,0)
col5 = (0.4,0.4,0)

a = 1
res = 6

# IMPORT MODEL DATA FROM CSV FILES

#model4 = np.loadtxt('feedback_model_output.csv',dtype=float,delimiter=',')
#
#hf2_c1 = haar_v2(model4[:,0],model4[:,1],a,0.25,res)
#hf2_c2 = haar_v2(model4[:,0],model4[:,2],a,0.25,res)
#hf2_c3 = haar_v2(model4[:,0],model4[:,3],a,0.25,res)
#hf2_c4 = haar_v2(model4[:,0],model4[:,4],a,0.25,res)
#hf2_sum = haar_v2(model4[:,0],model4[:,1]+model4[:,2]+model4[:,3]+model4[:,4],a,0.25,res)
#
#np.save('hf_sum_2',[hf2_c1,hf2_c2,hf2_c3,hf2_c4,hf2_sum])

hf2_c1,hf2_c2,hf2_c3,hf2_c4,hf2_sum = np.load('hf_sum_2.npy')

############################################
# PLOT FIGURE 
############################################

fig = plt.figure()
ax = plt.subplot(111)
lw=3
sc_al = 1
sc_size = 10
col_sum = (0,0,0)
h_col = (0.5,0.5,0.5)
h_lw = 3

lw=3
sc_al = 1
sc_size = 10
plt.plot(hf2_c1[1],hf2_c1[0],color=col1,linewidth=lw)
plt.plot(hf2_c2[1],hf2_c2[0],color=col3,linewidth=lw)
plt.plot(hf2_c3[1],hf2_c3[0],color=col2,linewidth=lw)
plt.plot(hf2_c4[1],hf2_c4[0],color=col4,linewidth=lw)

plt.plot(hf2_sum[1],hf2_sum[0]*1.2,color=col_sum,linewidth=lw)
ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel(r'$\Delta t$ (years)',fontsize=16)
plt.yticks([0.03,0.1,0.3,1,3,10],[0.03,0.1,0.3,1,3,10])
plt.ylabel(r'$\Delta T_{\rm rms}$ (K)',fontsize=16)

t_h = 10**(np.linspace(3.6,4.8,10))
T_h = 1.1*(t_h*10**(-3.6))**(0)
plt.text(10**4.2,1.2,r'$H\simeq0$',horizontalalignment='center',color=h_col,fontsize=20)
plt.plot(t_h,T_h,color=h_col,linewidth=h_lw,zorder=-20)   

plt.text(10**6.15,0.9,'slow random walk',rotation=59,color=col4,fontsize=16)
plt.text(10**2.4,0.22,r'1 kyr feedback',rotation=54,color=col1,fontsize=16)
plt.text(10**4.5,0.6,r'10 kyr feedback',rotation=0,color=col3,fontsize=16,horizontalalignment='center')
plt.text(10**6.3,0.27,r'100 kyr feedback',rotation=-48,color=col2,fontsize=16,horizontalalignment='center')
plt.text(10**2.5,0.45,r'sum',rotation=50,color=col_sum,fontsize=16,horizontalalignment='center')

plt.ylim(0.2,3.1)
plt.xlim(10**(1.8),10**7.2)

plt.show()
