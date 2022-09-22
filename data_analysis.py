#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':20})
plt.rc('font', family='serif')
plt.rc('text.latex')

############################################
# IMPORT DATA 
############################################

# CENOGRID (Westerhold et al. 2020) 
rawdata = np.loadtxt('isotope_data/westerhold_S33.tab',dtype=str,delimiter='\t',skiprows=92)
cenogrid = rawdata[:,[4,14,15]]
cenogrid[cenogrid == ''] = 'NaN'
cenogrid = np.flip(cenogrid.astype(float),axis=0)
cenogrid[:,0] = -cenogrid[:,0]

# Antarctic temperature stack (Parrenin et al. 2013)
rawdata = np.loadtxt('isotope_data/parrenin13_ATS.tab',dtype=str,delimiter='\t',skiprows=10)
ats = rawdata[:,:]
ats[ats == ''] = 'NaN'
ats = np.flip(ats.astype(float),axis=0)
ats[:,0] = -ats[:,0]

# LR04 stack 
rawdata = np.loadtxt('isotope_data/LR04.tab',dtype=float,delimiter='\t',skiprows=5)
lr04 = rawdata[:,:]
lr04 = np.flip(lr04.astype(float),axis=0)
lr04[:,0] = -lr04[:,0]

# huybers 06 
rawdata = np.loadtxt('isotope_data/huybers06.txt',dtype=float,delimiter='\t',skiprows=12)
h07 = rawdata[:,:]
h07 = np.flip(h07.astype(float),axis=0)
h07[:,0] = -h07[:,0]

# zachos 08
rawdata = np.loadtxt('isotope_data/zachos08.tab',dtype=str,delimiter='\t',skiprows=1)
z08 = rawdata[:,:]
z08[z08 == ''] = 'NaN'
z08 = np.flip(z08.astype(float),axis=0)
z08[:,0] = -z08[:,0]

############################################
# HAAR FLUCTUATION ANALYSIS 
############################################

def haar_v2(t,x,a,epsilon,res):
    # loosely following Lovejoy (2015)
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
    
    Dx = np.array(Dx)
    Dt = np.array(Dt)

    # average over uniform log-spaced bins
    bin_edges = np.arange(np.floor(np.log10(np.min(Dt))),np.ceil(np.log10(np.max(Dt))),1/res)
    bin_center = bin_edges[1:]-np.diff(bin_edges)
    bin_edges_real = 10**bin_edges
    n = np.histogram(Dt,bin_edges_real)[0]
    vals = np.histogram(Dt,bin_edges_real,weights=Dx)[0]/n

    # quantiles
    vals_lower = np.empty_like(vals)
    vals_upper = np.empty_like(vals)

    for i in range(0,len(bin_center)):
        temp = Dx[(Dt>bin_edges_real[i])*(Dt<bin_edges_real[i+1])]
        if len(temp)>0:
            vals_lower[i] = np.quantile(temp,0.05)
            vals_upper[i] = np.quantile(temp,0.95)
        else: 
            vals_lower[i] = np.nan
            vals_upper[i] = np.nan

    # truncate when n is a factor of 5 smaller than maximum
    inds = n>np.max(n)/5
    
    return np.sqrt(vals[inds]),10**bin_center[inds],n[inds],np.sqrt(vals_lower[inds]),np.sqrt(vals_upper[inds])

############################################
# MAIN CODE 
############################################

# put all times in same units, 
# and perform temperature scaling according to Hansen et al. (2013) (see also Materials and Methods) 
# I apologize for the repetitive code here. If you want to edit it, I recommend using find/replace features - CWA

t_ats = (ats[:,0]>-900)
t_lr = (lr04[:,0]>-6000)

cenogrid[:,0] = cenogrid[:,0]*10**6

# deep ocean temperatures
cenogrid[cenogrid[:,0]<-34*10**6,2]=-4*cenogrid[cenogrid[:,0]<-34*10**6,2]+12
cenogrid[(cenogrid[:,0]<-3.6*10**6)*(cenogrid[:,0]>-34*10**6),2]=5-8*(cenogrid[(cenogrid[:,0]<-3.6*10**6)*(cenogrid[:,0]>-34*10**6),2]-1.75)/3
cenogrid[cenogrid[:,0]>-3.6*10**6,2]=1-4.4*(cenogrid[cenogrid[:,0]>-3.6*10**6,2]-3.25)/3

# surface temperature
cenogrid[cenogrid[:,0]<-5.33*10**6,2]=cenogrid[cenogrid[:,0]<-5.33*10**6,2]+14.15
cenogrid[(cenogrid[:,0]<-1.81*10**6)*(cenogrid[:,0]>-5.33*10**6),2]=2.5*cenogrid[(cenogrid[:,0]<-1.81*10**6)*(cenogrid[:,0]>-5.33*10**6),2] + 12.15
cenogrid[cenogrid[:,0]>-1.81*10**6,2]=2*cenogrid[cenogrid[:,0]>-1.81*10**6,2]+12.25

# Note: the Huybers dataset has had long-term averages removed. For the Hansen et al. scaling to work, it needs to be offset to match the other time series (details don't matter for our results). 
h07[:,3]=h07[:,3]+4

ats[:,0] = ats[:,0]*10**3

z08[:,0] = z08[:,0]*10**6
z08[z08[:,0]<-34*10**6,1]=-4*z08[z08[:,0]<-34*10**6,1]+12
z08[(z08[:,0]<-3.6*10**6)*(z08[:,0]>-34*10**6),1]=5-8*(z08[(z08[:,0]<-3.6*10**6)*(z08[:,0]>-34*10**6),1]-1.75)/3
z08[z08[:,0]>-3.6*10**6,1]=1-4.4*(z08[z08[:,0]>-3.6*10**6,1]-3.25)/3

z08[z08[:,0]<-5.33*10**6,1]=z08[z08[:,0]<-5.33*10**6,1]+14.15
z08[(z08[:,0]<-1.81*10**6)*(z08[:,0]>-5.33*10**6),1]=2.5*z08[(z08[:,0]<-1.81*10**6)*(z08[:,0]>-5.33*10**6),1] + 12.15
z08[z08[:,0]>-1.81*10**6,1]=2*z08[z08[:,0]>-1.81*10**6,1]+12.25

lr04[:,0] = lr04[:,0]*10**3
lr04[(lr04[:,0]<-3.6*10**6)*(lr04[:,0]>-34*10**6),1]=5-8*(lr04[(lr04[:,0]<-3.6*10**6)*(lr04[:,0]>-34*10**6),1]-1.75)/3
lr04[lr04[:,0]>-3.6*10**6,1]=1-4.4*(lr04[lr04[:,0]>-3.6*10**6,1]-3.25)/3

lr04[lr04[:,0]<-5.33*10**6,1]=lr04[lr04[:,0]<-5.33*10**6,1]+14.15
lr04[(lr04[:,0]<-1.81*10**6)*(lr04[:,0]>-5.33*10**6),1]=2.5*lr04[(lr04[:,0]<-1.81*10**6)*(lr04[:,0]>-5.33*10**6),1] + 12.15
lr04[lr04[:,0]>-1.81*10**6,1]=2*lr04[lr04[:,0]>-1.81*10**6,1]+12.25

h07[:,0] = h07[:,0]*10**3
h07[(h07[:,0]<-3.6*10**6)*(h07[:,0]>-34*10**6),3]=5-8*(h07[(h07[:,0]<-3.6*10**6)*(h07[:,0]>-34*10**6),3]-1.75)/3
h07[h07[:,0]>-3.6*10**6,3]=1-4.4*(h07[h07[:,0]>-3.6*10**6,3]-3.25)/3

h07[h07[:,0]<-5.33*10**6,3]=h07[h07[:,0]<-5.33*10**6,3]+14.15
h07[(h07[:,0]<-1.81*10**6)*(h07[:,0]>-5.33*10**6),3]=2.5*h07[(h07[:,0]<-1.81*10**6)*(h07[:,0]>-5.33*10**6),3] + 12.15
h07[h07[:,0]>-1.81*10**6,3]=2*h07[h07[:,0]>-1.81*10**6,3]+12.25

# begin Haar fluctuation analysis
# begin with CENOGRID, we analyze both the whole thing and three subdivisions

res = 4 # number of bins per order of magnitude

v_cen = []
c_cen = []
n_cen = []
division_names = ['All','65-45 Ma', '45-25 Ma', '25-5 Ma']
divisions = [[-65,0],[-65,-45],[-45,-25],[-25,-5]]
a = 0.5 
for i in range(0,len(division_names)):
    t = (cenogrid[:,0]>divisions[i][0]*10**6)*(cenogrid[:,0]<divisions[i][1]*10**6)
    v,c,n,vl,vu = haar_v2(cenogrid[t,0],cenogrid[t,2],a,0.25,res)
    v_cen.append(v)
    c_cen.append(c)
    n_cen.append(n)

v_ats,c_ats,n_ats,vl_ats,vu_ats = haar_v2(ats[t_ats,0],ats[t_ats,1]/2,a,0.25,res)
v_lr04,c_lr04,n_lr04,vl_lr04,vu_lr04 = haar_v2(lr04[t_lr,0],lr04[t_lr,1],a,0.25,res)
v_h07,c_h07,n_h07,vl_h07,vu_h07 = haar_v2(h07[:,0],h07[:,3],a,0.25,res)
v_z08,c_z08,n_z08,vl_z08,vu_z08 = haar_v2(z08[:,0],z08[:,1],a,0.25,res)

###########################################
# PLOT FIGURE 3
###########################################

fig=plt.figure()
ax = plt.subplot2grid((3,1),(0,0),rowspan=2)
lw = 3
lw2 = 3
sc_al = 1
sc_size = 10
#ax.xaxis.tick_top()

# color definitions
col1 = (0.7,0,0)
col2 = (0,0,0.7)
col3 = (0.95,0.52,0.1)
col4 = (0.7,0.1,1)
col5 = (0.4,0.4,0)

wes20_col = (0,0,0)
z08_col = (0,1.00,0)
lr04_col = (0.3,0.3,1)
h06_col = (0.45,0.45,1)
pa13_col = (0.6,0.6,1)

# truncate fastest fluctuations (see discussion in Materials and Methods)
cen_ind = c_cen[0]>4000
z08_ind = c_z08>4000

wes20,=plt.plot(c_cen[0][cen_ind],v_cen[0][cen_ind],linewidth=lw,color=wes20_col,label='Westerhold et al. (2020)')
plt.scatter(c_cen[0][cen_ind],v_cen[0][cen_ind],linewidth=lw,color=wes20_col,alpha=sc_al,s=sc_size)

z08,=plt.plot(c_z08[z08_ind],v_z08[z08_ind],color=z08_col,linewidth=lw,label='Zachos et al. (2008)')
plt.scatter(c_z08[z08_ind],v_z08[z08_ind],color=z08_col,linewidth=lw,alpha=sc_al,s=sc_size)

lr04,=plt.plot(c_lr04,v_lr04,color=lr04_col,linewidth=lw,label='Lisiecki and Raymo (2005)')
plt.scatter(c_lr04,v_lr04,color=lr04_col,linewidth=lw,alpha=sc_al,s=sc_size)

h06,=plt.plot(c_h07,v_h07,color=h06_col,linewidth=lw,label='Huybers (2006)')
plt.scatter(c_h07,v_h07,color=h06_col,linewidth=lw,alpha=sc_al,s=sc_size)

ats_ind = c_ats>100
pa13,=plt.plot(c_ats[ats_ind],v_ats[ats_ind],color=pa13_col,linewidth=lw,label='Parrenin et al. (2013)')
plt.scatter(c_ats[ats_ind],v_ats[ats_ind],color=pa13_col,linewidth=lw,alpha=sc_al,s=sc_size)

ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel(r'$\Delta t$ (years)')
plt.ylabel(r'$\Delta T_{\rm rms}$ (K)')
plt.yticks([0.03,0.1,0.3,1,3,10],[0.03,0.1,0.3,1,3,10])

# plot schematic power laws
h_col = (0.5,0.5,0.5)
h_col_text = (0.4,0.4,0.4)

h_lw = 3.5

t_low = 10**(np.linspace(2.2,3.6,10))
T_low = 0.15*(t_low*10**(-2.2))**(0.6)

t_mid = [5000,500000] 
T_mid = [0.2,0.2] 

t_high = 10**(np.linspace(6.2,7.5,10))
T_high = 0.3*(t_high*10**(-6.2))**(0.5)

plt.plot(t_high,T_high,color=h_col,linewidth=h_lw,zorder=-20)   
plt.plot(t_low,T_low,color=h_col,linewidth=h_lw,zorder=-20)   
plt.plot(t_mid,T_mid,color=h_col,linewidth=h_lw,zorder=-20)   

plt.text(10**6.9,0.5,r'$H=0.5$',color=h_col_text,fontsize=20)
plt.text(10**4.7,0.125,r'$H\simeq0$?',color=h_col_text,ha='center',fontsize=20)
plt.text(10**2.5,0.7,r'$H=0.6$',color=h_col_text,fontsize=20)

plt.xlim(10**2,10**8)
plt.ylim(0.05,7)

arrow_color = (0.5,0.5,0.5)
arrow_head_width = 8 
arrow_line_width = 2

cen1_col = (0.5,0,0)
cen2_col = (0.75,0.0,0)
cen3_col = (1,0.0,0)

cen1,=plt.plot(c_cen[1][:],v_cen[1][:],linewidth=lw2,linestyle='--',label=division_names[1],color=cen1_col,zorder=-20)
cen2,=plt.plot(c_cen[2][:],v_cen[2][:],linewidth=lw2,linestyle='--',label=division_names[2],color=cen2_col,zorder=-20)
cen3,=plt.plot(c_cen[3][:],v_cen[3][:],linewidth=lw2,linestyle='--',label=division_names[3],color=cen3_col,zorder=-20)


# plot legends
lfsize = 14
first_legend = plt.legend(handles=[z08,wes20,cen1,cen2,cen3], loc='lower right',fontsize=lfsize)
ax = plt.gca().add_artist(first_legend)

plt.legend(handles=[lr04,h06,pa13], loc='upper left',fontsize=lfsize)

# plot various feedbacks, signs, and timescales

ax2 = plt.subplot2grid((3,1),(2,0),rowspan=1)
plt.xlim(10**2,10**8)
plt.ylim(0,1)

ax2.set_xscale("log")

ax2.set_yticks([])
ax2.set_xticks([])
plt.minorticks_off()

fs=16
plt.text(10**2.8,0.15,r"ocean mixing ($-$)",ha='center',va='center',fontsize=fs)
plt.text(10**2.5,0.4,r"vegetation ($+$)",ha='center',va='center',fontsize=fs)
plt.text(10**2.85,0.65,r"carbon cycle ($+,-$)",ha='center',va='center',fontsize=fs)


plt.text(2000,0.9,r"land ice sheets ($+$)",ha='center',va='center',fontsize=fs,color=lr04_col)

plt.text(10**4.2,0.40,"CaCO$_3$",ha='center',va='center',fontsize=fs)
plt.text(10**4.2,0.28,"equilibration",ha='center',va='center',fontsize=fs)
plt.text(10**4.2,0.10,r"($-$)",ha='center',va='center',fontsize=fs)

plt.text(10**5.5,0.5,r"silicate weathering ($-$)",ha='center',va='center',fontsize=fs)
plt.text(10**7,0.2,"plate tectonics",ha='center',va='center',fontsize=fs)
plt.text(10**7,0.85,"biological evolution",ha='center',va='center',fontsize=fs)

elw=2
ecs=4
plt.errorbar(2000000,0.75,xerr=[[10**6],[10**8]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)
plt.errorbar(2000000,0.1,xerr=[[10**6],[10**8]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)
plt.errorbar(50000,0.4,xerr=[[0],[2*10**6]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)

plt.errorbar(1,0.05,xerr=[[0],[3000]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)
plt.errorbar(1,0.3,xerr=[[0],[1000]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)
plt.errorbar(1,0.8,xerr=[[0],[30000]],fmt='none',ecolor=lr04_col,elinewidth=elw,capsize=ecs,zorder=-20)

plt.errorbar(5000,0.2,xerr=[[0],[45000]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)

plt.errorbar(0,0.55,xerr=[[0],[5000]],fmt='none',ecolor='black',elinewidth=elw,capsize=ecs,zorder=-20)

fig.text(0.02,0.94,r'\textbf{(a)',fontsize=20)
fig.text(0.02,0.3,r'\textbf{(b)',fontsize=20)

plt.show()
