import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from scipy.constants import *
import matplotlib.ticker as mtick

NumTempRadVel_NoHeat_1dr = np.loadtxt('NumTempRadVel_NoHeat_1dr.ascii')
n_NoHeat_1dr = NumTempRadVel_NoHeat_1dr[0,:]
T_NoHeat_1dr = NumTempRadVel_NoHeat_1dr[1,:]
r_NoHeat_1dr = NumTempRadVel_NoHeat_1dr[2,:]
u_NoHeat_1dr = NumTempRadVel_NoHeat_1dr[3,:]

NumTempRadVel_NoHeat_05dr = np.loadtxt('NumTempRadVel_NoHeat_05dr.ascii')
n_NoHeat_05dr = NumTempRadVel_NoHeat_05dr[0,:]
T_NoHeat_05dr = NumTempRadVel_NoHeat_05dr[1,:]
r_NoHeat_05dr = NumTempRadVel_NoHeat_05dr[2,:]
u_NoHeat_05dr = NumTempRadVel_NoHeat_05dr[3,:]

NumTempRadVel_NoHeat_01dr = np.loadtxt('NumTempRadVel_NoHeat_01dr.ascii')
n_NoHeat_01dr = NumTempRadVel_NoHeat_01dr[0,:]
T_NoHeat_01dr = NumTempRadVel_NoHeat_01dr[1,:]
r_NoHeat_01dr = NumTempRadVel_NoHeat_01dr[2,:]
u_NoHeat_01dr = NumTempRadVel_NoHeat_01dr[3,:]

NumTempRadVel_SolarMin_1dr = np.loadtxt('NumTempRadVel_SolarMin_1dr.ascii')
n_SolarMin_1dr = NumTempRadVel_SolarMin_1dr[0,:]
T_SolarMin_1dr = NumTempRadVel_SolarMin_1dr[1,:]
r_SolarMin_1dr = NumTempRadVel_SolarMin_1dr[2,:]
u_SolarMin_1dr = NumTempRadVel_SolarMin_1dr[3,:]

NumTempRadVel_SolarMin_05dr = np.loadtxt('NumTempRadVel_SolarMin_05dr.ascii')
n_SolarMin_05dr = NumTempRadVel_SolarMin_05dr[0,:]
T_SolarMin_05dr = NumTempRadVel_SolarMin_05dr[1,:]
r_SolarMin_05dr = NumTempRadVel_SolarMin_05dr[2,:]
u_SolarMin_05dr = NumTempRadVel_SolarMin_05dr[3,:]

NumTempRadVel_SolarMin_01dr = np.loadtxt('NumTempRadVel_SolarMin_01dr.ascii')
n_SolarMin_01dr = NumTempRadVel_SolarMin_01dr[0,:]
T_SolarMin_01dr = NumTempRadVel_SolarMin_01dr[1,:]
r_SolarMin_01dr = NumTempRadVel_SolarMin_01dr[2,:]
u_SolarMin_01dr = NumTempRadVel_SolarMin_01dr[3,:]

NumTempRadVel_SolarMean_1dr = np.loadtxt('NumTempRadVel_SolarMean_1dr.ascii')
n_SolarMean_1dr = NumTempRadVel_SolarMean_1dr[0,:]
T_SolarMean_1dr = NumTempRadVel_SolarMean_1dr[1,:]
r_SolarMean_1dr = NumTempRadVel_SolarMean_1dr[2,:]
u_SolarMean_1dr = NumTempRadVel_SolarMean_1dr[3,:]

NumTempRadVel_135SolarMean_1dr = np.loadtxt('NumTempRadVel_135SolarMean_1dr.ascii')
n_135SolarMean_1dr = NumTempRadVel_135SolarMean_1dr[0,:]
T_135SolarMean_1dr = NumTempRadVel_135SolarMean_1dr[1,:]
r_135SolarMean_1dr = NumTempRadVel_135SolarMean_1dr[2,:]
u_135SolarMean_1dr = NumTempRadVel_135SolarMean_1dr[3,:]



fig = plt.plot()
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.plot(T_NoHeat_1dr, r_NoHeat_1dr, "r-", lw=10, label=''r'$\Delta$''r = 1 km')
plt.plot(T_SolarMin_1dr, r_SolarMin_1dr, "r--", lw=10, label=''r'$\Delta$''r = 1 km')
plt.plot(T_NoHeat_05dr, r_NoHeat_05dr, "b-", lw=10, label=''r'$\Delta$''r = 0.5 km')
plt.plot(T_SolarMin_05dr, r_SolarMin_05dr, "b--", lw=10, label=''r'$\Delta$''r = 0.5 km')
plt.plot(T_NoHeat_01dr, r_NoHeat_01dr, "g-", lw=10, label=''r'$\Delta$''r = 0.1 km')
plt.plot(T_SolarMin_01dr, r_SolarMin_01dr, "g-.", lw=10, label=''r'$\Delta$''r = 0.1 km')
plt.xlabel("Temperature [K]")
plt.xlim(80, 100, 10)
plt.ylabel("Radial Distance [km]")
plt.ylim(1000, 6000, 1000)
plt.legend(loc=0)
mpl.rcParams.update({'font.size':30})
plt.show()
    

fig = plt.plot()
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.plot(T_NoHeat_1dr, r_NoHeat_1dr, "r-", lw=10, label='No Heating')
plt.plot(T_SolarMin_1dr, r_SolarMin_1dr, "b--", lw=10, label='Solar Minimum')
plt.plot(T_SolarMean_1dr, r_SolarMean_1dr, "g:", lw=10, label='Solar Mean')
plt.plot(T_135SolarMean_1dr, r_135SolarMean_1dr, "m-.", lw=10, label='1.35'r'$\times$''Solar Mean')
plt.xlabel("Temperature [K]")
plt.xlim(60, 140, 10)
plt.ylabel("Radial Distance [km]")
plt.ylim(1000, 15000, 1000)
plt.legend(loc=0)
mpl.rcParams.update({'font.size':30})
plt.show()


fig1, ax1 = plt.subplots()
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
ax1.plot(T_SolarMean_1dr, r_SolarMean_1dr, "k-", lw=10, label='S-FJ')
ax1.set_xlabel("Temperature [K]")
ax1.set_xlim(0, 140, 20)
ax1.set_ylabel("Radial Distance [km]")
ax1.legend(loc=0)
ax2 = ax1.twiny()
ax2.plot(n_SolarMean_1dr / 1E9, r_SolarMean_1dr, "k-", lw=10)
ax2.set_xlabel("Number Density [m"r'$^{-3}$'"]")
ax2.set_xscale("log")
ax2.set_xlim(1E10, 1E24)
ax2.set_ylim(1000, 10000, 1000)
mpl.rcParams.update({'font.size':30})
plt.show()