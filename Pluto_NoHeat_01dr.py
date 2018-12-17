import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
import matplotlib.ticker as mtick
import time


class Pluto:
    
    dr = None
    r = None

    n = None
    u = None
    T = None

    tau = None
    Q = None


    def __init__(self, Nr = 1000000, a = 1450.0):

        self.Nr = Nr
        self.a = a
        self.dr = 0.1

        self.r = a + self.dr * np.arange(Nr+1)


    def init_cond(self, kB = Boltzmann, T0 = 88.2, M_P = 1.309E22, \
     Grav = gravitational_constant, amu = physical_constants["atomic mass constant"], pi = pi, \
     sigma_UV = 1.8E-17, sigma_EUV = 0.91E-17, X_CH4 = 0.03, X_N2 = 0.97, eps_CH4 = 0.5, eps_N2 = 0.25, mu=0.5, \
     F_UV_min=8.25E-4, F_UV_mean=1.4E-3, F_UV_max=2.4E-3, \
     F_EUV_min=1.7E-4, F_EUV_mean=2.9E-4, F_EUV_max=4.9E-4):
        
        self.kB = kB/1E6                        #Boltzmann constant via SciPy; [J/K] = [kg km^2/ s^2 K]
        self.T0 = T0                            #Isothermal Temperature; [K]
        self.G = Grav/1E9                       #Gravitational Constant via SciPy; [km^3/kg s^2]
        self.pi = pi                            #Pi via SciPy
        self.M_P = M_P                          #mass of Pluto; [kg]
        self.m_N2 = 28 * amu[0]                 #mass of Nitrogen (28 * amu -- SciPy); [kg]
        self.sigma = 1E-24                      #Johnson et al., 2015; 10^-14 cm2 --> 10^-24 [km^2]
        self.kappa = 9.37/1E8                   #conductivity given in W/m K^2 --> kg km/ s^3 K^2

        self.sigma_UV = sigma_UV/1E10           #UV absorption cross section [km^2]
        self.sigma_EUV = sigma_EUV/1E10         #EUV absorption cross section [km^2]

        self.F_UV_min = F_UV_min/1E3           #UV Energy Flux [J/ km^2 s --> J = kg m^2/s^2 --> kg km^2/s^2]
        self.F_UV_mean = F_UV_mean/1E3
        self.F_UV_max = F_UV_max/1E3

        self.F_EUV_min = F_EUV_min/1E3         #EUV Energy Flux [J/ km^2 s]
        self.F_EUV_mean = F_EUV_mean/1E3
        self.F_EUV_max = F_EUV_max/1E3

        self.eps_CH4 = eps_CH4
        self.eps_N2 = eps_N2
        self.X_CH4 = X_CH4
        self.X_N2 = X_N2
        self.mu = mu
        
        self.q_UVmin_const = self.eps_CH4 * self.X_CH4 * self.sigma_UV * self.F_UV_min
        self.q_UVmean_const = self.eps_CH4 * self.X_CH4 * self.sigma_UV * self.F_UV_mean
        self.q_UVmax_const = self.eps_CH4 * self.X_CH4 * self.sigma_UV * self.F_UV_max
        self.tau_UV_const = ( self.sigma_UV * self.X_CH4 ) / self.mu

        self.q_EUVmin_const = self.eps_N2 * self.X_N2 * self.sigma_EUV * self.F_EUV_min
        self.q_EUVmean_const = self.eps_N2 * self.X_N2 * self.sigma_EUV * self.F_EUV_mean
        self.q_EUVmax_const = self.eps_N2 * self.X_N2 * self.sigma_EUV * self.F_EUV_max
        self.tau_EUV_const = ( self.sigma_EUV * self.X_N2 ) / self.mu

        self.n = np.zeros(self.Nr+1)              #setting up array of n values
        self.n[0] = 4E27                        #No. of N2 Particles at r0; [#/km^3]

        Kn0 = (self.G * self.M_P * self.m_N2) / (np.sqrt(2) * self.n[0] * self.sigma * self.r[0] * self.r[0] * self.kB * self.T0)
        print ""
        print "Knudsen at r0:", Kn0

        self.lambda0 = (self.G * self.M_P * self.m_N2) / (self.r[0] * self.T0 * self.kB)
        print ""
        print "Lambda at r0:", self.lambda0


    def evolve(self):

        T0 = np.zeros(self.Nr+1)

        for i in range(1, self.Nr+1):

            self.n[i] = self.update_n(self.r[i], self.r[0], self.n[0])

            Kn = (self.G * self.M_P * self.m_N2) / (np.sqrt(2) * self.n[i] * self.sigma * self.r[i] * self.r[i] * self.kB * self.T0)

            if Kn>1.0:                                     #stop for loop when exobase is reached, return number density, radius, temperature

                Kn0_x = Kn
                n0_x = self.n[i]
                r0_x = self.r[i]
                T0[0:i] = self.T0
                T0_x = self.T0
                i0_x = i

                return n0_x, r0_x, T0_x, self.n[0:i], self.r[0:i], T0[0:i]


    def update_n(self, ri, r0, n0):

        return n0 * np.exp( - self.lambda0 * ( ( 1. - (r0/ri) ) ) )


    def exobase_calc(self, n_x, r_x, T_x):

        v_th = np.sqrt( (8.0 * self.kB * T_x) / (self.pi * self.m_N2) )                                                #velocity distribution at exobase

        lamb_x = (self.G * self.M_P * self.m_N2 ) / (r_x * self.kB * T_x)                                              #Jeans parameter at exobase

        phi_x = self.pi * r_x  * r_x * n_x * v_th * (1.0 + lamb_x) * np.exp( -lamb_x )                                 #Jeans molecular escape at exobase
        
        phi_E_x = self.kB * T_x * (2 + (1.0 / (1.0 + lamb_x))) * phi_x                                                 #Jeans energy flux at exobase

        return phi_x, phi_E_x, lamb_x


    def evolve_Tr(self, phix, phiEx):

        n = np.zeros(self.Nr+1)
        n[0] = 4E27 

        T = np.zeros(self.Nr+1)
        T[0] = self.T0

        u = np.zeros(self.Nr+1)
        u0 = phix / (4.0 * self.pi * self.r[0] * self.r[0] * n[0])
        u[0] = u0

        for i in range(1, self.Nr+1):

            u[i] = phix / (4.0 * self.pi * n[i-1] * self.r[i-1] * self.r[i-1])

            T[i] = self.update_Tr_RK(T[i-1], self.r[i-1], u[i], phix, phiEx)

            I = (self.dr/3.0) * ( self.Simpson1(self.r[0], T[0]) + self.Simpson1(self.r[i], T[i]) )
            
            for k in range(1, i, 2):
                I += (self.dr/3.) * 4. * self.Simpson1(self.r[k], T[k])
            for k in range(2, i, 2):
                I += (self.dr/3.) * 2. * self.Simpson1(self.r[k], T[k])

            J = (self.dr/3.0) * ( self.Simpson_FD(u[1], u[0], T[0]) + self.Simpson_BD(u[i], u[i-1], T[i]) )

            for k in range(1, i, 2):
                J += (self.dr/3.) * 4.0 * self.Simpson_CD(u[k+1], u[k], u[k-1], T[k])
            for k in range(2, i, 2):
                J += (self.dr/3.) * 2.0 * self.Simpson_CD(u[k+1], u[k], u[k-1], T[k])

            n[i] = n[0] * (T[0]/T[i]) \
                    * np.exp( - ( (self.lambda0* self.r[0]* T[0] * I )  \
                        + ( ( (0.5 * self.m_N2) /self.kB ) * J ) ) )

            Kn = (self.G * self.M_P * self.m_N2) / (np.sqrt(2) * n[i] * self.sigma * self.r[i] * self.r[i] * self.kB * T[i])
            
            if Kn>1.0:
                Kn_x = Kn
                n_x = n[i]
                r_x = self.r[i]
                T_x = T[i]
                u_x = u[i]
                i_x = i

                return n_x, r_x, T_x, n[0:i], self.r[0:i], T[0:i], u[0:i]


    def Simpson1(self,r, T):
        return 1. / (r * r * T)


    def Simpson_BD(self, uR, uR_1, TR):
        return ( (uR*uR) - (uR_1*uR_1) ) / (self.dr*TR)


    def Simpson_FD(self, u1, u0, T0):
        return ( (u1*u1) - (u0*u0) ) / (self.dr*T0)


    def Simpson_CD(self, ukplus1, uk, uk_1, Tk):
        return ( (ukplus1*ukplus1) - 2*(uk*uk) + (uk_1*uk_1) ) / (self.dr*self.dr*Tk)


    def update_Tr_RK(self, Ti_1, ri_1, ui, phix, phiE):
        kappa = self.kappa * Ti_1                               

        dT_RK = -phiE + (phix * (0.5 * self.m_N2 * ui * ui) ) \
            + (phix * 3.5 * self.kB * Ti_1) - ( (phix * self.G * self.M_P * self.m_N2)/ ri_1 ) \

        dT_RK = dT_RK/(4.0 * self.pi * ri_1 * ri_1 * kappa)

        k1 = self.dr * dT_RK
        k2 = self.dr * (dT_RK + 0.5 * k1)
        k3 = self.dr * (dT_RK + 0.5 * k2)
        k4 = self.dr * (dT_RK + k3)

        return Ti_1 + ((1./6.)*(k1 + 2*k2 + 2*k3 + k4))


    def Simpson_q(self, r, n, tau):
        return r*r*n*np.exp(-tau)


    def convergence_iteration(self, nx, rx, Tx, n, r, T, phi, phiE):

        n_x_new, r_x_new, T_x_new, n_new, r_new, T_new, u_new = self.evolve_Tr(phi, phiE)
        phi_new, phiE_new, lamb_x = self.exobase_calc(n_x_new, r_x_new, T_x_new)
        return n_x_new, r_x_new, T_x_new, n_new, r_new, T_new, u_new, phi_new, phiE_new, lamb_x

start_time = time.time()

PlutoProj = Pluto()
PlutoProj.init_cond()

n0_x, r0_x, T0_x, n0, r0, T0 = PlutoProj.evolve()
print ""
print "Kn, n, r, T at exobase (Isothermal, 0 iteration):", n0_x, r0_x, T0_x
phi_xJ_0, phi_EJx_0, lamb_x_0 = PlutoProj.exobase_calc(n0_x, r0_x, T0_x)
print ""
print "phi, phiE at exobase (Isothermal, 0 iteration):", phi_xJ_0, phi_EJx_0
print ""

n1_x, r1_x, T1_x, n1, r1, T1, u1 = PlutoProj.evolve_Tr(phi_xJ_0, phi_EJx_0)
print ""
print "n, r, T at exobase (1st iteration):", n1_x, r1_x, T1_x
phi_xJ_1, phi_EJx_1, lamb_x_1 = PlutoProj.exobase_calc(n1_x, r1_x, T1_x)
print ""
print "phi, phiE at exobase (1st iteration):", phi_xJ_1, phi_EJx_1
print ""


phi_new = phi_xJ_1
phi_old = phi_xJ_0
phiE = phi_EJx_1
n_x = n1_x
r_x = r1_x
T_x = T1_x
n = n1
r = r1
T = T1
iteration = 1

while np.fabs(100.*(phi_new-phi_old)/(phi_old))> 3.0:
    iteration = iteration + 1
    print ""
    print "Iteration:", iteration
    print ""
    print "Percentage Difference between iteration:", (100.*(phi_new-phi_old)/(phi_old))
    phi_old = phi_new
    phiE_old = phiE
    n_x_old = n_x
    r_x_old = r_x
    T_x_old = T_x
    n_old = n
    r_old = r
    T_old = T
    n_x, r_x, T_x, n, r, T, u, phi_new, phiE, lamb_x = PlutoProj.convergence_iteration(n_x_old, r_x_old, T_x_old, n_old, r_old, T_old, phi_old, phiE_old)
    if np.fabs((100.*(phi_new-phi_old)/(phi_old))) < 3.0 :
        iteration = iteration + 1
        print "iteration of convergence", iteration
        print "Final Percentage Difference:", 100.*(phi_new-phi_old)/(phi_old), 100.*(phiE-phiE_old)/(phiE_old)
        break

TIME = time.time() - start_time
print "%s seconds" % TIME

NumTempRadVel_Results = [n, T, r, u]
phiphiElamb_Results = [phi_new, phiE, lamb_x]
Exobase_Results = [n_x, r_x, T_x]
Convergence_Results = [TIME, iteration]
np.savetxt('Convergence_NoHeat_01dr.ascii', Convergence_Results)
np.savetxt('NumTempRadVel_NoHeat_01dr.ascii', NumTempRadVel_Results)
np.savetxt('phiphiElamb_NoHeat_01dr.ascii', phiphiElamb_Results)
np.savetxt('ExobaseVals_NoHeat_01dr.ascii', Exobase_Results)