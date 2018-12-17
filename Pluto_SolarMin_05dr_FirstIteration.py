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
        self.dr = 0.5

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
        
        self.q_EUVmin_const = self.eps_N2 * self.X_N2 * self.sigma_EUV * self.F_EUV_min
        self.q_EUVmean_const = self.eps_N2 * self.X_N2 * self.sigma_EUV * self.F_EUV_mean
        self.q_EUVmax_const = self.eps_N2 * self.X_N2 * self.sigma_EUV * self.F_EUV_max
        
        self.tau_UV_const = ( self.sigma_UV * self.X_CH4 ) / self.mu
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
                print ""
                print "Kn, n, r, T(isothermal), u at exobase:", Kn0_x, n0_x, r0_x, T0_x, 0.0    #Knudsen number, number density and radial distance of exobase
                return i0_x, n0_x, r0_x, T0_x, self.n[0:i], self.r[0:i], T0[0:i]


    def update_n(self, ri, r0, n0):

        return n0 * np.exp( - self.lambda0 * ( ( 1. - (r0/ri) ) ) )


    def exobase_calc(self, n_x, r_x, T_x):

        v_th = np.sqrt( (8.0 * self.kB * T_x) / (self.pi * self.m_N2) )                                                #velocity distribution at exobase

        lamb_x = (self.G * self.M_P * self.m_N2 ) / (r_x * self.kB * T_x)                                              #Jeans parameter at exobase
        
        print ""
        print "Lambda at exobase:", lamb_x

        phi_x = self.pi * r_x  * r_x * n_x * v_th * (1.0 + lamb_x) * np.exp( -lamb_x )                                 #Jeans molecular escape at exobase
        
        phi_E_x = self.kB * T_x * (2 + (1.0 / (1.0 + lamb_x))) * phi_x                                                 #Jeans energy flux at exobase

        print ""
        print "phi, phiE at exobase", phi_x, phi_E_x

        return phi_x, phi_E_x


    def i_Kn01(self, i_x, n, r, T):

        for i in range(1, i_x):
            Kn = (self.G * self.M_P * self.m_N2) / (np.sqrt(2) * n[i] * self.sigma * r[i] * r[i] * self.kB * T[i])

            if Kn>0.1:
                r_Kn01 = r[0:i]                                        # radius from 0 -> Kn=0.1
                n_Kn01 = n[0:i]
                return i                                    # return number of iteration to reach Kn=0.1 


    def Tau(self, i):
        
        tau_CH4 = np.zeros(i+1)
        tau_N2 = np.zeros(i+1)

        tau_CH4[i] = self.tau_UV_const * self.n[i]
        tau_N2[i] = self.tau_EUV_const * self.n[i]

        for k in range(i-1,-1,-1):

            Tau_simp = (self.dr/3.) * ( self.n[i] + self.n[k] )                 #f(a) and f(b) terms: n[r(Kn=0.1)] and n[r(k)]
            
            if i%2 !=0:                                                         #if i is odd:
                for j in range(i-2, k, -2):
                    Tau_simp += 4. * (self.dr/3.) * ( self.n[j] )
                for j in range(i-1, k, -2):                                         # even term: k-1 -> 2
                    Tau_simp += 2 * (self.dr/3.) * ( self.n[j] )
            else:                                                               #if i is even:
                for j in range(i-1, k, -2):
                    Tau_simp += 4. * (self.dr/3.) * ( self.n[j] )
                for j in range(i-2, k, -2):                                         # even term: k-2 -> 2
                    Tau_simp += 2 * (self.dr/3.) * ( self.n[j] )

            tau_CH4[k] = self.tau_UV_const * Tau_simp
            tau_N2[k] = self.tau_EUV_const * Tau_simp

        return tau_CH4, tau_N2


    def Heating(self, i, tau_CH4, tau_N2):
        
        Q_CH4 = np.zeros(i+1)
        Q_N2 = np.zeros(i+1)

        for k in range(1, i+1):
            q_r_CH4 = (self.dr/3.) * ( self.Simpson_q(self.r[0], self.n[0], tau_CH4[0]) + self.Simpson_q(self.r[k], self.n[k], tau_CH4[k]) )
            q_r_N2 = (self.dr/3.) * ( self.Simpson_q(self.r[0], self.n[0], tau_N2[0]) + self.Simpson_q(self.r[k], self.n[k], tau_N2[k]) )

            for j in range(1, k, 2):
                q_r_CH4 += 4. * (self.dr/3.) * self.Simpson_q(self.r[j], self.n[j], tau_CH4[j] )
                q_r_N2 += 4. * (self.dr/3.) * self.Simpson_q(self.r[j], self.n[j], tau_N2[j] )
            for j in range(2, k, 2):
                q_r_CH4 += 2. * (self.dr/3.) * self.Simpson_q(self.r[j], self.n[j], tau_CH4[j] )
                q_r_N2 += 4. * (self.dr/3.) * self.Simpson_q(self.r[j], self.n[j], tau_N2[j] )
                        
            Q_CH4[k] = 4. * self.pi * self.q_UVmin_const * q_r_CH4
            Q_N2[k] = 4. * self.pi * self.q_EUVmin_const * q_r_N2

        Q0_CH4 = Q_CH4[i]
        Q0_N2 = Q_N2[i]
            
        return Q0_CH4, Q_CH4, Q0_N2, Q_N2


    def evolve_Tr(self, phix, phiEx, i_Kn01, Q0_CH4, Qr_CH4, Q0_N2, Qr_N2):

        self.n = np.zeros(self.Nr+1)
        self.n[0] = 4E27 

        self.T = np.zeros(self.Nr+1)
        self.T[0] = self.T0

        self.u = np.zeros(self.Nr+1)
        u0 = phix / (4.0 * self.pi * self.r[0] * self.r[0] * self.n[0])
        self.u[0] = u0

        q = np.zeros(self.Nr+1)

        for i in range(1, self.Nr+1):

            self.u[i] = phix / (4.0 * self.pi * self.n[i-1] * self.r[i-1] * self.r[i-1])

            if i<=i_Kn01:
                q[i] = (Q0_CH4 - Qr_CH4[i]) + (Q0_N2 - Qr_N2[i])

            self.T[i] = self.update_Tr_RK(self.T[i-1], self.r[i-1], self.u[i], phix, phiEx, q[i])

            I = (self.dr/3.0) * ( self.Simpson1(self.r[0], self.T[0]) + self.Simpson1(self.r[i], self.T[i]) )
            
            for k in range(1, i, 2):
                I += (self.dr/3.) * 4. * self.Simpson1(self.r[k], self.T[k])
            for k in range(2, i, 2):
                I += (self.dr/3.) * 2. * self.Simpson1(self.r[k], self.T[k])

            J = (self.dr/3.0) * ( self.Simpson_FD(self.u[1], self.u[0], self.T[0]) \
                 + self.Simpson_BD(self.u[i], self.u[i-1], self.T[i]) )

            for k in range(1, i, 2):
                J += (self.dr/3.) * 4.0 * \
                    self.Simpson_CD(self.u[k+1], self.u[k], self.u[k-1], self.T[k])
            for k in range(2, i, 2):
                J += (self.dr/3.) * 2.0 * \
                    self.Simpson_CD(self.u[k+1], self.u[k], self.u[k-1], self.T[k])

            self.n[i] = self.n[0] * (self.T[0]/self.T[i]) \
                    * np.exp( - ( (self.lambda0* self.r[0]* self.T[0] * I )  \
                        + ( ( (0.5 * self.m_N2) /self.kB ) * J ) ) )

            Kn = (self.G * self.M_P * self.m_N2) / (np.sqrt(2) * self.n[i] * self.sigma * self.r[i] * self.r[i] * self.kB * self.T[i])
        
            if Kn>1.0:
                Kn_x = Kn
                n_x = self.n[i]
                r_x = self.r[i]
                T_x = self.T[i]
                u_x = self.u[i]
                i_x = i

                print ""
                print "Kn, n, r, T, u at exobase:", Kn_x, n_x, r_x, T_x, u_x
                return i_x, n_x, r_x, T_x, self.n[0:i], self.r[0:i], self.T[0:i], self.u[0:i]


    def Simpson1(self,r, T):
        return 1. / (r * r * T)


    def Simpson_BD(self, uR, uR_1, TR):
        return ( (uR*uR) - (uR_1*uR_1) ) / (self.dr*TR)


    def Simpson_FD(self, u1, u0, T0):
        return ( (u1*u1) - (u0*u0) ) / (self.dr*T0)


    def Simpson_CD(self, ukplus1, uk, uk_1, Tk):
        return ( (ukplus1*ukplus1) - 2*(uk*uk) + (uk_1*uk_1) ) / (self.dr*self.dr*Tk)


    def update_Tr_RK(self, Ti_1, ri_1, ui, phix, phiE, q):
        kappa = self.kappa * Ti_1                               

        dT_RK = -phiE + q + (phix * (0.5 * self.m_N2 * ui * ui) ) \
            + (phix * 3.5 * self.kB * Ti_1) - ( (phix * self.G * self.M_P * self.m_N2)/ ri_1 ) \

        dT_RK = dT_RK/(4.0 * self.pi * ri_1 * ri_1 * kappa)

        k1 = self.dr * dT_RK
        k2 = self.dr * (dT_RK + 0.5 * k1)
        k3 = self.dr * (dT_RK + 0.5 * k2)
        k4 = self.dr * (dT_RK + k3)

        return Ti_1 + ((1./6.)*(k1 + 2*k2 + 2*k3 + k4))


    def Simpson_q(self, r, n, tau):
        return r*r*n*np.exp(-tau)


start_time = time.time()

PlutoProj = Pluto()
PlutoProj.init_cond()
i0_x, n0_x, r0_x, T0_x, n0, r0, T0 = PlutoProj.evolve()
phi_xJ_0, phi_EJx_0 = PlutoProj.exobase_calc(n0_x, r0_x, T0_x)
i1_Kn01 = PlutoProj.i_Kn01(i0_x, n0, r0, T0)
tau_CH4_1, tau_N2_1 = PlutoProj.Tau(i1_Kn01)
Q0_1_CH4, Qr_1_CH4, Q0_1_N2, Qr_1_N2 = PlutoProj.Heating(i1_Kn01,tau_CH4_1, tau_N2_1)
i1_x, n1_x, r1_x, T1_x, n1, r1, T1, u1 = PlutoProj.evolve_Tr(phi_xJ_0, phi_EJx_0, i1_Kn01, Q0_1_CH4, Qr_1_CH4, Q0_1_N2, Qr_1_N2)
phi_xJ_1, phi_EJx_1 = PlutoProj.exobase_calc(n1_x, r1_x, T1_x)

TIME = time.time() - start_time
print "%s seconds" % TIME

Exobase_Vals = [TIME, i1_x, n1_x, r1_x, T1_x]
Heat_Vals = [Qr_1_N2, Qr_1_N2]
NumTempRadVel_Vals = [n1, r1, T1, u1]
PhiPhiE_01_Vals = [phi_xJ_0, phi_EJx_0, phi_xJ_1, phi_EJx_1]

np.savetxt('Exobase_SolarMin_05dr_FirstIt.ascii', Exobase_Vals)
np.savetxt('HeatVals_SolarMin_05dr_FirstIt.ascii', Heat_Vals)
np.savetxt('NumTempRadVel_SolarMin_05dr_FirstIt.ascii', NumTempRadVel_Vals)
np.savetxt('phiphiE_SolarMin_05dr_FirstIt.ascii', PhiPhiE_01_Vals)