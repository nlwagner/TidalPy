#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:33:44 2024

@author: nlwagner25
"""

import numpy as np
import matplotlib.pyplot as plt

from TidalPy.RadialSolver import radial_solver
from TidalPy.utilities.spherical_helper import calculate_mass_gravity_arrays
from TidalPy.utilities.graphics.multilayer import yplot

# Purely Elastic Body
from TidalPy.rheology import Elastic

elastic_rheology = Elastic()

int_model = 'models/AK_med.txt'
# int_model = 'mars_MD.txt'

r,vp,vs,rho = np.loadtxt(int_model,usecols=(0,1,2,3),delimiter=None,unpack=True)
    
[r,ind,inv] = np.unique(r,True,True)
vp = vp[ind]
vs = vs[ind]
rho = rho[ind]

# if (r[0] < r[-1]):
#     print(":: Reversing model to start from core")
#     r = r[::-1]
#     vp = vp[::-1]
#     vs = vs[::-1]
#     rho = rho[::-1]

# Convert to S.I. Units
r = np.multiply(r,1000.)
vp = np.multiply(vp,1000.)
vs = np.multiply(vs,1000.)
rho = np.multiply(rho,1000.)

# Convert Seismic Velocities to Elastic Moduli
mu = np.multiply(np.square(vs),rho)
K  = np.subtract(np.multiply(np.square(vp),rho),mu*(4./3.))
lam = np.subtract(K, mu*(2./3.))

visc = 1E30
eta   = np.full(len(mu),visc)

#%%

# Define planetary parameters
planet_mass = 6.417e23
planet_radius = 3.39e6
planet_volume = (4. / 3.) * np.pi * (planet_radius**3)
planet_bulk_density = planet_mass / planet_volume

# forcing_frequency = 2. * np.pi / (687* 24.6 * 60 * 60)  # Phobos orbital freq


N = 2000

radius_array = np.ones((len(r)-1)*N)
shear_array = np.ones((len(r)-1)*N)
viscosity_array = np.ones((len(r)-1)*N)
bulk_mod_array = np.ones((len(r)-1)*N)
density_array = np.ones((len(r)-1)*N)

for i in range(0,len(r)-1):
    
    

    radius_array[i*N:(i+1)*N] = np.linspace(r[i],r[i+1],N)#np.concatenate(
        # (
        # np.linspace(0.1, ICB, N),
        # np.linspace(ICB, CMB, N+1)[1:],
        # np.linspace(CMB, lyr1, N+1)[1:],
        # np.linspace(lyr1, lyr2, N+1)[1:],
        # np.linspace(lyr2, lyr3, N+1)[1:],
        # np.linspace(lyr3, planet_radius, N+1)[1:]
        # )
    # )
    shear_array[i*N:(i+1)*N] = np.linspace(mu[i],mu[i+1],N)#np.concatenate(
    #     (
    #     8.74891794e+10 * np.ones(N, dtype=np.float64),
    #     0.0 * np.ones(N, dtype=np.float64),
    #     1.35859845e+11 * np.ones(N, dtype=np.float64),
    #     1.10990570e+11 * np.ones(N, dtype=np.float64),
    #     6.86982914e+10 * np.ones(N, dtype=np.float64),
    #     4.39284066e+10 * np.ones(N, dtype=np.float64),
    #     )
    # )
    viscosity_array[i*N:(i+1)*N] = np.linspace(eta[i],eta[i+1],N)#np.concatenate(
    #     (
    #     1.0e14 * np.ones(N, dtype=np.float64),
    #     1.0e20 * np.ones(N, dtype=np.float64),
    #     1.0e20 * np.ones(N, dtype=np.float64),
    #     1.0e20 * np.ones(N, dtype=np.float64),
    #     1.0e20 * np.ones(N, dtype=np.float64),
    #     1.0e20 * np.ones(N, dtype=np.float64),
    #     )
    # )
    bulk_mod_array[i*N:(i+1)*N] = np.linspace(K[i],K[i+1],N)#np.concatenate(
    #     (
    #     1.92228327e+11 * np.ones(N, dtype=np.float64),
    #     2.74228445e+11 * np.ones(N, dtype=np.float64),
    #     2.39757054e+11 * np.ones(N, dtype=np.float64),
    #     2.06221772e+11 * np.ones(N, dtype=np.float64),
    #     1.78498479e+11 * np.ones(N, dtype=np.float64),
    #     1.11196235e+11 * np.ones(N, dtype=np.float64),
    #     )
    # )
    density_array[i*N:(i+1)*N] = np.linspace(rho[i],rho[i+1],N)#np.concatenate(
    #     (
    #     7266. * np.ones(N, dtype=np.float64),
    #     7036. * np.ones(N, dtype=np.float64),
    #     4220. * np.ones(N, dtype=np.float64),
    #     4007. * np.ones(N, dtype=np.float64),
    #     3599. * np.ones(N, dtype=np.float64),
    #     2850. * np.ones(N, dtype=np.float64),
    #     )
    # )

volume_array, mass_array, gravity_array = \
    calculate_mass_gravity_arrays(radius_array, density_array)
    
    
Nf = 100    
freqs = np.logspace(1,5,Nf) 

hp_i_1 = np.zeros(Nf)
hl_i_1 = np.zeros(Nf)
hp_c_1 = np.zeros(Nf)
hl_c_1 = np.zeros(Nf) 


hp_i_2 = np.zeros(Nf)
hl_i_2 = np.zeros(Nf)
kp_i_2 = np.zeros(Nf)
kl_i_2 = np.zeros(Nf)
kp_c_2 = np.zeros(Nf)
hp_c_2 = np.zeros(Nf)
kl_c_2 = np.zeros(Nf)
hl_c_2 = np.zeros(Nf) 
    
    
for i in range(0,Nf):
    forcing_frequency = 2. * np.pi / (freqs[i] * 60 * 60)  # Phobos orbital freq
    # Calculate the "complex" shear (really all Im[mu] = 0)
    complex_shear = np.empty(radius_array.shape, dtype=np.complex128)
    elastic_rheology.vectorize_modulus_viscosity(forcing_frequency, shear_array, viscosity_array, complex_shear)
    





    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear,
            forcing_frequency,
            planet_bulk_density,
            layer_types=("solid","liquid","solid","solid"),
            is_static_by_layer=(False,True,False,False),
            is_incompressible_by_layer=(True,True,True,True),
            upper_radius_by_layer=(3000, 1876E3, 2500E3,3392E3),
            degree_l=1,
            solve_for=('tidal','loading'),
            use_kamata=True,
            integration_method='DOP853',
            integration_rtol = 1.0e-11,
            integration_atol = 1.0e-11,
            scale_rtols_by_layer_type = False,
            max_num_steps = 10_000_000,
            expected_size = 500,
            max_ram_MB = 500,
            max_step = 0,
            limit_solution_to_radius = True,
            nondimensionalize = True,
            verbose = False,
            raise_on_fail = True
            )


    hp_i_1[i] = radial_solution.h[0]-radial_solution.k[0]
    hl_i_1[i] = radial_solution.h[1]-radial_solution.k[1]

        
    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear,
            forcing_frequency,
            planet_bulk_density,
            layer_types=("solid","liquid","solid","solid"),
            is_static_by_layer=(False,True,False,False),
            is_incompressible_by_layer=(False,False,False,False),
            upper_radius_by_layer=(3000, 1876E3, 2500E3,3392E3),
            degree_l=1,
            solve_for=('tidal','loading'),
            use_kamata=False,
            integration_method='DOP853',
            integration_rtol = 1.0e-11,
            integration_atol = 1.0e-11,
            scale_rtols_by_layer_type = False,
            max_num_steps = 10_000_000,
            expected_size = 500,
            max_ram_MB = 500,
            max_step = 1000,
            limit_solution_to_radius = True,
            nondimensionalize = True,
            verbose = False,
            raise_on_fail = True
            )
    
    hp_c_1[i] = radial_solution.h[0]-radial_solution.k[0]
    hl_c_1[i] = radial_solution.h[1]-radial_solution.k[1]
    
    
    
    
    
    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear,
            forcing_frequency,
            planet_bulk_density,
            layer_types=("solid","liquid","solid","solid"),
            is_static_by_layer=(False,True,False,False),
            is_incompressible_by_layer=(True,True,True,True),
            upper_radius_by_layer=(3000, 1876E3, 2500E3,3392E3),
            degree_l=2,
            solve_for=('tidal','loading'),
            use_kamata=True,
            integration_method='DOP853',
            integration_rtol = 1.0e-8,
            integration_atol = 1.0e-8,
            scale_rtols_by_layer_type = False,
            max_num_steps = 10_000_000,
            expected_size = 500,
            max_ram_MB = 500,
            max_step = 0,
            limit_solution_to_radius = True,
            nondimensionalize = True,
            verbose = False,
            raise_on_fail = True
            )


    hp_i_2[i] = radial_solution.h[0]
    hl_i_2[i] = radial_solution.h[1]
    kp_i_2[i] = radial_solution.k[0]
    kl_i_2[i] = radial_solution.k[1]

        
    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear,
            forcing_frequency,
            planet_bulk_density,
            layer_types=("solid","liquid","solid","solid"),
            is_static_by_layer=(False,True,False,False),
            is_incompressible_by_layer=(False,False,False,False),
            upper_radius_by_layer=(3000, 1876E3, 2500E3,3392E3),
            degree_l=2,
            solve_for=('tidal','loading'),
            use_kamata=False,
            integration_method='DOP853',
            integration_rtol = 1.0e-8,
            integration_atol = 1.0e-8,
            scale_rtols_by_layer_type = False,
            max_num_steps = 10_000_000,
            expected_size = 500,
            max_ram_MB = 500,
            max_step = 1000,
            limit_solution_to_radius = True,
            nondimensionalize = True,
            verbose = False,
            raise_on_fail = True
            )
    
    hp_c_2[i] = radial_solution.h[0]
    hl_c_2[i] = radial_solution.h[1]
    kp_c_2[i] = radial_solution.k[0]
    kl_c_2[i] = radial_solution.k[1]

#%%

fig, axs = plt.subplots(3,2,figsize=(10,10))
axs[0,0].semilogx(freqs,hl_c_1,label="compressible")
axs[0,0].semilogx(freqs,hl_i_1,label="incompressible")
axs[0,0].set_title('$h^\prime_1$')
axs[0,0].axvline(687*24.6,ls='--',label="1 Martian Year")
axs[0,0].legend()
axs[0,0].grid()

axs[1,0].semilogx(freqs,hl_c_2,label="compressible")
axs[1,0].semilogx(freqs,hl_i_2,label="incompressible")
axs[1,0].axvline(687*24.6,ls='--',label="1 Martian Year")
axs[1,0].set_title('$h^\prime_2$')
axs[1,0].grid()

axs[2,0].semilogx(freqs,kl_c_2,label="compressible")
axs[2,0].semilogx(freqs,kl_i_2,label="incompressible")
axs[2,0].set_xlabel('frequency (hrs)')
axs[2,0].axvline(687*24.6,ls='--',label="1 Martian Year")
axs[2,0].set_title('$k^\prime_2$')
axs[2,0].grid()

axs[0,1].semilogx(freqs,hp_c_1,label="compressible")
axs[0,1].semilogx(freqs,hp_i_1,label="incompressible")
axs[0,1].set_title('$h_1$')
axs[0,1].axvline(687*24.6,ls='--',label="1 Martian Year")
axs[0,1].grid()

axs[1,1].semilogx(freqs,hp_c_2,label="compressible")
axs[1,1].semilogx(freqs,hp_i_2,label="incompressible")
axs[1,1].axvline(687*24.6,ls='--',label="1 Martian Year")
axs[1,1].set_title('$h_2$')
axs[1,1].grid()

axs[2,1].semilogx(freqs,kp_c_2,label="compressible")
axs[2,1].semilogx(freqs,kp_i_2,label="incompressible")
axs[2,1].set_xlabel('frequency (hrs)')
axs[2,1].axvline(687*24.6,ls='--',label="1 Martian Year")
axs[2,1].set_title('$k_2$')
axs[2,1].grid()

plt.tight_layout()
