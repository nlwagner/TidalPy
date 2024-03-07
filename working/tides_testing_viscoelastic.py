#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:12:23 2024

@author: nlwagner25
"""
import numpy as np
import matplotlib.pyplot as plt

from TidalPy.RadialSolver import radial_solver
from TidalPy.utilities.spherical_helper import calculate_mass_gravity_arrays
from TidalPy.utilities.graphics.multilayer import yplot

# Define planetary parameters
planet_mass = 6.39e23
planet_radius = 3.396e6
planet_volume = (4. / 3.) * np.pi * (planet_radius**3)
planet_bulk_density = planet_mass / planet_volume

forcing_frequency = 2. * np.pi / (687*24.6 * 60 * 60)  # Phobos orbital freq

# Simple interior structure of solid IC, liquid OC, solid mantle
N = 100
ICB = .1E3#(1. / 100.) * planet_radius
CMB = 1830E3#(1. / 2.) * planet_radius
lyr1 = (3396-42)*1E3

radius_array = np.concatenate(
    (
    np.linspace(0.1, ICB, N),
    np.linspace(ICB, CMB, N+1)[1:],
    np.linspace(CMB, lyr1, N+1)[1:],
    np.linspace(lyr1, planet_radius, N+1)[1:]
    )
)
shear_array = np.concatenate(
    (
    0.0 * np.ones(N, dtype=np.float64),
    0.0 * np.ones(N, dtype=np.float64),
    71.96E9 * np.ones(N, dtype=np.float64),
    20.24E9 * np.ones(N, dtype=np.float64),
    )
)
viscosity_array = np.concatenate(
    (
    1.0e30 * np.ones(N, dtype=np.float64),
    1.0e30 * np.ones(N, dtype=np.float64),
    1.0e19 * np.ones(N, dtype=np.float64),
    1.0e30 * np.ones(N, dtype=np.float64),
    )
)
bulk_mod_array = np.concatenate(
    (
    1.92228327e+11 * np.ones(N, dtype=np.float64),
    2.74228445e+11 * np.ones(N, dtype=np.float64),
    2.39757054e+11 * np.ones(N, dtype=np.float64),
    1.11196235e+11 * np.ones(N, dtype=np.float64),
    )
)
density_array = np.concatenate(
    (
    6165.98 * np.ones(N, dtype=np.float64),
    6165.98 * np.ones(N, dtype=np.float64),
    3423.15 * np.ones(N, dtype=np.float64),
    2582. * np.ones(N, dtype=np.float64),
    )
)

volume_array, mass_array, gravity_array = \
    calculate_mass_gravity_arrays(radius_array, density_array)
    
    
    
    
from TidalPy.rheology import Andrade

andrade_rheology = Andrade((0.3,0.03)) # zeta=0.1 for eta=log(20). zeta=0.3 for eta=log(19)
# Calculate the "complex" shear (really all Im[mu] = 0)
complex_shear_andrade = np.empty(radius_array.shape, dtype=np.complex128)
andrade_rheology.vectorize_modulus_viscosity(forcing_frequency, shear_array, viscosity_array, complex_shear_andrade)

degs = np.arange(1,11)
kp_i = np.zeros(10)
hp_i = np.zeros(10)
kl_i = np.zeros(10)
hl_i = np.zeros(10)

kp_c = np.zeros(10)
hp_c = np.zeros(10)
kl_c = np.zeros(10)
hl_c = np.zeros(10)


for i in range(1,11):
    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear_andrade,
            forcing_frequency,
            planet_bulk_density,
            layer_types=('liquid', 'liquid', 'solid', 'solid'),
            is_static_by_layer=(True, True, False, False),
            # is_incompressible_by_layer=(False, False, False),
            is_incompressible_by_layer=(False,True,True,True),
            upper_radius_by_layer=(ICB, CMB, lyr1, planet_radius),
            degree_l=i,
            solve_for=('tidal','loading'),
            use_kamata=False,
            integration_method='DOP853',
            integration_rtol = 1.0e-12,
            integration_atol = 1.0e-12,
            scale_rtols_by_layer_type = False,
            max_num_steps = 10_000_000,
            expected_size = 1000,
            max_ram_MB = 1000,
            max_step = 0,
            limit_solution_to_radius = True,
            nondimensionalize = True,
            verbose = False,
            raise_on_fail = True
            )

    kp_i[i-1] = radial_solution.k[0]
    hp_i[i-1] = radial_solution.h[0]
    kl_i[i-1] = radial_solution.k[1]
    hl_i[i-1] = radial_solution.h[1]
    
    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear_andrade,
            forcing_frequency,
            planet_bulk_density,
            layer_types=('liquid', 'liquid', 'solid', 'solid'),
            is_static_by_layer=(True, True, False, False),
            is_incompressible_by_layer=(False, False, False, False),
            # is_incompressible_by_layer=(False,True,True,True,True,True),
            upper_radius_by_layer=(ICB, CMB, lyr1, planet_radius),
            degree_l=i,
            solve_for=('tidal','loading'),
            use_kamata=False,
            integration_method='DOP853',
            integration_rtol = 1.0e-12,
            integration_atol = 1.0e-12,
            scale_rtols_by_layer_type = False,
            max_num_steps = 1_000_000,
            expected_size = 500,
            max_ram_MB = 1000,
            max_step = 0,
            limit_solution_to_radius = True,
            nondimensionalize = True,
            verbose = False,
            raise_on_fail = True
            )

    kp_c[i-1] = radial_solution.k[0]
    hp_c[i-1] = radial_solution.h[0]
    kl_c[i-1] = radial_solution.k[1]
    hl_c[i-1] = radial_solution.h[1]
    print(i)

print(hl_c[0])
#%%
import sys
import os
sys.path.append(os.getcwd() + "petricca")
lln_file1 = ("petricca/lln_MARS_MODEL8_ALMA.txt")
lln_file2 = ("petricca/lln_MARS_MODEL383_ALMA.txt")
lln_file3 = ("petricca/lln_MARS_MODEL750_ALMA.txt")
lln_file4 = ("petricca/lln_MARS_MODEL754_ALMA.txt")
lln_file5 = ("petricca/lln_MARS_MODEL758_ALMA.txt")
lln_file6 = ("petricca/lln_MARS_MODEL766_ALMA.txt")
lln_file7 = ("petricca/lln_MARS_MODEL774_ALMA.txt")
lln_file8 = ("petricca/lln_MARS_MODEL1133_ALMA.txt")
lln_file9 = ("petricca/lln_MARS_MODEL1483_ALMA.txt")
n,h_p1,l_p1,k_p1 = np.loadtxt(lln_file1,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p2,l_p2,k_p2 = np.loadtxt(lln_file2,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p3,l_p3,k_p3 = np.loadtxt(lln_file3,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p4,l_p4,k_p4 = np.loadtxt(lln_file4,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p5,l_p5,k_p5 = np.loadtxt(lln_file5,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p6,l_p6,k_p6 = np.loadtxt(lln_file6,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p7,l_p7,k_p7 = np.loadtxt(lln_file7,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p8,l_p8,k_p8 = np.loadtxt(lln_file8,delimiter=None,unpack=True,usecols=(0,1,2,3))
n,h_p9,l_p9,k_p9 = np.loadtxt(lln_file9,delimiter=None,unpack=True,usecols=(0,1,2,3))

n_met = np.arange(2,21)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)

# n1 = np.arange(0,11)
plt.plot(degs,hl_i,'-.',label="incompressible",ms=8,color='red')
plt.plot(degs,hl_c,'-.',label="compressible",ms=8,color='blue')
# 
# plt.plot(np.arange(0, 21,1),h_p1,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p1,'k--',alpha=0.3,label=r'Petricca et. al. 2022')
# plt.plot(np.arange(0, 21,1),h_p2,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p2,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p3,'k',alpha=0.3)
plt.plot(np.arange(0, 21,1),h_p3,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p4,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p4,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p5,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p5,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p6,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p6,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p7,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p7,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p8,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p8,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),h_p9,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),h_p9,'k--',alpha=0.3)
# 
# plt.plot(n_met,h_met1468,'-.',label="Metivier",ms=8,color='green')

plt.xticks(np.arange(0, 21,1))
plt.xlim([2,10])
# plt.ylim([-.5,-0.1])
plt.ylim([-.5,-0.10])


plt.tick_params(labelsize='large')


#plt.legend(loc='best',fontsize='large',ncol=1)
plt.title('Load Love Numbers $(h^\prime)$',fontsize = 'xx-large')
plt.xlabel(r'$n$',fontsize='xx-large')
plt.grid()
#plt.savefig((outdir+figname),orientation='portrait',format='pdf')



# Plot
plt.subplot(1,2,2)
plt.plot(degs,kl_i,'-.',label="incompressible",ms=8,color='red')
plt.plot(degs,kl_c,'-.',label="compressible",ms=8,color='blue')
# plt.plot(n_met,k_met1468,'-.',label="Metivier",ms=8,color='green')

#plt.plot(np.arange(0, 21,1),k_p1,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p1,'k--',alpha=0.3,label=r'Petricca et. al. 2022')
#plt.plot(np.arange(0, 21,1),k_p2,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p2,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p3,'k',alpha=0.3)
plt.plot(np.arange(0, 21,1),k_p3,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p4,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p4,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p5,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p5,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p6,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p6,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p7,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p7,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p8,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p8,'k--',alpha=0.3)
#plt.plot(np.arange(0, 21,1),k_p9,'k',alpha=0.3)
# plt.plot(np.arange(0, 21,1),k_p9,'k--',alpha=0.3)

plt.xticks(np.arange(0, 11,1))
plt.xlim([2,10])
plt.ylim([-0.25,-0.02])
plt.tick_params(labelsize='large')

plt.legend(loc='best',fontsize='12',ncol=1)
plt.title('Load Love Numbers $(k^\prime)$',fontsize = 'xx-large')
plt.xlabel(r'$n$',fontsize='xx-large')
plt.grid()
plt.tight_layout()
# plt.savefig((outdir+'Figure_S1.pdf'),dpi=300,orientation='landscape',format='pdf')
plt.show()


# print('Purely Elastic Body')
# print(f'k2={radial_solution.k}, h2={radial_solution.h}, l2={radial_solution.l}')  
    
    
#%%   