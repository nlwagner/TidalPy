#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:33:50 2024

@author: nlwagner25
"""


import numpy as np
import matplotlib.pyplot as plt

import sys
import os

from TidalPy.RadialSolver import radial_solver
from TidalPy.utilities.spherical_helper import calculate_mass_gravity_arrays
from TidalPy.utilities.graphics.multilayer import yplot

sys.path.append(os.getcwd() + "models")

int_model = 'models/MD_med.txt'
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

forcing_frequency = 2. * np.pi / (687* 24.6 * 60 * 60)  # Phobos orbital freq


N = 50

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
    
    
    
# Purely Elastic Body
from TidalPy.rheology import Elastic

elastic_rheology = Elastic()
# Calculate the "complex" shear (really all Im[mu] = 0)
complex_shear = np.empty(radius_array.shape, dtype=np.complex128)
shear_array = np.ascontiguousarray(shear_array)
viscosity_array = np.ascontiguousarray(viscosity_array)
elastic_rheology.vectorize_modulus_viscosity(forcing_frequency, shear_array, viscosity_array, complex_shear)


degs = np.arange(1,11)
kp_i = np.zeros(10)
hp_i = np.zeros(10)
kl_i = np.zeros(10)
hl_i = np.zeros(10)

kp_c = np.zeros(10)
hp_c = np.zeros(10)
kl_c = np.zeros(10)
hl_c = np.zeros(10)

static = np.full(len(r),False)
static[np.where(mu==0)] = True
static = tuple(static.tolist())

ur = tuple(np.copy(r))


comp = tuple(np.full(len(r),True).tolist())

ucomp = tuple(np.full(len(r),False).tolist())


for i in range(0,11):
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
            degree_l=i,
            solve_for=('tidal','loading'),
            use_kamata=True,
            integration_method='DOP853',
            integration_rtol = 1.0e-10,
            integration_atol = 1.0e-10,
            scale_rtols_by_layer_type = False,
            max_num_steps = 1_000_000,
            expected_size = 500,
            max_ram_MB = 500,
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
            complex_shear,
            forcing_frequency,
            planet_bulk_density,
            layer_types=("solid","liquid","solid"),
            is_static_by_layer=(False,True,False),
            is_incompressible_by_layer=(False, False, False),
            upper_radius_by_layer=(3000, 1876E3, 3392E3),
            degree_l=i,
            solve_for=('tidal','loading'),
            use_kamata=False,
            integration_method='DOP853',
            integration_rtol = 1.0e-10,
            integration_atol = 1.0e-10,
            scale_rtols_by_layer_type = False,
            max_num_steps = 1_000_000,
            expected_size = 500,
            max_ram_MB = 500,
            max_step = 1000,
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
    


#%%

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)

# n1 = np.arange(0,11)
plt.plot(degs,hl_i,'-.',label="incompressible",ms=8,color='red')
plt.plot(degs,hl_c,'-.',label="compressible",ms=8,color='blue')

plt.xticks(np.arange(0, 21,1))
plt.xlim([1,10])
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

plt.xticks(np.arange(0, 11,1))
plt.xlim([1,10])
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
    
    