#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:30:55 2024

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

int_model = 'models/AK_med.txt'
# int_model = 'mars_MD.txt'

r,vp,vs,rho = np.loadtxt(int_model,usecols=(0,1,2,3),delimiter=None,unpack=True)
    
[r,ind,inv] = np.unique(r,True,True)
vp = vp[ind]
vs = vs[ind]
rho = rho[ind]

if (r[0] < r[-1]):
    print(":: Reversing model to start from core")
    r = r[::-1]
    vp = vp[::-1]
    vs = vs[::-1]
    rho = rho[::-1]

# Convert to S.I. Units
r = np.multiply(r,1000.)
vp = np.multiply(vp,1000.)
vs = np.multiply(vs,1000.)
rho = np.multiply(rho,1000.)

# Convert Seismic Velocities to Elastic Moduli
mu = np.multiply(np.square(vs),rho)
K  = np.subtract(np.multiply(np.square(vp),rho),mu*(4./3.))
lam = np.subtract(K, mu*(2./3.))

visc = np.inf
eta   = np.full(len(mu),visc)



# Define planetary parameters
planet_mass = 6.417e23
planet_radius = 3.39e6
planet_volume = (4. / 3.) * np.pi * (planet_radius**3)
planet_bulk_density = planet_mass / planet_volume

forcing_frequency = 2. * np.pi / (687* 24.6 * 60 * 60)  # Phobos orbital freq

# Simple interior structure of solid IC, liquid OC, solid mantle
# N = 100
ICB = 500E3#(1. / 100.) * planet_radius
CMB = 1468E3#(1. / 2.) * planet_radius
lyr1 = 2033E3
lyr2 = 2360E3
lyr3 = 3280E3

radius_array = r#np.concatenate(
    # (
    # np.linspace(0.1, ICB, N),
    # np.linspace(ICB, CMB, N+1)[1:],
    # np.linspace(CMB, lyr1, N+1)[1:],
    # np.linspace(lyr1, lyr2, N+1)[1:],
    # np.linspace(lyr2, lyr3, N+1)[1:],
    # np.linspace(lyr3, planet_radius, N+1)[1:]
    # )
# )
shear_array = mu#np.concatenate(
#     (
#     8.74891794e+10 * np.ones(N, dtype=np.float64),
#     0.0 * np.ones(N, dtype=np.float64),
#     1.35859845e+11 * np.ones(N, dtype=np.float64),
#     1.10990570e+11 * np.ones(N, dtype=np.float64),
#     6.86982914e+10 * np.ones(N, dtype=np.float64),
#     4.39284066e+10 * np.ones(N, dtype=np.float64),
#     )
# )
viscosity_array = eta#np.concatenate(
#     (
#     1.0e14 * np.ones(N, dtype=np.float64),
#     1.0e20 * np.ones(N, dtype=np.float64),
#     1.0e20 * np.ones(N, dtype=np.float64),
#     1.0e20 * np.ones(N, dtype=np.float64),
#     1.0e20 * np.ones(N, dtype=np.float64),
#     1.0e20 * np.ones(N, dtype=np.float64),
#     )
# )
bulk_mod_array = K#np.concatenate(
#     (
#     1.92228327e+11 * np.ones(N, dtype=np.float64),
#     2.74228445e+11 * np.ones(N, dtype=np.float64),
#     2.39757054e+11 * np.ones(N, dtype=np.float64),
#     2.06221772e+11 * np.ones(N, dtype=np.float64),
#     1.78498479e+11 * np.ones(N, dtype=np.float64),
#     1.11196235e+11 * np.ones(N, dtype=np.float64),
#     )
# )
density_array = rho#np.concatenate(
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

state = np.full(len(r),'liquid' )
state[np.where(mu!=0)] = 'solid'
state = tuple(state.tolist())

comp = tuple(np.full(len(r),True).tolist())

ucomp = tuple(np.full(len(r),False).tolist())


for i in range(1,11):
    radial_solution = \
        radial_solver(
            radius_array,
            density_array,
            gravity_array,
            bulk_mod_array,
            complex_shear,
            forcing_frequency,
            planet_bulk_density,
            layer_types=(state),
            is_static_by_layer=(static),
            # is_incompressible_by_layer=(False, False, False),
            is_incompressible_by_layer=(comp),
            upper_radius_by_layer=(ICB, CMB, lyr1, lyr2, lyr3, planet_radius),
            degree_l=i,
            solve_for=('tidal','loading'),
            use_kamata=False,
            integration_method='DOP853',
            integration_rtol = 1.0e-15,
            integration_atol = 1.0e-15,
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
            layer_types=(state),
            is_static_by_layer=(static),
            is_incompressible_by_layer=(ucomp),
            # is_incompressible_by_layer=(False,True,True,True,True,True),
            upper_radius_by_layer=(ICB, CMB, lyr1, lyr2, lyr3, planet_radius),
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




n_met = np.arange(2,21)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)

# n1 = np.arange(0,11)
plt.plot(degs,hl_i,'-.',label="incompressible",ms=8,color='red')
plt.plot(degs,hl_c,'-.',label="compressible",ms=8,color='blue')

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
    
    