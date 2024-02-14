# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
""" Functions to calculate the initial conditions for an overlying liquid layer above another liquid layer.

For liquid-liquid layer interfaces: all the radial functions are continuous expect for if you are moving
from a dynamic layer to static layer or vice versa.

References
----------
S74   : Saito (1974; J. Phy. Earth; DOI: 10.4294/jpe1952.22.123)
TS72  : Takeuchi, H., and M. Saito (1972), Seismic surface waves, Methods Comput. Phys., 11, 217–295.
"""

from scipy.constants import G as G_

from libc.math cimport pi

cdef double G
G = G_


cdef void cf_solve_upper_y_at_interface(
        double complex* lower_layer_y_ptr,
        double complex* upper_layer_y_ptr,
        size_t num_sols_lower,
        size_t num_sols_upper,
        size_t num_y_lower,
        size_t num_y_upper,
        int lower_layer_type,
        bint lower_is_static,
        bint lower_is_incompressible,
        int upper_layer_type,
        bint upper_is_static,
        bint upper_is_incompressible,
        double interface_gravity,
        double liquid_density,
        double G_to_use = G
        ) noexcept nogil:

    cdef size_t yi_lower, yi_upper, soli_lower, soli_upper

    cdef bint solid_solid, solid_liquid, liquid_solid, liquid_liquid
    solid_solid   = (lower_layer_type == 0) and (upper_layer_type == 0)
    solid_liquid  = (lower_layer_type == 0) and not (upper_layer_type == 0)
    liquid_solid  = not (lower_layer_type == 0) and (upper_layer_type == 0)
    liquid_liquid = not (lower_layer_type == 0) and not (upper_layer_type == 0)

    cdef bint static_static, static_dynamic, dynamic_static, dynamic_dynamic
    static_static   = lower_is_static and upper_is_static
    static_dynamic  = lower_is_static and not upper_is_static
    dynamic_static  = not lower_is_static and upper_is_static
    dynamic_dynamic = not lower_is_static and not upper_is_static

    cdef bint compress_compress, compress_incompress, incompress_compress, incompress_incompress
    compress_compress     = not lower_is_incompressible and not upper_is_incompressible
    compress_incompress   = not lower_is_incompressible and upper_is_incompressible
    incompress_compress   = lower_is_incompressible and not upper_is_incompressible
    incompress_incompress = lower_is_incompressible and upper_is_incompressible
    # TODO: Compressibility is not currently taken into account. This needs to be checked asap!

    # Other constants that may be needed
    cdef double complex lambda_1, lambda_2, coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6, frac_1, frac_2
    cdef double complex const_1, g_const

    g_const = 4. * pi * G_to_use

    if solid_solid:
        if static_static or static_dynamic or dynamic_static or dynamic_dynamic:
            # Does not matter if the layers are static or dynamic, solid-solid exchange perfectly.
            # All solutions carry over
            for yi_lower in range(num_y_lower):
                # They have same number of ys
                yi_upper = yi_lower
                for soli_lower in range(num_sols_lower):
                    # They have same number of sols
                    soli_upper = soli_lower
                    upper_layer_y_ptr[soli_upper * num_y_upper + yi_upper] = \
                        lower_layer_y_ptr[soli_lower * num_y_lower + yi_lower]
    elif liquid_liquid:
        # Liquid interfaces carry everything over if they are both static or both dynamic.
        if static_static or dynamic_dynamic:
            # All solutions carry over
            for yi_lower in range(num_y_lower):
                # They have same number of ys
                yi_upper = yi_lower
                for soli_lower in range(num_sols_lower):
                    # They have same number of sols
                    soli_upper = soli_lower
                    upper_layer_y_ptr[soli_upper * num_y_upper + yi_upper] = \
                        lower_layer_y_ptr[soli_lower * num_y_lower + yi_lower]
        elif static_dynamic:
            # For an upper layer that is liquid and dynamic there will be two independent solutions that need an initial guess.
            # # Solution 1
            # y_1_dynamic = 0
            upper_layer_y_ptr[0] = 0. + 0.j
            # y_2_dynamic = -rho * y_5_static
            upper_layer_y_ptr[1] = -liquid_density * lower_layer_y_ptr[0]
            # y_5_dynamic = y_5_static
            upper_layer_y_ptr[2] = lower_layer_y_ptr[0]
            # y_6_dynamic = y_7_static + (4 pi G rho / g) * y_5_static
            upper_layer_y_ptr[3] = \
                lower_layer_y_ptr[1] + (g_const * liquid_density / interface_gravity) * lower_layer_y_ptr[0]
            # # Solution 2
            # y_1_dynamic = 1.
            upper_layer_y_ptr[1 * num_y_upper + 0] = 1. + 0.j
            # y_2_dynamic = rho * g * y_1_dynamic
            upper_layer_y_ptr[1 * num_y_upper + 1] = \
                liquid_density * interface_gravity * upper_layer_y_ptr[1 * num_y_upper + 0]
            # y_5_dynamic = 0.
            upper_layer_y_ptr[1 * num_y_upper + 2] = 0. + 0.j
            # y_6_dynamic = -4 pi G rho y_1_dynamic
            upper_layer_y_ptr[1 * num_y_upper + 3] = \
                -g_const * liquid_density * upper_layer_y_ptr[1 * num_y_upper + 0]
        elif dynamic_static:
            # lambda_j = (y_2j - rho * ( g * y_1j - y_5j))
            lambda_1 = \
                lower_layer_y_ptr[1] - \
                liquid_density * (interface_gravity * lower_layer_y_ptr[0] - lower_layer_y_ptr[2])
            lambda_2 = \
                lower_layer_y_ptr[1 * num_y_lower + 1] - \
                liquid_density * (interface_gravity * lower_layer_y_ptr[1 * num_y_lower + 0] -
                                  lower_layer_y_ptr[1 * num_y_lower + 2])

            # Set the first coefficient to 1. It will be solved for later on during the collapse phase.
            coeff_1 = 1. + 0.j
            # The other coefficient which is related to 1 via...
            coeff_2 = -(lambda_1 / lambda_2) * coeff_1

            coeff_3 = lower_layer_y_ptr[3] + g_const * lower_layer_y_ptr[1] / interface_gravity
            coeff_4 = \
                lower_layer_y_ptr[1 * num_y_lower + 3] + \
                g_const * lower_layer_y_ptr[1 * num_y_lower + 1] / interface_gravity

            # y^liq(st)_5 = C^liq(dy)_1 * y^liq(dy)_5,1 + C^liq(dy)_2 * y^liq(dy)_5,2
            upper_layer_y_ptr[0] = coeff_1 * lower_layer_y_ptr[2] + coeff_2 * lower_layer_y_ptr[1 * num_y_lower + 2]
            # y^liq(st)_7 = C^liq(dy)_1 * y^liq(dy)_7,1 + C^liq(dy)_2 * y^liq(dy)_7,2
            upper_layer_y_ptr[1] = coeff_1 * coeff_3 + coeff_2 * coeff_4
        else:
            # How did you get here...
            pass
    elif liquid_solid:
        if dynamic_dynamic or dynamic_static:
            # As far as I am aware, the dynamic_static and dynamic_dynamic solutions are the same.

            # See Eqs. 148-149 in TS72
            # For a dynamic solid layer there will be three independent solutions that we need an initial guess for.
            for soli_upper in range(3):
                if soli_upper == 0 or soli_upper == 1:
                    # For a dynamic liquid layer there will be two independent solutions at the top of the layer
                    upper_layer_y_ptr[soli_upper * num_y_upper + 0] = lower_layer_y_ptr[soli_upper * num_y_lower + 0]
                    upper_layer_y_ptr[soli_upper * num_y_upper + 1] = lower_layer_y_ptr[soli_upper * num_y_lower + 1]
                    upper_layer_y_ptr[soli_upper * num_y_upper + 4] = lower_layer_y_ptr[soli_upper * num_y_lower + 2]
                    upper_layer_y_ptr[soli_upper * num_y_upper + 5] = lower_layer_y_ptr[soli_upper * num_y_lower + 3]

                    # For solutions 1 and 2: y_3 and y_4 for the solid layer are zero
                    upper_layer_y_ptr[soli_upper * num_y_upper + 2] = 0. + 0.j
                    upper_layer_y_ptr[soli_upper * num_y_upper + 3] = 0. + 0.j
                else:
                    # For the third solid solution all the y's are set to zero except y_3.
                    upper_layer_y_ptr[soli_upper * num_y_upper + 0] = 0. + 0.j
                    upper_layer_y_ptr[soli_upper * num_y_upper + 1] = 0. + 0.j
                    upper_layer_y_ptr[soli_upper * num_y_upper + 2] = 1. + 0.j
                    upper_layer_y_ptr[soli_upper * num_y_upper + 3] = 0. + 0.j
                    upper_layer_y_ptr[soli_upper * num_y_upper + 4] = 0. + 0.j
                    upper_layer_y_ptr[soli_upper * num_y_upper + 5] = 0. + 0.j
        elif static_dynamic or static_static:
            # As far as I am aware, the static_static and static_dynamic solutions are the same.

            # Eqs. 20 in S74
            # For a static liquid layer there will be one independent solutions at the top of the layer
            # We need to solve for six solutions in the static or dynamic solid upper layer
            upper_layer_y_ptr[0] = 0. + 0.j
            # y_2_sol = -rho * y_5_liq
            upper_layer_y_ptr[1] = -liquid_density * lower_layer_y_ptr[0]
            upper_layer_y_ptr[2] = 0. + 0.j
            upper_layer_y_ptr[3] = 0. + 0.j
            # y_5_sol = y_5_liq
            upper_layer_y_ptr[4] = lower_layer_y_ptr[0]
            # y_6_sol = y_7_liq + (4 pi G rho / g) y_5_liq
            upper_layer_y_ptr[5] = \
                lower_layer_y_ptr[1] + (g_const * liquid_density / interface_gravity) * lower_layer_y_ptr[0]
            # y_1_sol = 1.
            upper_layer_y_ptr[1 * num_y_upper + 0] = 1. + 0.j
            # y_2_sol = rho * g * y_1_sol
            upper_layer_y_ptr[1 * num_y_upper + 1] = \
                liquid_density * interface_gravity * upper_layer_y_ptr[1 * num_y_upper + 0]
            upper_layer_y_ptr[1 * num_y_upper + 2] = 0. + 0.j
            upper_layer_y_ptr[1 * num_y_upper + 3] = 0. + 0.j
            upper_layer_y_ptr[1 * num_y_upper + 4] = 0. + 0.j
            # y_6_sol = -4 pi G rho y_1_sol
            upper_layer_y_ptr[1 * num_y_upper + 5] = \
                -g_const * liquid_density * upper_layer_y_ptr[1 * num_y_upper + 0]

            upper_layer_y_ptr[2 * num_y_upper + 0] = 0. + 0.j
            upper_layer_y_ptr[2 * num_y_upper + 1] = 0. + 0.j
            # y_3_sol = 1.
            upper_layer_y_ptr[2 * num_y_upper + 2] = 1. + 0.j
            upper_layer_y_ptr[2 * num_y_upper + 3] = 0. + 0.j
            upper_layer_y_ptr[2 * num_y_upper + 4] = 0. + 0.j
            upper_layer_y_ptr[2 * num_y_upper + 5] = 0. + 0.j
    elif solid_liquid:

        if dynamic_dynamic or static_dynamic:
            # As far as I am aware, static_dynamic and dynamic_dynamic should be the same.
            # Eqs. 140-144 in TS72
            # Three solutions in the lower solid layer and two in the upper liquid.

            for soli_upper in range(2):
                # Solve for y^liq_1, y^liq_2, y^liq_5, y^liq_6 (TS72 Eq. 143)
                #    Note that the liquid solution does not have y_3, y_4 which are index 2, 3 for solid solution.
                coeff_1 = lower_layer_y_ptr[soli_upper * num_y_lower + 3] / lower_layer_y_ptr[2 * num_y_lower + 3]

                upper_layer_y_ptr[soli_upper * num_y_upper + 0] = \
                    lower_layer_y_ptr[soli_upper * num_y_lower + 0] - coeff_1 * lower_layer_y_ptr[2 * num_y_lower + 0]
                upper_layer_y_ptr[soli_upper * num_y_upper + 1] = \
                    lower_layer_y_ptr[soli_upper * num_y_lower + 1] - coeff_1 * lower_layer_y_ptr[2 * num_y_lower + 1]
                upper_layer_y_ptr[soli_upper * num_y_upper + 2] = \
                    lower_layer_y_ptr[soli_upper * num_y_lower + 4] - coeff_1 * lower_layer_y_ptr[2 * num_y_lower + 4]
                upper_layer_y_ptr[soli_upper * num_y_upper + 3] = \
                    lower_layer_y_ptr[soli_upper * num_y_lower + 5] - coeff_1 * lower_layer_y_ptr[2 * num_y_lower + 5]
        elif dynamic_static or static_static:
            # As far as I am aware, static_static and dynamic_static should work the same.
            # Eq. 21 in S74
            frac_1 = -lower_layer_y_ptr[0 * num_y_lower + 3] / lower_layer_y_ptr[2 * num_y_lower + 3]
            frac_2 = -lower_layer_y_ptr[1 * num_y_lower + 3] / lower_layer_y_ptr[2 * num_y_lower + 3]

            # lambda_j = (y_2j + f_j y_23) - rho( g(y_1j + f_j y_13) - (y_5j + f_j y_53))
            lambda_1 = \
                lower_layer_y_ptr[1] + frac_1 * lower_layer_y_ptr[2 * num_y_lower + 1] - \
                liquid_density * (
                        interface_gravity * (
                            lower_layer_y_ptr[0] + frac_1 * lower_layer_y_ptr[2 * num_y_lower + 0]) -
                        (lower_layer_y_ptr[4] + frac_1 * lower_layer_y_ptr[2 * num_y_lower + 4])
                )
            lambda_2 = \
                lower_layer_y_ptr[1 * num_y_lower + 1] + frac_2 * lower_layer_y_ptr[2 * num_y_lower + 1] - \
                liquid_density * (
                        interface_gravity * (
                            lower_layer_y_ptr[1 * num_y_lower + 0] + frac_2 * lower_layer_y_ptr[2 * num_y_lower + 0]) -
                        (lower_layer_y_ptr[1 * num_y_lower + 4] + frac_2 * lower_layer_y_ptr[2 * num_y_lower + 4])
                )

            # Set the first coefficient to 1. It will be solved for later on during the collapse phase.
            coeff_1 = 1.
            # The other two coefficients are related to 1 via...
            coeff_2 = -(lambda_1 / lambda_2) * coeff_1
            coeff_3 = frac_1 * coeff_1 + frac_2 * coeff_2

            const_1 = (g_const / interface_gravity)

            coeff_4 = lower_layer_y_ptr[0 * num_y_lower + 5] + const_1 * lower_layer_y_ptr[0 * num_y_lower + 1]
            coeff_5 = lower_layer_y_ptr[1 * num_y_lower + 5] + const_1 * lower_layer_y_ptr[1 * num_y_lower + 1]
            coeff_6 = lower_layer_y_ptr[2 * num_y_lower + 5] + const_1 * lower_layer_y_ptr[2 * num_y_lower + 1]

            # y^liq_5 = C^sol_1 * y^sol_5,1 + C^sol_2 * y^sol_5,2 + C^sol_3 * y^sol_5,3
            upper_layer_y_ptr[0] = \
                coeff_1 * lower_layer_y_ptr[0 * num_y_lower + 4] + \
                coeff_2 * lower_layer_y_ptr[1 * num_y_lower + 4] + \
                coeff_3 * lower_layer_y_ptr[2 * num_y_lower + 4]
            # y^liq_7 = C^sol_1 * y^sol_7,1 + C^sol_2 * y^sol_7,2 + C^sol_3 * y^sol_7,3
            upper_layer_y_ptr[1] = \
                coeff_1 * coeff_4 + coeff_2 * coeff_5 + coeff_3 * coeff_6


def solve_upper_y_at_interface(
        double complex[:, ::1] lower_layer_y_view,
        double complex[:, ::1] upper_layer_y_view,
        int lower_layer_type,
        bint lower_is_static,
        bint lower_is_incompressible,
        int upper_layer_type,
        bint upper_is_static,
        bint upper_is_incompressible,
        double interface_gravity,
        double liquid_density,
        double G_to_use = G
        ):
    
    # Pull out number of solutions and y's
    cdef size_t num_sols_lower = lower_layer_y_view.shape[0]
    cdef size_t num_sols_upper = upper_layer_y_view.shape[0]
    cdef size_t num_y_lower    = lower_layer_y_view.shape[1]
    cdef size_t num_y_upper    = upper_layer_y_view.shape[1]

    # Run pure-c function
    cf_solve_upper_y_at_interface(
        &lower_layer_y_view[0, 0],
        &upper_layer_y_view[0, 0],
        num_sols_lower,
        num_sols_upper,
        num_y_lower,
        num_y_upper,
        lower_layer_type,
        lower_is_static,
        lower_is_incompressible,
        upper_layer_type,
        upper_is_static,
        upper_is_incompressible,
        interface_gravity,
        liquid_density,
        G_to_use = G
        )
