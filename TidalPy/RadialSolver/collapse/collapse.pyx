# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cdef void cf_collapse_layer_solution(
        double complex* solution_ptr,
        double complex* constant_vector_ptr,
        double complex** storage_by_solution,
        double* layer_radius_ptr,
        double* layer_density_ptr,
        double* layer_gravity_ptr,
        double frequency_to_use,
        size_t layer_start_index,
        size_t num_layer_slices,
        unsigned char num_sols,
        unsigned char max_num_y,
        unsigned char num_ys,
        unsigned char num_output_ys,
        unsigned char ytype_i,
        bool_cpp_t layer_is_solid,
        bool_cpp_t layer_is_static,
        bool_cpp_t layer_is_incomp
        ) noexcept nogil:

    # Use constant vectors to find the full y from all of the solutions in this layer
    cdef unsigned char solution_i
    cdef unsigned char y_i
    cdef unsigned char lhs_y_index
    cdef size_t slice_i
    cdef size_t slice_i_shifted
    cdef size_t slice_end
    cdef bool_cpp_t calculate_y3

    slice_end = layer_start_index + num_layer_slices
    calculate_y3 = False

    if (not layer_is_solid) and (not layer_is_static):
        # For this layer type we can also calculate y3 based on the other ys. We will do this at the end.
        calculate_y3 = True
    
    y_i = 0
    while max_num_y > y_i:
        lhs_y_index = ytype_i * max_num_y + y_i
        # Bail out early for ys that the layer type is not defined for.
        if not layer_is_solid:
            if layer_is_static:
                # Liquid static layers only has y5 (stored at index 0).
                if y_i != 4:
                    # All results should be NAN which the solution pointer should already be set to. Continue.
                    continue
            else:
                # Liquid dynamic layers have y1, y2, y5, y6 (indices 0, 1, 2, 3)
                # For this layer type we can also calculate y3 based on the other ys.
                if (y_i == 2) or (y_i == 3):
                    # y4 == NAN for liquid dynamic layers; y3 will be calculated later.
                    continue
        
        # Otherwise we will collapse the solutions together for each slice.
        solution_i = 0
        while num_sols > solution_i:
            # New solution; reset slice index
            slice_i = 0
            slice_i_shifted = layer_start_index
            while slice_end > slice_i_shifted:
                if solution_i == 0:
                    # Initialize values
                    solution_ptr[slice_i_shifted * num_output_ys + lhs_y_index] = \
                        (constant_vector_ptr[solution_i] *
                        storage_by_solution[solution_i][slice_i * num_ys + y_i])
                else:
                    # Add new results to old value
                    solution_ptr[slice_i_shifted * num_output_ys + lhs_y_index] += \
                        (constant_vector_ptr[solution_i] *
                        storage_by_solution[solution_i][slice_i * num_ys + y_i])

                slice_i += 1
                slice_i_shifted += 1
            solution_i += 1
        y_i += 1
    
    if calculate_y3:
        # Handle y3 for dynamic liquid layers.
        lhs_y_index = ytype_i * max_num_y
        slice_i = 0
        slice_i_shifted = layer_start_index
        while slice_end > slice_i_shifted:
            solution_ptr[slice_i_shifted * num_output_ys + (lhs_y_index + 2)] = \
                (1. / (frequency_to_use**2 * layer_radius_ptr[slice_i])) * \
                (solution_ptr[slice_i_shifted * num_output_ys + (lhs_y_index + 0)] * layer_gravity_ptr[slice_i] -
                solution_ptr[slice_i_shifted * num_output_ys + (lhs_y_index + 1)] / layer_density_ptr[slice_i] -
                solution_ptr[slice_i_shifted * num_output_ys + (lhs_y_index + 4)])

            slice_i += 1
            slice_i_shifted += 1
