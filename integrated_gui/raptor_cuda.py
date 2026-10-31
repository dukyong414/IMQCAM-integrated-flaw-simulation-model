"""Numba-CUDA melt-mask kernel for ORNL Raptor (CPU fallback remains in raptor.core)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

_GEOMETRY_EPSILON = 1.0e-12
_INSTALLED = False
_KERNEL = None


def cuda_melt_mask_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _build_cuda_kernel():
    """One monolithic kernel — avoids nested device-function typing pitfalls."""
    from numba import cuda

    @cuda.jit
    def _melt_mask_kernel(
        resolution,
        melt_mask,
        origin,
        voxel_shape,
        tile_size,
        candidate_offsets,
        candidate_indices,
        path_z_values,
        e0,
        e1,
        inv_distances,
        L0_sqr,
        L1_sqr,
        z_bounds,
        lower_z_bounds,
        upper_z_bounds,
        spectral_table_offsets,
        spectral_table,
        spectral_sample_starts,
        spectral_full_point_counts,
        centroids,
        height_shape_factor,
        depth_shape_factor,
    ):
        column_index = cuda.grid(1)
        n_columns = voxel_shape[0] * voxel_shape[1]
        if column_index >= n_columns:
            return

        n_tile_y = (voxel_shape[1] + tile_size - 1) // tile_size
        nz = voxel_shape[2]
        voxel_x = column_index // voxel_shape[1]
        voxel_y = column_index - voxel_x * voxel_shape[1]
        tile_index = (voxel_x // tile_size) * n_tile_y + voxel_y // tile_size
        candidate_start = candidate_offsets[tile_index]
        candidate_end = candidate_offsets[tile_index + 1]
        vx = origin[0] + voxel_x * resolution
        vy = origin[1] + voxel_y * resolution
        eps = 1.0e-12

        for candidate_position in range(candidate_start, candidate_end):
            vector_index = candidate_indices[candidate_position]
            vec_cx = vx - centroids[vector_index, 0]
            vec_cy = vy - centroids[vector_index, 1]
            dot_e0_xy = vec_cx * e0[vector_index, 0] + vec_cy * e0[vector_index, 1]
            if dot_e0_xy * dot_e0_xy > L0_sqr[vector_index]:
                continue
            dot_e1_xy = vec_cx * e1[vector_index, 0] + vec_cy * e1[vector_index, 1]
            if dot_e1_xy * dot_e1_xy > L1_sqr[vector_index]:
                continue

            time_fraction = 0.0
            if inv_distances[vector_index] > 0.0:
                time_fraction = dot_e1_xy * inv_distances[vector_index] + 0.5
            if time_fraction < 0.0:
                time_fraction = 0.0
            elif time_fraction > 1.0:
                time_fraction = 1.0

            # evaluate_spectral_table
            table_start = spectral_table_offsets[vector_index]
            point_count = spectral_full_point_counts[vector_index]
            position = time_fraction * (point_count - 1)
            lower = int(position)
            upper = lower + 1
            if upper > point_count - 1:
                upper = point_count - 1
            local_lower = lower - spectral_sample_starts[vector_index]
            local_upper = upper - spectral_sample_starts[vector_index]
            fraction = position - float(lower)
            inv_fraction = 1.0 - fraction
            width = (
                spectral_table[table_start + local_lower, 0] * inv_fraction
                + spectral_table[table_start + local_upper, 0] * fraction
            )
            depth = (
                spectral_table[table_start + local_lower, 1] * inv_fraction
                + spectral_table[table_start + local_upper, 1] * fraction
            )
            height = (
                spectral_table[table_start + local_lower, 2] * inv_fraction
                + spectral_table[table_start + local_upper, 2] * fraction
            )

            local_y = dot_e0_xy
            local_y_sqr = local_y * local_y
            half_width = width * 0.5
            y_scaled = local_y / half_width
            y_term = y_scaled * y_scaled

            # conservative_vertical_interval
            interval_status = 0
            local_z_lower = 0.0
            local_z_upper = 0.0
            if (
                width > 0.0
                and height > 0.0
                and depth > 0.0
                and height_shape_factor > 0.0
                and depth_shape_factor > 0.0
            ):
                a = half_width
                inflated_y = math.fabs(local_y) - resolution - eps
                if inflated_y < 0.0:
                    inflated_y = 0.0
                if inflated_y > a:
                    interval_status = 2
                else:
                    remaining = 1.0 - (inflated_y / a) * (inflated_y / a)
                    if remaining < 0.0:
                        remaining = 0.0
                    if height_shape_factor == 1.0:
                        height_scale = remaining
                    elif height_shape_factor == 2.0:
                        height_scale = math.sqrt(remaining)
                    else:
                        height_scale = remaining ** (1.0 / height_shape_factor)
                    if depth_shape_factor == 1.0:
                        depth_scale = remaining
                    elif depth_shape_factor == 2.0:
                        depth_scale = math.sqrt(remaining)
                    else:
                        depth_scale = remaining ** (1.0 / depth_shape_factor)
                    inflation = resolution + eps
                    interval_status = 1
                    local_z_lower = -depth * depth_scale - inflation
                    local_z_upper = height * height_scale + inflation

            if interval_status == 2:
                continue

            lower_z = lower_z_bounds[vector_index]
            upper_z = upper_z_bounds[vector_index]
            path_z = path_z_values[vector_index]
            if interval_status == 1:
                dynamic_lower_z = int(math.floor((path_z + local_z_lower - origin[2]) / resolution)) - 1
                dynamic_upper_z = int(math.ceil((path_z + local_z_upper - origin[2]) / resolution)) + 1
                if dynamic_lower_z > lower_z:
                    lower_z = dynamic_lower_z
                if dynamic_upper_z < upper_z:
                    upper_z = dynamic_upper_z
                if lower_z > upper_z:
                    continue

            for voxel_z in range(lower_z, upper_z + 1):
                flat_index = column_index * nz + voxel_z
                vz = origin[2] + voxel_z * resolution
                if vz < z_bounds[vector_index, 0] or vz > z_bounds[vector_index, 1]:
                    continue
                local_z = vz - path_z

                # classify_horizontal_melt_and_boundary
                if local_z >= 0.0:
                    b = height
                    n = height_shape_factor
                else:
                    b = depth
                    n = depth_shape_factor
                radius_sqr = local_y_sqr + local_z * local_z
                is_voxel_melted = False
                is_voxel_boundary = False
                if radius_sqr < resolution * resolution:
                    m0 = half_width
                    if height < m0:
                        m0 = height
                    if depth < m0:
                        m0 = depth
                    center_distance = -m0
                    is_voxel_melted = center_distance < eps
                    is_voxel_boundary = math.fabs(center_distance) - resolution <= eps
                else:
                    radius = math.sqrt(radius_sqr)
                    inverse_radius = 1.0 / radius
                    z_scaled = math.fabs(local_z) / b
                    if n == 1.0:
                        z_term = z_scaled
                    elif n == 2.0:
                        z_term = z_scaled * z_scaled
                    else:
                        z_term = z_scaled**n
                    melt_scale = 1.0 - eps * inverse_radius
                    if melt_scale < 0.0:
                        melt_scale = 0.0
                    boundary_delta = (resolution + eps) * inverse_radius
                    lower_scale = 1.0 - boundary_delta
                    if lower_scale < 0.0:
                        lower_scale = 0.0
                    upper_scale = 1.0 + boundary_delta
                    if n == 1.0:
                        melt_value = y_term * melt_scale * melt_scale + z_term * melt_scale
                        lower_value = y_term * lower_scale * lower_scale + z_term * lower_scale
                        upper_value = y_term * upper_scale * upper_scale + z_term * upper_scale
                    elif n == 2.0:
                        base_value = y_term + z_term
                        melt_value = base_value * melt_scale * melt_scale
                        lower_value = base_value * lower_scale * lower_scale
                        upper_value = base_value * upper_scale * upper_scale
                    else:
                        melt_value = y_term * melt_scale * melt_scale + z_term * melt_scale**n
                        lower_value = y_term * lower_scale * lower_scale + z_term * lower_scale**n
                        upper_value = y_term * upper_scale * upper_scale + z_term * upper_scale**n
                    is_voxel_melted = melt_value < 1.0
                    is_voxel_boundary = lower_value <= 1.0 and upper_value >= 1.0

                previous_phase = melt_mask[flat_index]
                if is_voxel_melted:
                    melt_mask[flat_index] = 1
                if is_voxel_boundary:
                    melt_mask[flat_index] = 2
                if previous_phase > 1 and is_voxel_boundary:
                    melt_mask[flat_index] = 3

    return _melt_mask_kernel


def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_cuda_kernel()
    return _KERNEL


def compute_melt_mask_kernel_cuda(
    resolution: float,
    melt_mask: np.ndarray,
    origin: np.ndarray,
    voxel_shape: np.ndarray,
    tile_size: int,
    candidate_offsets: np.ndarray,
    candidate_indices: np.ndarray,
    path_z_values: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    inv_distances: np.ndarray,
    L0_sqr: np.ndarray,
    L1_sqr: np.ndarray,
    z_bounds: np.ndarray,
    lower_z_bounds: np.ndarray,
    upper_z_bounds: np.ndarray,
    spectral_table_offsets: np.ndarray,
    spectral_table: np.ndarray,
    spectral_sample_starts: np.ndarray,
    spectral_full_point_counts: np.ndarray,
    centroids: np.ndarray,
    height_shape_factor: np.float64,
    depth_shape_factor: np.float64,
) -> np.ndarray:
    """GPU drop-in for ``raptor.core.compute_melt_mask_kernel``."""
    from numba import cuda

    kernel = _get_kernel()
    n_columns = int(voxel_shape[0]) * int(voxel_shape[1])
    threads = 256
    blocks = max(1, (n_columns + threads - 1) // threads)

    d_melt = cuda.to_device(np.ascontiguousarray(melt_mask))
    d_origin = cuda.to_device(np.ascontiguousarray(origin, dtype=np.float64))
    d_shape = cuda.to_device(np.ascontiguousarray(voxel_shape, dtype=np.int64))
    d_cand_off = cuda.to_device(np.ascontiguousarray(candidate_offsets))
    d_cand_idx = cuda.to_device(np.ascontiguousarray(candidate_indices))
    d_path_z = cuda.to_device(np.ascontiguousarray(path_z_values))
    d_e0 = cuda.to_device(np.ascontiguousarray(e0))
    d_e1 = cuda.to_device(np.ascontiguousarray(e1))
    d_inv = cuda.to_device(np.ascontiguousarray(inv_distances))
    d_l0 = cuda.to_device(np.ascontiguousarray(L0_sqr))
    d_l1 = cuda.to_device(np.ascontiguousarray(L1_sqr))
    d_zb = cuda.to_device(np.ascontiguousarray(z_bounds))
    d_lz = cuda.to_device(np.ascontiguousarray(lower_z_bounds))
    d_uz = cuda.to_device(np.ascontiguousarray(upper_z_bounds))
    d_toff = cuda.to_device(np.ascontiguousarray(spectral_table_offsets))
    d_table = cuda.to_device(np.ascontiguousarray(spectral_table))
    d_sstarts = cuda.to_device(np.ascontiguousarray(spectral_sample_starts))
    d_pcounts = cuda.to_device(np.ascontiguousarray(spectral_full_point_counts))
    d_cent = cuda.to_device(np.ascontiguousarray(centroids))

    kernel[blocks, threads](
        float(resolution),
        d_melt,
        d_origin,
        d_shape,
        int(tile_size),
        d_cand_off,
        d_cand_idx,
        d_path_z,
        d_e0,
        d_e1,
        d_inv,
        d_l0,
        d_l1,
        d_zb,
        d_lz,
        d_uz,
        d_toff,
        d_table,
        d_sstarts,
        d_pcounts,
        d_cent,
        float(height_shape_factor),
        float(depth_shape_factor),
    )
    cuda.synchronize()
    d_melt.copy_to_host(melt_mask)
    return melt_mask


def install_cuda_melt_mask(force: bool = False) -> bool:
    """Patch ``raptor.core.compute_melt_mask_kernel`` to the CUDA implementation."""
    global _INSTALLED
    if _INSTALLED and not force:
        return True
    if not cuda_melt_mask_available():
        return False
    import raptor.core as core

    if not hasattr(install_cuda_melt_mask, "_cpu_kernel"):
        install_cuda_melt_mask._cpu_kernel = core.compute_melt_mask_kernel
    _get_kernel()
    core.compute_melt_mask_kernel = compute_melt_mask_kernel_auto
    _INSTALLED = True
    return True


def uninstall_cuda_melt_mask() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    import raptor.core as core

    cpu = getattr(install_cuda_melt_mask, "_cpu_kernel", None)
    if cpu is not None:
        core.compute_melt_mask_kernel = cpu
    _INSTALLED = False


def enable_for_device(device: str) -> Optional[str]:
    """Enable CUDA melt-mask when ``device == 'gpu'``. Returns a short log line."""
    if device != "gpu":
        uninstall_cuda_melt_mask()
        return "Raptor melt-mask kernel: Numba CPU"
    try:
        if install_cuda_melt_mask():
            return "Raptor melt-mask kernel: CUDA GPU"
    except Exception as exc:
        uninstall_cuda_melt_mask()
        return f"Raptor melt-mask kernel: CPU fallback ({exc})"
    return "Raptor melt-mask kernel: CPU fallback (CUDA unavailable)"


def compute_melt_mask_kernel_auto(*args, **kwargs):
    """Try CUDA; on failure restore and run the original CPU Numba kernel."""
    try:
        return compute_melt_mask_kernel_cuda(*args, **kwargs)
    except Exception:
        uninstall_cuda_melt_mask()
        cpu = getattr(install_cuda_melt_mask, "_cpu_kernel", None)
        if cpu is None:
            raise
        return cpu(*args, **kwargs)
