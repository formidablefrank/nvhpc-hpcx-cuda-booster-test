# CUDA-Aware MPI Matrix Multiplication Tests

This repository contains CUDA-aware MPI experiments for Leonardo Booster nodes. It run a distributed matrix multiplication in Fortran using NVHPC, OpenACC, CUDA-aware communication through HPC-X MPI, and parallel HDF5 output.

## Repository Layout

- `dist_matmul.f90` - distributed matrix multiplication benchmark. It initializes local matrix blocks, performs a ring exchange of `A` blocks, computes local matrix multiplication on GPUs with OpenACC, validates the local result, writes a parallel HDF5 file, and prints timing for initialization, computation, communication, I/O, validation, and total runtime.
- `job_dist_matmul.sh` - Slurm batch script for building `dist_matmul.f90` and running scaling tests on 1, 2, 4, and 8 nodes. Each scale point is run twice: baseline and tuned.
- `binder.sh` - per-rank GPU and UCX device binding helper. In baseline mode it only assigns one GPU per local rank. In tuned mode it also pins each local rank to a matching UCX network device.
- `hpcx-only-env.sh` - loads the NVHPC/HPC-X module and exposes the Spack-built CUDA, NCCL, HDF5, NetCDF, and PnetCDF prefixes.
- `build-hpcx-only-env.sh` - concretizes and installs the Spack environment in `env-nvhpc-hpcx/`.
- `env-nvhpc-hpcx/spack.yaml` - Spack environment definition.
- `logs/` - Slurm stdout/stderr files.

## Software Stack

The scripts assume the following software stack built using Spack:

- NVHPC 25.11
- HPC-X 2.20
- CUDA 12.2.2 from the Spack environment (compatible with NVIDIA-SMI 535.274.02, Driver Version: 535.274.02, CUDA Version: 12.2)
- NCCL
- HDF5 and NetCDF-C/Fortran built with MPI (HPC-X) and Fortran (NVHPC)

<!--The current NetCDF-C build is MPI-enabled but not pthread-backed async:

- `netcdf-c@4.9.2 +mpi`
- `--has-parallel4 -> yes`
- `--has-parallel -> yes`
- `--has-pnetcdf -> no`
- HDF5 has `threadsafe=false`-->

## Build the Spack Environment

Replace relevant paths on `env-nvhpc-hpcx/spack.yaml` the run this command once, or whenever the YAML file changes:

```bash
./build-hpcx-only-env.sh
```

This installs packages under:

```text
/leonardo_scratch/large/userexternal/$USER/spack-install/env-nvhpc-hpcx
```

To load the environment in an interactive shell:

```bash
source ./hpcx-only-env.sh
```

That script exports paths such as `CUDA_HOME`, `NVHPC_CUDA_HOME`, `HDF5_HOME`, `NETCDF_C_HOME`, `NETCDF_FORTRAN_HOME`, `PNETCDF_HOME`, and `HPCX_MPI_HOME`.

## Building Manually

The Slurm script builds the program automatically, but the manual command is:

```bash
source ./hpcx-only-env.sh

"${HPCX_MPI_HOME}/bin/mpif90" -O3 -acc -gpu=cc80 -Minfo=accel \
  -I"${HDF5_HOME}/include" \
  -L"${HDF5_HOME}/lib" \
  dist_matmul.f90 -o dist_matmul.x \
  -lhdf5_fortran -lhdf5
```

## Run the Scaling Benchmark

Submit the Slurm job:

```bash
sbatch job_dist_matmul.sh
```

The job requests 8 nodes and runs these scale points:

```text
1 node  / 4 ranks
2 nodes / 8 ranks
4 nodes / 16 ranks
8 nodes / 32 ranks
```

Each point is run in two modes:

- `baseline` - no explicit MPI binding/mapping, no `UCX_NET_DEVICES`, no `UCX_TLS`, no `OMPI_MCA_pml`, no `OMPI_MCA_osc`, no `UCX_RNDV_THRESH`, and no `PMIX_MCA_gds` tuning. `binder.sh` still assigns `CUDA_VISIBLE_DEVICES` by local rank so ranks use separate GPUs.
- `tuned` - uses `--bind-to core --map-by ppr:4:node:PE=8`, environment UCX/Open MPI settings mentioned above, and  `UCX_NET_DEVICES` and GPU binding per local rank from `binder.sh`.

Set `REPORT_BINDINGS=1` to include Open MPI binding reports:

```bash
REPORT_BINDINGS=1 sbatch job_dist_matmul.sh
```

Set `UCX_LOG_LEVEL` if UCX logs are needed:

```bash
UCX_LOG_LEVEL=info sbatch job_dist_matmul.sh
```

## Outputs

Slurm output and error files are written to:

```text
logs/slurm-matmul-<jobid>.out
logs/slurm-matmul-<jobid>.err
```

Each run writes a separate HDF5 output file:

```text
C_dist_baseline_1nodes_4ranks.h5
C_dist_tuned_1nodes_4ranks.h5
C_dist_baseline_2nodes_8ranks.h5
C_dist_tuned_2nodes_8ranks.h5
C_dist_baseline_4nodes_16ranks.h5
C_dist_tuned_4nodes_16ranks.h5
C_dist_baseline_8nodes_32ranks.h5
C_dist_tuned_8nodes_32ranks.h5
```

The timing line has this format:

```text
TIMING ranks=<n> init_s=<s> computation_s=<s> communication_s=<s> io_s=<s> validation_s=<s> total_s=<s>
```

Timings are reduced with `MPI_MAX` across ranks, so each value represents the slowest rank for that component. Communication time includes MPI post time and MPI wait time. Because communication is overlapped with GPU computation, the reported communication component is the exposed communication cost, not a fully non-overlapped transfer time.

## Notes

- The benchmark currently uses `N = 32768` in `dist_matmul.f90`; memory use is high and the Slurm scripts request exclusive GPU nodes.
- `N` must be divisible by the MPI world size.
- The Fortran benchmark uses OpenACC `host_data use_device` around MPI calls, so CUDA-aware MPI support is required for device-buffer communication.
- If the NVHPC compiler reports a missing CUDA toolkit, check that `hpcx-only-env.sh` exports `NVHPC_CUDA_HOME` and `NVCOMPILER_CUDA_HOME` to the Spack CUDA 12.2.2 prefix.
