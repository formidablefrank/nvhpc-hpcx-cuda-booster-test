# CUDA-Aware MPI Matrix Multiplication Tests

This repository contains CUDA-aware MPI experiments for Leonardo Booster nodes. It run a distributed matrix multiplication in Fortran using NVHPC, OpenACC, CUDA-aware communication through HPC-X MPI, and parallel HDF5 output.

## Repository Layout

- `dist_matmul.f90` - distributed matrix multiplication benchmark. It initializes local matrix blocks, performs a ring exchange of `A` blocks, computes local matrix multiplication on GPUs with OpenACC, validates the local result, writes a parallel HDF5 file, and prints timing for initialization, computation, communication, I/O, validation, and total runtime.
- `job_dist_matmul.sh` - Slurm batch script for building `dist_matmul.f90` and running scaling tests on 1, 2, 4, and 8 nodes. Each scale point is run twice: baseline and tuned.
- `dist_matmul_stdpar.f90` - experimental Fortran `do concurrent`/NVHPC stdpar version of the matrix multiplication benchmark. It uses managed memory semantics instead of explicit OpenACC data regions and `host_data` device pointers.
- `job_dist_matmul_stdpar.sh` - Slurm batch script for building and running the stdpar variant with the same baseline/tuned scaling structure.
- `binder.sh` - per-rank GPU and UCX device binding helper. In baseline mode it only assigns one GPU per local rank. In tuned mode it also pins each local rank to a matching UCX network device.
- `hpcx-only-env.sh` - loads the NVHPC/HPC-X module and exposes the Spack-built CUDA, NCCL, HDF5, NetCDF, and PnetCDF prefixes.
- `build-hpcx-only-env.sh` - concretizes and installs the Spack environment in `env-nvhpc-hpcx/`.
- `env-nvhpc-hpcx/spack.yaml` - Spack environment definition.
- `logs/` - Slurm stdout/stderr files.

## Software Stack

The scripts assume the following software stack built using Spack:

- NVHPC 25.11 (external module)
- HPC-X 2.20 (external module)
- CUDA 12.2.2 from the Spack environment (compatible with NVIDIA-SMI 535.274.02, Driver Version: 535.274.02, CUDA Version: 12.2)
- NCCL and CUDNN
- HDF5 and NetCDF-C/Fortran and Parallel NetCDF

<!--The current NetCDF-C build is MPI-enabled but not pthread-backed async:

- `netcdf-c@4.9.2 +mpi`
- `--has-parallel4 -> yes`
- `--has-parallel -> yes`
- `--has-pnetcdf -> no`
- HDF5 has `threadsafe=false`-->

Dependency graph:

```
                        +----------------------------+
                        |   nvhpc@25.11 (Compiler)   |
                        +--------------+-------------+
                                       |
       +-------------------------------+-------------------------------+
       |                                                               |
       |  [ CUDA Branch ]                                              |  [ MPI / NetCDF Branch ]
       |                                                               |
       |       +-------------------+                                   |       +--------------------+
       |       |    cuda@12.2.2    |                                   |       |   hpcx-mpi@2.20    |
       |       +---------+---------+                                   |       +-------+---+--------+
       |                 |                                             |               |   |
       v                 v                                             v               v   v
     +-------------------+                                           +-------------------+ |
     |   nccl@2.22.3-1   | <─────────────────────────────────────────|    hdf5@1.14.3    | |
     +-------------------+                                           +--------+----------+ |
                                                                              |            |
     +-------------------+                                                    |            |
     |  cudnn@9.2.0.82-12| <──────────────────────────────────────────────────┼────────────+
     +-------------------+                                                    |            |
                                                                              v            v
                                                                     +-------------------+ |
                                                                     |  netcdf-c@4.9.2   | |
                                                                     +--------+----------+ |
                                                                              |            |
                                                                              v            v
                                                                     +-------------------------+
                                                                     |  netcdf-fortran@4.6.1   |
                                                                     +-------------------------+

                                                                     +-------------------------+
                                                                     | parallel-netcdf@1.12.3  |
                                                                     +-------------------------+
                                                                     (Fed by nvhpc & hpcx-mpi)
```


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

To run the experimental stdpar variant instead:

```bash
sbatch job_dist_matmul_stdpar.sh
```

The stdpar script compiles with:

```bash
-stdpar=gpu -gpu=cc80,mem:managed
```

This variant uses `do concurrent` for GPU-parallel initialization, matrix multiplication, and block-copy loops. It does not use OpenACC `host_data use_device`, so MPI sees managed Fortran arrays rather than explicit OpenACC device pointers. Treat it as a portability and programming-model comparison against the OpenACC CUDA-aware MPI version, not as a guaranteed faster replacement.

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

The stdpar script writes similarly named files with a `C_dist_stdpar_` prefix:

```text
C_dist_stdpar_baseline_1nodes_4ranks.h5
C_dist_stdpar_tuned_1nodes_4ranks.h5
...
C_dist_stdpar_baseline_8nodes_32ranks.h5
C_dist_stdpar_tuned_8nodes_32ranks.h5
```

The timing line has this format:

```text
TIMING ranks=<n> init_s=<s> computation_s=<s> communication_s=<s> io_s=<s> validation_s=<s> total_s=<s>
```

Timings are reduced with `MPI_MAX` across ranks, so each value represents the slowest rank for that component. Communication time includes MPI post time and MPI wait time. Because communication is overlapped with GPU computation, the reported communication component is the exposed communication cost, not a fully non-overlapped transfer time.

## Results Analysis

The latest analyzed run is `logs/slurm-matmul-43839900.out`. Plots and parsed CSV files are available under `plots/`:

- `plots/timing_components_baseline_vs_tuned.png` - per-component comparison for initialization, computation, communication, I/O, validation, and total time.
- `plots/timing_total_baseline_vs_tuned.png` - total runtime plus tuned speedup over baseline.
- `plots/timing_baseline_vs_tuned.csv` - parsed component timings.
- `plots/timing_total_summary.csv` - total runtime and speedup summary.

Total runtime from this run:

| Nodes | Ranks | Baseline total (s) | Tuned total (s) | Baseline / tuned |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 59.154 | 55.188 | 1.07x |
| 2 | 8 | 36.268 | 32.103 | 1.13x |
| 4 | 16 | 20.166 | 17.932 | 1.12x |
| 8 | 32 | 11.478 | 11.727 | 0.98x |

The tuned configuration improves total time on 1, 2, and 4 nodes, with the strongest gain around 2-4 nodes. The dominant improvement is I/O time: tuned I/O drops from 9.06 s to 5.15 s on 1 node, from 10.59 s to 6.35 s on 2 nodes, and from 6.87 s to 4.57 s on 4 nodes. Computation time is essentially unchanged between baseline and tuned runs because both modes execute the same GPU kernels with the same rank count and problem decomposition.

At 8 nodes, tuned total time is slightly worse in this sample. The main reason is that tuned I/O is not better at this scale in the latest run (4.57 s tuned vs 4.39 s baseline), while exposed communication is also higher (0.709 s tuned vs 0.617 s baseline). This difference is small compared with the full runtime and should be interpreted as one-run variability unless repeated measurements show the same trend.

The communication component is not a pure network bandwidth measurement. The code posts nonblocking MPI receives/sends, launches the OpenACC kernel, waits for the GPU computation, and then waits for MPI. Therefore `communication_s` mostly measures MPI posting plus residual wait time after computation overlap. A tuned run can show slightly higher `communication_s` while still being faster overall if computation finishes earlier, overlap changes, or I/O improves enough to dominate the total runtime. For a clean communication-only comparison, add a separate benchmark mode that times the device-buffer ring exchange without the matrix kernel and HDF5 write, or disable overlap by waiting for MPI before launching computation.

Validation in the latest run reports relative errors around `1e-14` to `1e-13`, which is consistent with double-precision roundoff for results with magnitude near `1e18`. The nonzero absolute errors are expected at this scale and should be judged using `max_rel_error`, not only `max_abs_error`.

## Notes

- The benchmark currently uses `N = 32768` in `dist_matmul.f90`; memory use is high and the Slurm scripts request exclusive GPU nodes.
- `N` must be divisible by the MPI world size.
- The OpenACC benchmark uses `host_data use_device` around MPI calls, so CUDA-aware MPI support is required for explicit device-buffer communication.
- The stdpar benchmark relies on NVHPC managed memory behavior for MPI/HDF5 visibility. If it runs slower or shows different communication behavior, that is expected and should be interpreted as a programming-model comparison.
- If the NVHPC compiler reports a missing CUDA toolkit, check that `hpcx-only-env.sh` exports `NVHPC_CUDA_HOME` and `NVCOMPILER_CUDA_HOME` to the Spack CUDA 12.2.2 prefix.

## References
- [NVIDIA HPC SDK 25.11 release notes](https://docs.nvidia.com/hpc-sdk/archive/25.11/pdf/hpc-sdk2511rn.pdf)
- [Spack environments](https://spack.readthedocs.io/en/latest/environments.html)
- [Programming for NVIDIA GPUs](https://www.nas.nasa.gov/hecc/support/kb/programming-for-nvidia-gpus_647.html)
