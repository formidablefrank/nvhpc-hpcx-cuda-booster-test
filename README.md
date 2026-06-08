# CUDA-Aware MPI test application for Leonardo Booster nodes

This repository contains benchmarking experiments for Leonardo Booster nodes. It runs distributed matrix multiplication in Fortran using NVHPC, OpenACC/stdpar, CUDA-aware communication through HPC-X MPI, and parallel HDF5 output and readback.

## Contents

- `dist_matmul.f90` - distributed matrix multiplication benchmark program. It initializes local matrix blocks, performs a ring exchange of `A` blocks, computes local matrix multiplication on GPUs with OpenACC, validates the local result, writes an HDF5 file in parallel, and reads the written hyperslab back collectively. It also prints timing for initialization, computation, communication, I/O, error validation, and total runtime. The file is written to `$FAST` filesystem of Leonardo.
- `job_dist_matmul.sh` - Slurm batch script for building `dist_matmul.f90` and running scaling tests on 1, 2, 4, and 8 nodes.
- `dist_matmul_stdpar.f90` - NVHPC stdpar `do concurrent` version of the benchmark. It uses separate-memory stdpar offload instead of explicit OpenACC data regions and `host_data` device pointers.
- `job_dist_matmul_stdpar.sh` - Slurm batch script for building and running the stdpar version.
- `binder.sh` - per-rank GPU and UCX device binding helper. In baseline mode it only assigns one GPU per local rank. In tuned mode it also pins each local rank to a matching UCX network device.
- `hpcx-only-env.sh` - loads the NVHPC/HPC-X module and exposes the Spack-built CUDA, NCCL, HDF5, NetCDF, and PnetCDF prefixes.
- `build-hpcx-only-env.sh` - concretizes and installs the Spack environment in `env-nvhpc-hpcx/`.
- `env-nvhpc-hpcx/spack.yaml` - Spack environment definition.
- `logs/` - Slurm stdout/stderr files.

## Software Stack

The benchmark assumes the following software stack built using Spack:

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

Replace relevant paths on `env-nvhpc-hpcx/spack.yaml` then run this command once or whenever the YAML file changes:

```bash
./build-hpcx-only-env.sh
```

This installs packages under:

```text
/leonardo_scratch/large/userexternal/$USER/spack-install/env-nvhpc-hpcx
```

When debugging, load the environment in an interactive shell:

```bash
srun --account $ACCOUNT --partition=boost_usr_prod --qos=boost_qos_dbg --nodes=1 --ntasks=4 --ntasks-per-node=32 --gres=gpu:4 --pty bash
source ./hpcx-only-env.sh
...
```

That script exports paths such as `CUDA_HOME`, `NVHPC_CUDA_HOME`, `HDF5_HOME`, `NETCDF_C_HOME`, `NETCDF_FORTRAN_HOME`, `PNETCDF_HOME`, and `HPCX_MPI_HOME`.

## Compilation

The Slurm script compiles the program before running it so you don't need to do this. In any case the command is:

```bash
"${HPCX_MPI_HOME}/bin/mpif90" -O3 -acc -gpu=cc80 -Minfo=accel \
  -I"${HDF5_HOME}/include" \
  -L"${HDF5_HOME}/lib" \
  dist_matmul.f90 -o dist_matmul.x \
  -lhdf5_fortran -lhdf5
```

For stdpar variant, the command is:

```bash
"${HPCX_MPI_HOME}/bin/mpif90" -O3 -stdpar=gpu -gpu=cc80,mem:separate -Minfo=stdpar \
  -I"${HDF5_HOME}/include" \
  -L"${HDF5_HOME}/lib" \
  dist_matmul_stdpar.f90 -o dist_matmul_stdpar.x \
  -lhdf5_fortran -lhdf5
```

This variant uses `do concurrent` for parallel initialization, matrix multiplication, and block-copy loops offloaded to GPU. It does not use `host_data use_device` directives, so MPI sees ordinary host Fortran arrays with compiler-managed device copies rather than explicit OpenACC device pointers. Treat it as a portability and programming-model comparison against the OpenACC CUDA-aware MPI version, not as a guaranteed faster replacement.

## Run the Scaling Benchmark

Submit the Slurm job:

```bash
sbatch job_dist_matmul.sh
```

To run the stdpar variant instead:

```bash
sbatch job_dist_matmul_stdpar.sh
```

The job requests 8 nodes and runs each scaling point:

```text
1 node  / 4 ranks
2 nodes / 8 ranks
4 nodes / 16 ranks
8 nodes / 32 ranks
```

Each point is run in two modes:

- `baseline` - no explicit MPI binding/mapping, no per-rank `UCX_NET_DEVICES`, no `UCX_RNDV_THRESH`, and no `numactl --localalloc`. It still keeps the CUDA-aware MPI requirements (`OMPI_MCA_pml=ucx`, `OMPI_MCA_osc=ucx`, and `UCX_TLS` with CUDA transports), because the OpenACC code passes device pointers to MPI. `binder.sh` still assigns `CUDA_VISIBLE_DEVICES` by local rank so ranks use separate GPUs.
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

After the parallel write, each rank reopens the file with the MPI-IO HDF5 file access property list and collectively reads back the same hyperslab it wrote. The readback check compares the file data against the rank-local `C_loc` buffer:

```text
READBACK max_abs_error=<s>
TIMING_IO ranks=<n> parallel_write_s=<s> parallel_read_s=<s>
```

The timing line has this format:

```text
TIMING ranks=<n> init_s=<s> computation_s=<s> communication_s=<s> io_s=<s> io_write_s=<s> io_read_s=<s> validation_s=<s> total_s=<s>
```

Timings are obtained using `MPI_MAX` reduction across ranks, so each value comes from the slowest rank for that component. Communication time includes MPI post time and MPI wait time. Because communication is overlapped with GPU computation, the reported communication component is the exposed communication cost, not a fully non-overlapped transfer time. The `io_s` value includes both the parallel write and the parallel readback; `io_write_s` and `io_read_s` report those phases separately. The `TIMING_IO` line repeats the same split with explicit parallel read/write labels for easier parsing.

## Results

Plots and parsed CSV files are available under `plots/` and are generated from `logs/slurm-matmul-44177268.out`:

- `plots/generate_timing_plots.py` - parser and plot generator for Slurm timing logs.
- `plots/timing_components_baseline_vs_tuned.png` - runtime comparison for initialization, computation, communication, parallel HDF5 write, parallel HDF5 readback, validation, and total time.
- `plots/timing_total_baseline_vs_tuned.png` - total runtime plus speedup after tuning.
- `plots/timing_baseline_vs_tuned.csv` - parsed component timings, validation errors, and readback errors.
- `plots/timing_total_summary.csv` - total runtime, speedup, and I/O deltas.

Regenerate the CSVs and plots with:

```bash
conda activate rl-gpu
python plots/generate_timing_plots.py logs/slurm-matmul-44177268.out
```

Total runtime from this run:

![speedup](plots/timing_total_baseline_vs_tuned.png)

| Nodes | Ranks | Baseline total (s) | Tuned total (s) | Baseline / tuned |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 66.069 | 64.925 | 1.02x |
| 2 | 8 | 43.949 | 57.942 | 0.76x |
| 4 | 16 | 29.205 | 32.180 | 0.91x |
| 8 | 32 | 16.031 | 16.414 | 0.98x |

The latest run shows that computation scaling remains strong and essentially independent of the tuning mode. Baseline computation time drops from 47.61 s on 1 node to 23.84 s, 11.93 s, and 5.96 s on 2, 4, and 8 nodes. Tuned computation differs by less than a few milliseconds at every scale, so the tuning choices are not affecting GPU kernel throughput.

The total-runtime difference is dominated by HDF5 I/O and, in this run, mostly by write behavior. Tuned is slightly faster on 1 node because parallel write time is lower. At 2 nodes, tuned write time jumps to 26.91 s versus 12.84 s baseline, causing a large tuned slowdown. At 4 nodes, tuned write and read are both slower. At 8 nodes, tuned is only slightly slower overall, with a small write penalty partly offset by slightly faster readback.

| Nodes | Baseline write (s) | Tuned write (s) | Baseline read (s) | Tuned read (s) | I/O delta baseline - tuned (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.969 | 7.431 | 2.490 | 2.919 | 1.116 |
| 2 | 12.841 | 26.908 | 3.031 | 2.920 | -13.959 |
| 4 | 12.978 | 14.852 | 1.754 | 2.717 | -2.838 |
| 8 | 6.556 | 6.947 | 1.890 | 1.770 | -0.273 |

![timing](plots/timing_components_baseline_vs_tuned.png)

This run is an important contrast with the previous one: the large 2-node tuned regression is a parallel HDF5 write regression, not a compute or readback issue. Communication is also somewhat higher in tuned mode at multi-node scale, but the exposed communication increase is much smaller than the I/O change. For example, at 2 nodes tuned communication is only about 0.084 s higher, while tuned write time is about 14.07 s higher.

All readbacks report `READBACK max_abs_error=0`, so the parallel HDF5 read path is returning exactly the same local values that were written. Relative validation errors are around `1e-14` to `1e-13`, which is consistent with double-precision roundoff for results with magnitude near `1e18`. The nonzero absolute errors are expected at this scale and should be judged using `max_rel_error`, not only `max_abs_error`.

The tuned configuration can make parallel HDF5 read or write slower even though it is useful for GPU-aware MPI traffic. HDF5 collective I/O is not just a direct GPU-to-HCA transfer. In this program the file operations use host-resident `C_loc` after `!$acc update self(C_loc)`, and HDF5 then uses MPI-IO collective algorithms to aggregate file accesses. Those algorithms are sensitive to rank order, aggregator selection, file-system lock/stripe placement, and the Open MPI I/O component in use. Pinning each rank to a specific CPU/GPU/HCA path can improve communication locality while also changing which ranks become slow I/O participants or aggregators. Because collective HDF5 calls complete at the pace of the slowest participating ranks, a few slower ranks can increase the reported maximum read or write time.

The read path is especially sensitive to file-system cache state and collective buffering choices. A tuned run may read from the same file correctly but still be slower if its rank/HCA placement creates less favorable access concurrency, if the MPI-IO aggregators are mapped onto ranks with poorer file-system locality, or if the previous write left data in a cache/layout state that benefits the baseline read more than the tuned read. The current data shows this variability clearly: in `44177268`, tuned readback is slightly faster at 2 and 8 nodes but slower at 1 and 4 nodes, while tuned write is much worse at 2 nodes.

The communication component is also not a pure network bandwidth measurement. The code posts nonblocking MPI receive/send, launches the OpenACC kernel, waits for GPU computation, and then waits for the nonblocking MPI calls. Therefore `communication_s` mostly measures MPI post overhead plus any exposed wait after compute overlap. Tuned mode pins ranks to cores and maps each local rank to a specific HCA through `UCX_NET_DEVICES`. That can reduce ambiguity, but it can also remove UCX's ability to choose another rail dynamically, expose imbalance if one HCA path is busier, and add overhead from stricter endpoint/device selection. Since the compute phase is nearly identical between modes, even a small change in when the GPU kernel finishes can expose more of the pending MPI wait. In this run, tuned communication is higher at 2, 4, and 8 nodes, but the increase is small compared with the HDF5 I/O swings.

For follow-up measurements, repeat the 2-, 4-, and 8-node points several times and alternate baseline/tuned order. If the tuned I/O penalty persists, isolate HDF5 behavior separately from GPU/HCA placement by comparing OMPIO versus ROMIO, collective versus independent dataset transfer, and filesystem striping for the output directory. Also test tuned CPU/GPU binding without forcing per-rank `UCX_NET_DEVICES`; that separates core/NUMA effects from rail/HCA selection. The current data is enough to say that tuned mapping is not reliably better for the HDF5 write/read phase on the fast scratch target used in this run.

## Notes

- The benchmark currently uses `N = 32768` in `dist_matmul.f90` and reports distributed validation plus HDF5 readback checks. Memory use is high and the Slurm scripts request exclusive GPU nodes.
- `N` must be divisible by the MPI world size.
- The OpenACC benchmark uses `host_data use_device` around MPI calls, so CUDA-aware MPI support is required for explicit device-buffer communication.
- The stdpar benchmark uses NVHPC separate-memory behavior so MPI/HDF5 operate on host arrays while stdpar kernels use compiler-managed device copies. If it runs slower or shows different communication behavior, that is expected and should be interpreted as a programming-model comparison.
- If the NVHPC compiler reports a missing CUDA toolkit, check that `hpcx-only-env.sh` exports `NVHPC_CUDA_HOME` and `NVCOMPILER_CUDA_HOME` to the Spack CUDA 12.2.2 prefix.
- The batch scripts set `OMPI_MCA_fcoll=^vulcan` to avoid an Open MPI OMPIO `vulcan` file-collective crash observed during parallel HDF5 writes.

## References
- [NVIDIA HPC SDK 25.11 release notes](https://docs.nvidia.com/hpc-sdk/archive/25.11/pdf/hpc-sdk2511rn.pdf)
- [Spack environments](https://spack.readthedocs.io/en/latest/environments.html)
- [Programming for NVIDIA GPUs](https://www.nas.nasa.gov/hecc/support/kb/programming-for-nvidia-gpus_647.html)

## Diagnostics

```
nvidia-smi

Tue Jun  2 18:46:44 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.274.02             Driver Version: 535.274.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM-64GB           On  | 00000000:1D:00.0 Off |                    0 |
| N/A   44C    P0              76W / 467W |  33254MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM-64GB           On  | 00000000:56:00.0 Off |                    0 |
| N/A   44C    P0              74W / 465W |  33254MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM-64GB           On  | 00000000:8F:00.0 Off |                    0 |
| N/A   43C    P0              72W / 448W |  33254MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM-64GB           On  | 00000000:C8:00.0 Off |                    0 |
| N/A   43C    P0              76W / 458W |  33254MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    255872      C   ...pcx-cuda-booster-test/dist_matmul.x    33246MiB |
|    1   N/A  N/A    255873      C   ...pcx-cuda-booster-test/dist_matmul.x    33246MiB |
|    2   N/A  N/A    255874      C   ...pcx-cuda-booster-test/dist_matmul.x    33246MiB |
|    3   N/A  N/A    255875      C   ...pcx-cuda-booster-test/dist_matmul.x    33246MiB |
+---------------------------------------------------------------------------------------+

nvidia-smi topo -m

        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV4     NV4     NV4     PXB     NODE    NODE    NODE    0-15    0               N/A
GPU1    NV4      X      NV4     NV4     NODE    PXB     NODE    NODE    0-15    0               N/A
GPU2    NV4     NV4      X      NV4     NODE    NODE    PXB     NODE    0-15    0               N/A
GPU3    NV4     NV4     NV4      X      NODE    NODE    NODE    PXB     0-15    0               N/A
NIC0    PXB     NODE    NODE    NODE     X      NODE    NODE    NODE
NIC1    NODE    PXB     NODE    NODE    NODE     X      NODE    NODE
NIC2    NODE    NODE    PXB     NODE    NODE    NODE     X      NODE
NIC3    NODE    NODE    NODE    PXB     NODE    NODE    NODE     X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3

```
