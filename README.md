# ai-cluster-validator

Zero-dependency userspace smoke test for AI clusters. This repo validates multi-node PyTorch DDP initialization, GPU affinity, and NCCL fabric connectivity under Slurm orchestration.

This project is designed to help users and platform engineers run a **fast AI cluster preflight** for multi-GPU, multi-node environments before launching production training workloads. It gives you a deterministic signal that:

- Slurm can launch the expected GPU topology across nodes.
- PyTorch distributed (NCCL backend) can form and complete a collective ring.
- GPU-to-rank mapping and local affinity are sane.
- Basic InfiniBand/NVLink capability is visible from userspace.

## Why this exists

In production AI clusters, failures often happen before training logic starts:

- Incorrect Slurm task layout vs GPU count.
- Broken NCCL socket/interface routing on multihomed hosts.
- Regressed CUDA/NVIDIA driver compatibility.
- Partial IB/HCA activation after node provisioning.

`ai-cluster-validator` is a compact preflight harness that surfaces these failures with one Slurm submit.

## Tested baseline environment

The following environment is validated and reflected in the sample output below.

| Component | Value |
| --- | --- |
| CycleCloud | `8.8.3-3667` |
| Slurm | `25.05.5` |
| Slurm Partition | `hpc` |
| Scheduler VM SKU | `Standard_D8s_v6` |
| Compute VM SKU | `Standard_ND96asr_v4` |
| OS Images Tested | `microsoft-dsvm:ubuntu-hpc:2204:latest`, `microsoft-dsvm:ubuntu-hpc:2404:latest` |
| PyTorch | `2.12.0+cu130` |
| CUDA Runtime | `13.0` |
| NCCL Fabric Target | `2.29.7` |

## Repository layout

- `bootstrap_env.sh`: Creates shared Python virtual environment under `/shared/apps/pytorch_env` and installs required Python packages.
- `ddp_smoke_test.slurm`: Slurm job launcher that configures distributed environment variables and starts one rank per GPU.
- `ddp_mesh_ping.py`: Distributed validation workload. Collects node/GPU/network inventory and executes DDP all-reduce validation.

## End-to-end execution

### 1) Clone

```bash
git clone https://github.com/vinil-v/ai-cluster-validator.git
cd ai-cluster-validator
```

### 2) Bootstrap userspace runtime

```bash
sudo bash bootstrap_env.sh
```

What the bootstrap does:

1. Creates `/shared/apps/pytorch_env` (shared location expected across compute nodes).
2. Creates a clean venv with `python3 -m venv`.
3. Upgrades `pip`, `setuptools`, `wheel`.
4. Installs `torch`, `torchvision`, `torchaudio`, and `psutil`.

### 3) Submit smoke test job

```bash
sbatch ddp_smoke_test.slurm
squeue
```

Example scheduler response:

```text
Submitted batch job 2
			 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
				 2       hpc ddp_smok azureuse CF       0:03      2 ddpcluster-hpc-[1-2]
```

### 4) Inspect log output

```bash
cat ai_infra_smoke_test_<jobid>.log
```

Success marker (must appear):

```text
SUCCESS: DDP Multi-Node AllReduce Ring Complete!
```

## How the validation works (technical deep dive)

## 1. Slurm launch topology

`ddp_smoke_test.slurm` requests:

- `--nodes=2`
- `--ntasks-per-node=8`
- `--gpus-per-node=8`
- `--cpus-per-task=12`

This yields:

- `WORLD_SIZE = SLURM_NTASKS = 16` total distributed ranks.
- Exactly one process per GPU on ND96asr_v4 (8x A100).

Ranks are launched with:

```bash
srun --cpu-bind=none bash -c "... python3 ddp_mesh_ping.py"
```

`--cpu-bind=none` avoids accidental CPU pinning constraints that can conflict with NCCL or cluster-specific topology hints.

## 2. DDP rendezvous and process-group bring-up

`ddp_smoke_test.slurm` sets:

- `MASTER_ADDR`: first hostname in `$SLURM_JOB_NODELIST`.
- `MASTER_PORT`: dynamically chosen free port from high ephemeral range `49152-65535`, with fallback `29500`.
- `RANK`: from `$SLURM_PROCID`.
- `LOCAL_RANK`: from `$SLURM_LOCALID`.

`ddp_mesh_ping.py` initializes:

```python
dist.init_process_group(
	backend="nccl",
	init_method=f"tcp://{master_addr}:{master_port}",
	world_size=world_size,
	rank=rank
)
torch.cuda.set_device(local_rank)
```

This confirms that Slurm rank metadata, TCP rendezvous, CUDA device selection, and NCCL backend initialization are all coherent.

## 3. Fabric and affinity controls

The job exports:

- `NCCL_DEBUG=WARN`
- `NCCL_IB_DISABLE=0` (IB enabled)
- `NCCL_P2P_DISABLE=0` (GPU peer path enabled)
- `NCCL_IGNORE_CPU_AFFINITY=1`
- `GLOO_SOCKET_IFNAME=eth0`
- `NCCL_SOCKET_IFNAME=eth0`

This creates predictable network behavior in multihomed environments and keeps logs concise while still exposing critical warnings.

## 4. Node-level telemetry captured per rank

Each rank emits structured metadata gathered from userspace:

- Node identity: `SLURMD_NODENAME`/hostname.
- GPU details: model string and VRAM size from CUDA properties.
- System memory via `psutil`.
- CPU model via `/proc/cpuinfo`.
- OS version via `/etc/os-release`.
- Kernel version via `platform.release()`.
- NVIDIA driver version via `/proc/driver/nvidia/version`.
- PyTorch/CUDA/NCCL runtime versions.
- IB HCA state/rate via `/sys/class/infiniband/*/ports/*/{state,rate}`.
- Basic peer access capability via `torch.cuda.can_device_access_peer`.

All rank objects are gathered on rank 0 using `dist.gather_object(...)` and rendered as a consolidated cluster report.

## 5. Functional distributed test

After telemetry collection, each rank executes a simple DDP training step:

1. Build `nn.Linear(10, 10)` on local GPU.
2. Wrap in `DistributedDataParallel`.
3. Run forward + MSE loss + backward.
4. Execute `dist.all_reduce(loss_tensor, SUM)` across all ranks.
5. Compute global average loss as a deterministic completion signal.

This validates that collective communications and gradient synchronization are operational end-to-end.

## Expected output interpretation

Given two ND96asr_v4 nodes, expected topology section should show:

- `16` ranks total.
- Node 1: local ranks `0..7`.
- Node 2: local ranks `0..7`.
- GPU model typically `NVIDIA A100-SXM4-40GB` (name may be truncated in table formatting).

Environment deep dive should confirm:

- Correct OS/kernel and NVIDIA driver loaded.
- PyTorch/CUDA/NCCL versions aligned with image/runtime.
- IB ports in `ACTIVE` state (HDR/QDR as provisioned).

Network section should reflect selected interface (`eth0`) and include the terminal pass condition:

```text
SUCCESS: DDP Multi-Node AllReduce Ring Complete!
```

## Customization guide

To adapt this smoke test to another cluster shape:

1. Edit Slurm directives in `ddp_smoke_test.slurm` (`--nodes`, `--ntasks-per-node`, `--gpus-per-node`, partition).
2. Ensure shared venv path in `bootstrap_env.sh` and activation path in `ddp_smoke_test.slurm` are valid cluster-wide.
3. Set `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to your management or data-plane interface policy.
4. Re-run and confirm `WORLD_SIZE` and topology table match physical expectations.

## Operational use cases

- New cluster bring-up acceptance test.
- Post-maintenance validation after driver/CUDA/NCCL upgrades.
- Regression gate in image lifecycle pipelines.
- Fast preflight before large-scale distributed training jobs.

## Notes

- This tool is intentionally lightweight and userspace-focused. It does not benchmark performance; it validates correctness and connectivity.
- The included DDP step is a smoke test, not a training benchmark.
- For large clusters, run this as a first gate before deeper perf tools (NCCL-tests, allreduce microbenchmarks, model-level scaling runs).

## Example verified run snapshot

From your captured execution:

- Master node: `ddpcluster-hpc-1`
- Dynamic port: `53593`
- World size: `16`
- NCCL version observed: `2.29.7+cuda13.2`
- Final status: `SUCCESS: DDP Multi-Node AllReduce Ring Complete!`

This confirms that Slurm orchestration, multi-node NCCL collectives, GPU mapping, and core fabric discovery are all functioning in the tested environment.

## Complete reference output

Use this full log as a known-good reference for structure and success markers.

```text
Master Node IP/Hostname: ddpcluster-hpc-1
Dynamically Assigned Port: 53593
Total Execution Ranks: 16
===============================================================================================
	HPC CLUSTER INTERACTION MONITOR
===============================================================================================
--> Initializing DDP on Master Node : ddpcluster-hpc-1
--> Dynamic Coordination Port     : 53593
--> Target World Cluster Size      : 16 GPUs
-----------------------------------------------------------------------------------------------

===============================================================================================
											 CLUSTER HARDWARE TOPOLOGY REPORT
===============================================================================================
| Rank | Node Name    | Local ID | GPU Model          | VRAM     | Sys Mem   | CPU Cores |
-----------------------------------------------------------------------------------------------
| 0    | ddpcluster-hpc-1 | 0        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 1    | ddpcluster-hpc-1 | 1        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 2    | ddpcluster-hpc-1 | 2        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 3    | ddpcluster-hpc-1 | 3        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 4    | ddpcluster-hpc-1 | 4        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 5    | ddpcluster-hpc-1 | 5        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 6    | ddpcluster-hpc-1 | 6        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 7    | ddpcluster-hpc-1 | 7        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 8    | ddpcluster-hpc-2 | 0        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 9    | ddpcluster-hpc-2 | 1        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 10   | ddpcluster-hpc-2 | 2        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 11   | ddpcluster-hpc-2 | 3        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 12   | ddpcluster-hpc-2 | 4        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 13   | ddpcluster-hpc-2 | 5        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 14   | ddpcluster-hpc-2 | 6        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
| 15   | ddpcluster-hpc-2 | 7        | NVIDIA A100-SXM4-4 | 39.5 GB  | 885.8 GB  | 96 Cores  |
===============================================================================================
											 NODE ENVIRONMENT DEEP DIVE
-----------------------------------------------------------------------------------------------
[ddpcluster-hpc-1] Details:
	--> CPU Microarchitecture : AMD EPYC 7V12 64-Core Processor
	--> Operating System      : Ubuntu 22.04.5 LTS
	--> Kernel Base Version   : 5.15.0-1110-azure
	--> Nvidia Driver Loaded  : 580.126.20
	--> PyTorch Environment   : v2.12.0+cu130
	--> CUDA Runtime Version  : v13.0
	--> NCCL Fabric Target    : v2.29.7
	--> Discovered InfiniBand HCAs:
				- mlx5_an0:1 (4: ACTIVE - 40 Gb/sec (4X QDR))
				- mlx5_ib0:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib1:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib2:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib3:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib4:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib5:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib6:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib7:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
-----------------------------------------------------------------------------------------------
[ddpcluster-hpc-2] Details:
	--> CPU Microarchitecture : AMD EPYC 7V12 64-Core Processor
	--> Operating System      : Ubuntu 22.04.5 LTS
	--> Kernel Base Version   : 5.15.0-1110-azure
	--> Nvidia Driver Loaded  : 580.126.20
	--> PyTorch Environment   : v2.12.0+cu130
	--> CUDA Runtime Version  : v13.0
	--> NCCL Fabric Target    : v2.29.7
	--> Discovered InfiniBand HCAs:
				- mlx5_an0:1 (4: ACTIVE - 40 Gb/sec (4X QDR))
				- mlx5_ib0:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib1:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib2:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib3:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib4:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib5:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib6:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
				- mlx5_ib7:1 (4: ACTIVE - 200 Gb/sec (4X HDR))
-----------------------------------------------------------------------------------------------
										 NETWORK INTERCONNECT & FABRIC STATUS
-----------------------------------------------------------------------------------------------
--> Target Communication Interface (NCCL_SOCKET_IFNAME) : eth0
--> Active Telemetry Tracking Level (NCCL_DEBUG)       : WARN
--> Inter-GPU Topo Link Verification                 : Active (P2P/NVLink Capable)
-----------------------------------------------------------------------------------------------
 SUCCESS: DDP Multi-Node AllReduce Ring Complete!
--> Computed System Verification Convergence Loss    : 1.398719
===============================================================================================

NCCL version 2.29.7+cuda13.2
```
