#!/usr/bin/env python3
import os
import sys
import socket
import platform
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def get_os_release():
    """Extracts a clean OS version string from /etc/os-release safely."""
    try:
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=")[1].strip().strip('"')
        return platform.system()
    except Exception:
        return "Unknown OS"

def get_cpu_model():
    """Extracts the specific processor architecture model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        return platform.processor()
    except Exception:
        return "Unknown CPU Model"

def get_nvidia_driver_version():
    """Reads the driver version string from the proc kernel module tree."""
    driver_path = "/proc/driver/nvidia/version"
    try:
        if os.path.exists(driver_path):
            with open(driver_path, "r") as f:
                first_line = f.readline()
                parts = first_line.split("x86_64")
                if len(parts) > 1:
                    return parts[1].split()[0].strip()
        return "Unknown"
    except Exception:
        return "Unknown"

def get_active_ib_devices():
    """Queries the kernel sysfs layers to gather physical InfiniBand status."""
    ib_dir = "/sys/class/infiniband"
    devices_status = []
    
    if not os.path.exists(ib_dir):
        return ["None Detected (sysfs path missing)"]
        
    try:
        devices = sorted(os.listdir(ib_dir))
        for dev in devices:
            ports_dir = os.path.join(ib_dir, dev, "ports")
            if os.path.exists(ports_dir):
                for port in os.listdir(ports_dir):
                    state_path = os.path.join(ports_dir, port, "state")
                    rate_path = os.path.join(ports_dir, port, "rate")
                    
                    state = "Unknown"
                    rate = "Unknown"
                    
                    if os.path.exists(state_path):
                        with open(state_path, "r") as f:
                            state = f.read().strip()
                    if os.path.exists(rate_path):
                        with open(rate_path, "r") as f:
                            rate = f.read().strip()
                            
                    devices_status.append(f"{dev}:{port} ({state} - {rate})")
        
        return devices_status if devices_status else ["No active ports found"]
    except Exception as e:
        return [f"Error scanning IB devices: {str(e)}"]

def get_nvlink_status(local_rank):
    """Checks if basic P2P/NVLink capability is active for this device pair."""
    try:
        if torch.cuda.is_available():
            current_dev = local_rank
            peer_dev = (local_rank + 1) % torch.cuda.device_count()
            if current_dev != peer_dev and torch.cuda.can_device_access_peer(current_dev, peer_dev):
                return "Active (P2P/NVLink Capable)"
        return "Internal Link Only"
    except Exception:
        return "Unknown"

def setup_distributed():
    """Initializes the distributed environment using Slurm environment variables."""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    if rank == 0:
        print("=" * 95)
        print("  HPC CLUSTER INTERACTION MONITOR")
        print("=" * 95)
        print(f"--> Initializing DDP on Master Node : {master_addr}")
        print(f"--> Dynamic Coordination Port     : {master_port}")
        print(f"--> Target World Cluster Size      : {world_size} GPUs")
        print("-" * 95)
        sys.stdout.flush()

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def run_test():
    rank, local_rank, world_size = setup_distributed()
    
    # --- Local Metadata Gathering ---
    nodename = os.environ.get('SLURMD_NODENAME', socket.gethostname())
    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_mem = f"{torch.cuda.get_device_properties(local_rank).total_memory / (1024**3):.1f} GB"
    
    # System RAM mapping
    sys_mem_total = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    
    # Kernel & OS Information
    kernel_ver = platform.release()
    os_ver = get_os_release()
    cpu_model = get_cpu_model()
    
    # Driver & Toolkits
    nv_driver = get_nvidia_driver_version()
    cuda_ver = torch.version.cuda
    pytorch_ver = torch.__version__
    
    # NCCL Runtime Parsing
    try:
        nccl_ver = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        nccl_ver = "Unknown"
        
    ib_env = os.environ.get('NCCL_SOCKET_IFNAME', 'Not Explicitly Set')
    nccl_debug_level = os.environ.get('NCCL_DEBUG', 'None')
    
    try:
        cpu_affinity = len(os.sched_getaffinity(0))
        cpu_slice = f"{cpu_affinity} Cores"
    except AttributeError:
        cpu_slice = "Unknown"

    nvlink_info = get_nvlink_status(local_rank)
    ib_devices = get_active_ib_devices()

    # Pack local node info to pass to Rank 0
    local_data = {
        'rank': rank,
        'node': nodename,
        'l_rank': local_rank,
        'gpu': gpu_name,
        'vram': gpu_mem,
        'sys_mem': sys_mem_total,
        'cores': cpu_slice,
        'cpu_model': cpu_model,
        'kernel': kernel_ver,
        'os': os_ver,
        'driver': nv_driver,
        'cuda': cuda_ver,
        'pytorch': pytorch_ver,
        'nccl': nccl_ver,
        'nvlink': nvlink_info,
        'ib_devices': ib_devices
    }

    # Gather data from all ranks onto Rank 0
    gather_list = [None] * world_size if rank == 0 else None
    dist.gather_object(local_data, gather_list, dst=0)

    # --- Simple DDP Computational Block ---
    torch.manual_seed(42 + rank)
    model = nn.Linear(10, 10).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    inputs = torch.randn(20, 10).to(local_rank)
    targets = torch.randn(20, 10).to(local_rank)
    loss_fn = nn.MSELoss()
    
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    
    loss_tensor = loss.detach().clone()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    dist.barrier()

    # --- Structured Presentation Report Generation ---
    if rank == 0:
        print("\n" + "=" * 95)
        print("                       CLUSTER HARDWARE TOPOLOGY REPORT")
        print("=" * 95)
        print(f"| {'Rank':<4} | {'Node Name':<12} | {'Local ID':<8} | {'GPU Model':<18} | {'VRAM':<8} | {'Sys Mem':<9} | {'CPU Cores':<9} |")
        print("-" * 95)
        
        for data in sorted(gather_list, key=lambda x: x['rank']):
            print(f"| {data['rank']:<4} | {data['node']:<12} | {data['l_rank']:<8} | {data['gpu'][:18]:<18} | {data['vram']:<8} | {data['sys_mem']:<9} | {data['cores']:<9} |")
        
        print("=" * 95)
        print("                       NODE ENVIRONMENT DEEP DIVE")
        print("-" * 95)
        
        # Track unique nodes to output distinct system environments cleanly
        printed_nodes = set()
        for data in sorted(gather_list, key=lambda x: x['rank']):
            if data['node'] not in printed_nodes:
                print(f"[{data['node']}] Details:")
                print(f"  --> CPU Microarchitecture : {data['cpu_model']}")
                print(f"  --> Operating System      : {data['os']}")
                print(f"  --> Kernel Base Version   : {data['kernel']}")
                print(f"  --> Nvidia Driver Loaded  : {data['driver']}")
                print(f"  --> PyTorch Environment   : v{data['pytorch']}")
                print(f"  --> CUDA Runtime Version  : v{data['cuda']}")
                print(f"  --> NCCL Fabric Target    : v{data['nccl']}")
                print(f"  --> Discovered InfiniBand HCAs:")
                for dev_str in data['ib_devices']:
                    print(f"        - {dev_str}")
                print("-" * 95)
                printed_nodes.add(data['node'])

        print("                     NETWORK INTERCONNECT & FABRIC STATUS")
        print("-" * 95)
        print(f"--> Target Communication Interface (NCCL_SOCKET_IFNAME) : {ib_env}")
        print(f"--> Active Telemetry Tracking Level (NCCL_DEBUG)       : {nccl_debug_level}")
        print(f"--> Inter-GPU Topo Link Verification                 : {gather_list[0]['nvlink']}")
        print("-" * 95)
        print(" SUCCESS: DDP Multi-Node AllReduce Ring Complete!")
        print(f"--> Computed System Verification Convergence Loss    : {global_avg_loss:.6f}")
        print("=" * 95 + "\n")
        sys.stdout.flush()

    cleanup()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA architecture runtime is missing on this node execution block.")
        sys.exit(1)
    run_test()