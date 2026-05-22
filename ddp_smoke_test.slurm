#!/bin/bash
#SBATCH --job-name=ddp_smoke_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12          
#SBATCH --output=ai_infra_smoke_test_%j.log
#SBATCH --error=ai_infra_smoke_test_%j.err
#SBATCH --partition=hpc

# --- Activate Centralized NFS PyTorch Environment ---
source /shared/apps/pytorch_env/bin/activate

# --- Network & InfiniBand Tuning Layout ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Force NCCL to bypass checking thread affinity bounds when using the Azure topo.xml
export NCCL_IGNORE_CPU_AFFINITY=1

# --- Multihoming Management Interface Controls ---
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# --- PyTorch Master Node Discovery Setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Dynamically select an available open port in the  range (49152-65535)
export MASTER_PORT=$(ss -tln | awk 'NR>1 {print $4}' | awk -F: '{print $NF}' | sort -nu | awk '
    BEGIN { srand(); min=49152; max=65535; range=max-min+1; }
    { used[$1]=1 }
    END {
        for (i=0; i<100; i++) {
            port = int(rand()*range) + min;
            if (!(port in used)) {
                print port;
                exit;
            }
        }
        print 29500; # Fallback default if generation block fails
    }
')

export WORLD_SIZE=$SLURM_NTASKS

echo "Master Node IP/Hostname: $MASTER_ADDR"
echo "Dynamically Assigned Port: $MASTER_PORT"
echo "Total Execution Ranks: $WORLD_SIZE"

# --- Run the Workload ---

srun --cpu-bind=none bash -c "
    source /shared/apps/pytorch_env/bin/activate;
    export RANK=\$SLURM_PROCID;
    export LOCAL_RANK=\$SLURM_LOCALID;
    python3 ddp_mesh_ping.py
"