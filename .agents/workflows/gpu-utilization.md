---
description: GPU Usage On Remote DGX Server
---

# GPU Utilization Guidelines for Agents

## Purpose

Use this document before running GPU-intensive work on the remote DGX / shared compute environment. The goal is to use the GPU assigned to the job efficiently, avoid disrupting other users, and collect enough evidence to diagnose low utilization.

**Treat the machine as shared unless the administrator explicitly confirms otherwise.** Do not assume a GPU is exclusively yours just because `nvidia-smi` lists it.

## 1. Inspect before running

Start with read-only checks. Run them in the same environment and on the same node where the workload will execute.

```bash
hostname
date
nvidia-smi
nvidia-smi -L
```

If supported by the installed driver, inspect utilization and memory periodically:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --loop=2
```

Stop the continuous query with `Ctrl+C`. If a query field is unsupported, use plain `nvidia-smi` instead.

Before running a training or inference job, record:
- Hostname / compute node
- GPU model and index
- GPU memory used and total
- Current GPU utilization
- Existing GPU processes, if visible
- Active Python interpreter and environment
- Whether a scheduler allocation is active

Example Python environment checks:

```bash
which python
python --version
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

If PyTorch is not installed in the project's existing environment, do not install it just to perform this check. Report that the check could not run.

## 2. Respect the cluster's resource-allocation policy

- Check project instructions and the cluster's documented workflow before launching GPU jobs.
- If a workload manager such as Slurm is configured, use the allocation and submission mechanism required by the cluster. Do not bypass the scheduler to obtain a GPU.
- Confirm the scheduler and GPU-request syntax from local documentation or administrators. Do not assume every cluster uses Slurm or supports the same flags.
- On Slurm systems, inspect available commands and job state as appropriate, for example `squeue --me` and `scontrol show job "$SLURM_JOB_ID"` inside an allocated job.
- Do not run long training jobs on a login/head node unless the administrator explicitly permits it.
- Request only the GPU, CPU, memory, and runtime resources the task needs. Avoid requesting all GPUs by default.
- Do not kill, suspend, or interfere with processes belonging to other users. If GPU contention is suspected, record the evidence and contact the administrator.

## 3. Select the correct GPU

Do not hard-code GPU `0` without checking the allocation and visibility rules.

```bash
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
```

Inside scheduled jobs, the scheduler or container runtime may expose only assigned devices. Respect `CUDA_VISIBLE_DEVICES` and the framework's visible-device mapping. A logical `cuda:0` inside a job may not correspond to physical GPU index 0 on the host.

Do not modify global GPU visibility or scheduler settings to get access to another device. If the assigned GPU is unavailable, stop and investigate the allocation rather than silently switching to a device used by someone else.

## 4. Establish a small baseline first

Before full training or processing a large dataset:

1. Run a small representative batch or a short smoke test.
2. Confirm that the model and tensors are actually on the GPU.
3. Confirm that the run completes and produces valid output.
4. Measure elapsed time, peak memory if available, and GPU utilization.
5. Scale up only after the smoke test succeeds.

For PyTorch, inspect the device in the running code:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
```

This snippet assumes PyTorch is already installed. It is a diagnostic, not a reason to alter the environment.

## 5. Diagnose low GPU utilization systematically

Low utilization does not automatically mean that more GPU resources are needed. Check these causes in order:

1. **Is the workload on CUDA?** Confirm the framework reports CUDA available and that model parameters / tensors are placed on the intended device.
2. **Is the GPU assigned and visible?** Check scheduler allocation, `CUDA_VISIBLE_DEVICES`, container settings, and runtime logs.
3. **Is the GPU waiting for input?** Check CPU utilization, data-loader workers, preprocessing, storage throughput, decompression, and network-mounted datasets.
4. **Is the workload too small?** Small batches or tiny models may not keep a large GPU busy. Benchmark batch sizes safely; monitor memory and avoid out-of-memory crashes.
5. **Is synchronization or transfer overhead dominant?** Avoid unnecessary CPU↔GPU copies and excessive per-batch synchronization. Profile before making changes.
6. **Is another workload sharing the device?** Use available process / scheduler information. Do not assume every process is visible to every user.
7. **Is the job still initializing or compiling?** Warm-up, data loading, CUDA initialization, and compilation can produce temporary low utilization.

Make one evidence-based change at a time and compare throughput, memory use, and correctness against the baseline. Do not optimize for a high utilization percentage at the expense of valid results or total throughput.

## 6. Monitor and record the run

For a short diagnostic sample:

```bash
nvidia-smi
```

For a supported driver, a CSV snapshot can be saved for later comparison:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

For long-running jobs, prefer the cluster's approved monitoring/logging tools or capture periodic samples in the job script. Avoid unbounded monitoring loops that generate excessive logs.

Record:
- Command / job ID and start time
- Hostname and assigned device(s)
- Dataset subset and batch size
- Framework and CUDA runtime versions, when available
- Throughput (for example, samples/second or batches/second)
- GPU utilization and memory observations
- Exit status, logs, and validation results

## 7. Safety rules: never do these without explicit authorization

- **Do not change GPU compute mode** using `nvidia-smi -c` or equivalent. Compute-mode changes can affect other users and may require administrative permissions.
- Do not reset GPUs, alter power limits or clocks, change persistence mode, or run privileged GPU-management commands.
- Do not kill other users' GPU processes or change their jobs.
- Do not install, upgrade, or remove CUDA, drivers, PyTorch, or system packages just to improve utilization.
- Do not recreate the project's virtual environment or change Python versions without an approved, evidence-based plan.
- Do not start full-dataset processing or long training runs before a small validation run.
- Do not claim that the GPU was used merely because CUDA is installed or `nvidia-smi` works. Verify device placement and workload behavior.

The NVIDIA Base Command Manager 11 manual describes `nvidia-smi` GPU compute-mode changes, including exclusive and prohibited modes. These settings are system-wide GPU behavior, not ordinary per-job tuning; agents must not alter them as a routine optimization.

## 8. Agent decision procedure

Before a GPU workload:

1. Read this file and the project's own instructions.
2. Inspect the host, visible GPUs, existing processes, environment, and scheduler state using read-only commands.
3. Determine the approved resource-allocation method.
4. If the allocation or ownership of a GPU is unclear, stop and ask rather than guessing.
5. Run a small smoke test and verify correctness and device placement.
6. Measure baseline utilization, memory, and throughput.
7. Change one bottleneck at a time and re-measure.
8. Stop if the GPU is unavailable, memory is insufficient, other users may be affected, or privileged configuration changes appear necessary.
9. Report evidence, commands, results, and remaining uncertainty.

## 9. Completion report template

```text
GPU / node:
Allocation method / job ID:
Environment:
Workload and dataset subset:
Device-placement evidence:
GPU utilization / memory observations:
Baseline throughput:
Changes made:
Validation result:
Commands run:
Outstanding issues:
```

## Source and environment note

These guidelines are conservative defaults for agents working on a shared GPU system. The uploaded NVIDIA Base Command Manager 11 User Manual (revision dated 2025-11-21) covers GPU tools in section 7.5.2 and workload management / Slurm in chapters 4–5. Actual cluster configuration and administrator policy take precedence. Confirm local scheduler details and available commands before using them.
