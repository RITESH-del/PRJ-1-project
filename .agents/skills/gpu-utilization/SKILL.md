---
name: gpu-utilization
description: Help users inspect, request, and troubleshoot NVIDIA GPU usage on an NVIDIA Base Command Manager 11 cluster. Use this skill whenever a user asks about GPU utilization, GPU availability, GPU allocation, CUDA jobs, low GPU usage, nvidia-smi, Slurm/PBS GPU requests, BCM User Portal monitoring, PromQL resource reports, or GPU-backed Jupyter kernels—even if they do not use the phrase “GPU utilization.” Give manual-grounded commands and distinguish scheduler allocation from runtime utilization; do not invent cluster-specific partitions, metrics, or permissions.
compatibility: Requires a BCM 11 cluster account for live checks; commands may require a login/compute node, CUDA module, Slurm or PBS module, and permissions configured by the cluster administrator.
---

# GPU utilization on BCM 11

Use this skill as a cautious runbook for NVIDIA Base Command Manager 11 user environments. The supplied BCM User Manual explains how to allocate GPUs, use `nvidia-smi` for device state and compute mode, and view device/job measurables through the User Portal. It does **not** define one universal utilization metric name or guarantee that every cluster exposes the same scheduler resources. Treat local configuration as authoritative.

## First classify the request

Identify which of these the user needs:

1. **Is a GPU visible?** Check from the compute context where the job runs, not only from a login node.
2. **Is a GPU allocated to my job?** Inspect the scheduler request and job details.
3. **How busy is the GPU?** Use `nvidia-smi` if available, or a User Portal Monitoring/Accounting dashboard configured by the administrator.
4. **Why is utilization low?** Separate allocation, visibility, memory use, compute mode, application behavior, and scheduler placement.
5. **How do I request a GPU?** Provide the scheduler-specific resource request and a minimal verification step.

Ask only for missing facts that change the command: scheduler (`Slurm`, `PBS`, or Kubernetes/Jupyter), whether the command runs inside an allocated job, and the GPU index or node if relevant. Never ask for passwords, tokens, or private cluster credentials.

## Safety and evidence rules

- Do not run GPU-mutating commands unless the user explicitly asks for an administrative/device-state change and has permission. In particular, `nvidia-smi -i <id> -c <mode>` changes compute mode and can affect other users.
- Prefer read-only inspection: `nvidia-smi -L`, `nvidia-smi`, scheduler status/detail commands, and portal dashboards.
- Do not infer utilization from memory allocation. Allocated memory, active processes, and compute utilization answer different questions.
- Do not assume GPU numbering is stable across nodes. Use the scheduler’s allocation and the node-local device list.
- Do not assume the examples’ CUDA versions, partitions, queues, GPU resource names, or PromQL metric names exist on the user’s cluster. Tell the user what must be adapted.
- If a command is unavailable, report the missing tool and use the appropriate alternate path; do not recommend installing software on a managed cluster without administrator guidance.

## Environment and CUDA visibility

For interactive or batch work, run checks in the same execution environment as the application. The manual describes loading the `cuda<version>/toolkit` module for general CUDA use; the exact version is cluster-specific.

```bash
module avail
module load cuda<version>/toolkit   # replace with a version installed on this cluster
nvidia-smi -L
nvidia-smi
```

`nvidia-smi -L` lists visible GPUs. `nvidia-smi` provides the device and process state exposed by the installed NVIDIA driver. If no GPU is listed, check that the command is running on an allocated GPU node, that the scheduler request succeeded, and that the cluster’s CUDA/driver modules are configured. Do not conclude that the application is using the GPU merely because a GPU is visible.

For a time series, use the `nvidia-smi` query options supported by the installed driver, or use the BCM User Portal Monitoring mode. The manual does not prescribe a single query field, sampling interval, or dashboard metric name, so first inspect `nvidia-smi --help-query-gpu`/`nvidia-smi --help` on the target cluster and confirm the reported field’s meaning. If the cluster exposes Prometheus metrics, use the administrator-provided PromQL query in Accounting and Reporting mode rather than guessing a metric name.

## Slurm: request, inspect, and monitor

The manual’s Slurm examples use a GPU resource directive of the form `--gres=gpu:<count>` and show `--gres=gpu:<number>` in the job-script options. Site configuration may use a different GRES name or additional constraints; verify with the administrator or `sbatch --help`.

Minimal batch pattern:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gpu-check
#SBATCH --partition=<partition>       # omit if the cluster has a default
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gpu-check-%j.out

module load cuda<version>/toolkit
hostname
nvidia-smi -L
nvidia-smi
python your_program.py
```

Submit and inspect the job with the site’s Slurm commands:

```bash
sbatch gpu-check.slurm
squeue -j <job_id>
scontrol show job <job_id>
```

Use `scontrol show job` to verify the job’s allocated node/GRES rather than assuming the request was honored. Once the job is running, run `nvidia-smi` from within that allocation (for example with the site’s `srun` workflow) or inspect the output log. A login-node result is not evidence about the compute node running the job.

## PBS Professional/OpenPBS: request, inspect, and monitor

The manual describes GPU resources as administrator-configured PBS attributes. Its examples include `ngpu=1` for one GPU per node and `gpu_id=0` to request a specific device ID. Resource names and queue policy are site-specific.

Illustrative PBS pattern:

```bash
#!/usr/bin/env bash
#PBS -N gpu-check
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=2:ngpus=1
#PBS -o gpu-check.out
#PBS -e gpu-check.err

module load cuda<version>/toolkit
hostname
nvidia-smi -L
nvidia-smi
python your_program.py
```

Submit and inspect with the site’s PBS commands:

```bash
qsub gpu-check.pbs
qstat -a
qstat -f <job_id>
```

If `ngpus`, `gpu_id`, or `select` is rejected, do not silently translate it into an arbitrary resource name. Ask for the cluster’s configured PBS GPU resource definitions. The manual also documents `module add head pbpo` before PBS commands on a default-style BCM setup; use the local module setup if it differs.

## BCM User Portal

Use the User Portal when the user needs historical or dashboard-based utilization rather than a one-shot node check:

1. Open the portal URL supplied by the cluster administrator.
2. In **Monitoring** mode, expand/filter the measurable tree and drag a device or cluster measurable into a plot panel. The manual gives “occupation” as an example filter term, not a universal GPU metric name.
3. In **Accounting and Reporting** mode, run administrator-approved PromQL queries for per-user/account resource use. Export results to CSV or Excel when needed.
4. Use the **Workload** page for job state and the **Nodes** page to confirm which nodes have GPUs; the Nodes page reports GPUs when present.

State the time range, node/job/user scope, and metric definition in the result. A portal plot is only meaningful if the measurable’s unit and aggregation are known.

## Diagnosing low utilization

Use this order so scheduler and application problems are not conflated:

1. **Allocation:** scheduler detail shows the requested GPU resource and assigned node.
2. **Visibility:** inside the job, `nvidia-smi -L` lists the expected device(s).
3. **Process:** `nvidia-smi` shows whether the application process is present on the expected GPU.
4. **Compute mode:** check the device’s compute mode. BCM documents mode `0` as default/multiple applications, `1` as exclusive-thread, `2` as prohibited, and `3` as exclusive-process. Do not change it without authorization.
5. **Application:** confirm the framework selected CUDA, data loading is not starving the device, and synchronization/batch sizing is appropriate. These application checks require evidence from the program, not the BCM manual alone.
6. **Measurement:** confirm that the portal metric or `nvidia-smi` field is a utilization measure rather than memory usage, allocation, or a short-lived sample.

When reporting a diagnosis, include exact commands run, node/job context, timestamp, GPU index, and the observed output. Mark cluster-specific assumptions explicitly.

## Jupyter and Kubernetes boundary

BCM’s Jupyter Kernel Creator can expose template options for workload manager, queue, number/type of GPUs, and environment modules. For a GPU-backed notebook, use the administrator-provided GPU kernel template and verify inside the kernel with `nvidia-smi -L`; do not edit a shared kernel template unless authorized.

For Kubernetes, GPU requests and device-plugin behavior are administrator-configured and are outside the manual’s end-user GPU command examples. Use the provided BCM/Kubernetes template or manifest conventions, then verify inside the pod. Do not convert a Slurm/PBS request into an invented Kubernetes resource key.

## Response format

For a user-facing answer, use this compact structure:

1. **Context:** scheduler/execution location and what “utilization” means here.
2. **Commands:** copy-pasteable commands with placeholders clearly marked.
3. **What to verify:** allocation, visibility, process, and metric interpretation.
4. **If it fails:** the next diagnostic branch, preserving scheduler/site uncertainty.
5. **Safety note:** only when a command changes device state or affects other users.

Cite the relevant manual section/page in prose: GPU tools (7.2, 7.5.2, pages 55–58), Slurm GPU directives (5.3.2, page 39), PBS GPU resource examples (6.1.2, pages 45–47), User Portal monitoring/accounting (10.5–10.6, pages 83–84), and Jupyter GPU template customization (11.4, page 94). The manual revision is dated November 21, 2025; flag that local cluster configuration may be newer or different.
