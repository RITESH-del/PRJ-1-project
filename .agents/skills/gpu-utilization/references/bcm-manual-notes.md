# BCM 11 GPU reference notes

Source: NVIDIA Base Command Manager 11 User Manual, revision `d7cd1f0f8`, dated 2025-11-21.

These notes preserve the manual facts used by `SKILL.md`; they are intentionally not a replacement for the cluster administrator’s local configuration.

## GPU and CUDA

- Chapter 7 introduces general-purpose GPU use and states that CUDA toolkit modules are cluster-installed and version-dependent (section 7.2, pp. 55–56).
- `nvidia-smi` is documented as the NVIDIA System Management Interface utility (section 7.5.2, pp. 57–58).
- `nvidia-smi -L` lists GPUs.
- The manual demonstrates compute-mode changes with `nvidia-smi -i 0 -c 1`, describes modes 0–3, and releases exclusivity with `nvidia-smi -i 0 -c 0`. These are device-state changes and require authorization.
- The manual’s CUDA example loads a CUDA toolkit module, compiles with `nvcc`, and verifies execution on a GPU node.

## Slurm

- Slurm job-script directives include a GPU request in the form `--gres=gpu:<number>` (section 5.3.2, p. 39).
- The manual shows job submission with `sbatch`, queue inspection with `squeue`, and job details through Slurm status tools (sections 5.3.4–5.3.5, pp. 40–41).
- Partitions, module names, GPU resource names, and policy are cluster-specific.

## PBS Professional/OpenPBS

- PBS resource examples include GPU attributes such as `ngpu=1` and `gpu_id=0`; GPU resources are preconfigured by the administrator (section 6.1.2, pp. 45–47).
- PBS jobs are submitted with `qsub`; status and details are inspected with `qstat` (section 6.2, pp. 49–52).
- The manual’s default-style setup loads `head` and `pbpo` modules before PBS use, but local deployments may differ.

## User Portal

- The Nodes page reports GPUs when present (section 10.3, pp. 81–82).
- Monitoring mode provides plot panels populated by dragging measurables from a filterable tree; “occupation” is an example filter term, not a guaranteed GPU metric name (section 10.5, p. 83).
- Accounting and Reporting mode runs PromQL queries and supports tables, charts, CSV, and Excel export (section 10.6, pp. 83–84).

## Jupyter

- Jupyter Kernel Creator templates can expose workload manager, queue, number/type of GPUs, and module customization options (section 11.4, p. 94).
- Use the administrator-provided template and verify GPU visibility inside the running kernel.

## Interpretation boundary

The manual does not specify one universal GPU utilization query, Prometheus metric name, sampling interval, or scheduler resource name for all BCM clusters. Answers must distinguish GPU allocation, device visibility, process presence, memory use, compute mode, and runtime compute utilization, and must label site-specific values as placeholders.
