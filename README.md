# FashionMNIST Training and Profiling

This project trains a simple PyTorch classifier on FashionMNIST, saves the trained weights to `model.pth`, and runs a quick single-sample inference check at the end of `main.py`.

## Project Layout

- `main.py`: entry point that downloads data, trains the model, saves weights, and runs a sample prediction
- `app/data.py`: dataset and dataloader helpers
- `app/model.py`: model definition plus save/load helpers
- `app/train.py`: training and evaluation loops
- `app/utils.py`: device selection, class labels, and NVTX helpers for profiling
- `reports/`: profiling output and related artifacts

## Requirements

- Python 3.14 or newer
- PyTorch and TorchVision
- CUDA-capable GPU if you want to use the profiling command below

The repository uses the PyTorch CUDA wheel index defined in `pyproject.toml`.

## Install

If you are using `uv`, install dependencies with:

```bash
uv sync
```

If you prefer a manual environment, make sure `torch` and `torchvision` are installed with a compatible CUDA build.

## Run Training

From the project root:

```bash
uv run main.py
```

What the script does:

1. Downloads FashionMNIST into `data/` if needed.
2. Builds a small fully connected network.
3. Trains for 5 epochs.
4. Saves the weights to `model.pth`.
5. Loads the saved model and prints one prediction example.

## Nsight Systems Profiling

Use the following command to profile the script with Nsight Systems inside the NVIDIA PyTorch container. It uses `python3` because the NVIDIA PyTorch image already includes it:

```bash
docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --rm -it \
  --mount type=bind,src=$(pwd),dst=/workspace/test \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --trace-fork-before-exec=true \
    -o /workspace/test/reports/nsys_report \
    python3 /workspace/test/main.py
```

The profile output is written under `reports/` in the workspace.

### View Profiling Results

Once profiling is complete, visualize the results with:

```bash
nsys-ui ./reports/profile_output.nsys-rep
```

## Nsight Compute Profiling

You can profile kernel-level metrics with Nsight Compute (`ncu`). The project already emits NVTX ranges (see `app/utils.py`), which makes the `ncu` timeline easier to interpret.

Example: run `ncu` inside the NVIDIA PyTorch container and write output under `reports/ncu`.

```bash
docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --rm -it \
  --mount type=bind,src=$(pwd),dst=/workspace/test \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  ncu --target-processes all --set full -o /workspace/test/reports/ncu/ncu_report \
    python3 /workspace/test/main.py
```

Or use the included lightweight wrapper from the project root (requires `ncu` available locally):

```bash
./scripts/profile_ncu.sh
```

To stop a long run automatically, set `NCU_TIMEOUT` before launching the wrapper:

```bash
NCU_TIMEOUT=30m ./scripts/profile_ncu.sh
```

A similar wrapper exists for Nsight Systems:

```bash
./scripts/profile_nsys.sh
```

Both wrapper scripts prefer running the project under the local `uv` virtual environment (using `uv run ...`) when `uv` is available; otherwise they fall back to `python3`.

The generated report prefix will be under `reports/ncu/` (e.g. `reports/ncu/ncu_report.ncu-rep`). Use the Nsight Compute UI to open the `.ncu-rep` file for detailed kernel metrics and source correlation.

## Notes

- The training loop emits NVTX ranges, so the profiling timeline is easier to read.
- The script falls back to CPU if no accelerator is available, but profiling is most useful with a GPU.
