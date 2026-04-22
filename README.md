# CLIP-Adapter

This folder contains a minimal CLIP-Adapter implementation and scripts for benchmarking across several domain adaptation settings as part of **NTU MSAI 6126 Project II – Output Space UDA and LMM Transfer**. 
The code is written from scratch based on the original [CLIP-Adapter](https://arxiv.org/abs/2110.04544) paper and simplified for educational purposes.


## Project Structure

- Datasets: `./dataset`
- Report: `./report/report.md`
- Figures: `./report/imgs`

## Environment

Create (or reuse) conda env `clip_adapter`:

```bash
conda create -n clip_adapter --clone cyclegan -y
conda run -n clip_adapter python -m pip install -U open_clip_torch pyarrow datasets
```

## Running Experiments

The main entry is `train_clip_adapter.py`. It logs results into the report automatically.

Example (MNIST → USPS, quick sanity run):

```bash
conda run -n clip_adapter python train_clip_adapter.py \
  --dataset mnist2usps --source mnist --target usps \
  --lr 1e-3 --alpha 0.2 --bottleneck 64 \
  --epochs 1 --batch_size 128 --max_steps 200 \
  --device cuda:0 --precision fp16 --save_ckpt
```

For required benchmarks:

- MNIST → USPS
- SVHN → MNIST
- Office-31 (amazon → webcam)
- Office-Home (Art → Real_World)
- PACS (photo → sketch)

## Evaluation Only

Use `eval_clip_adapter.py` to evaluate zero-shot and (optionally) a trained adapter checkpoint:

```bash
conda run -n clip_adapter python eval_clip_adapter.py \
  --dataset office31 --source amazon --target webcam \
  --ckpt ./checkpoints/<exp>.pt \
  --device cuda:0 --precision fp16
```
