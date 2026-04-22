# CLIP-Adapter Experiments Report

- workspace: `ClipAdapter`
- datasets: `ClipAdapter/dataset`
- report_imgs: `ClipAdapter/report/imgs`

## clipadapter_mnist2usps_mnist2usps_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:13:49
- dataset: mnist2usps (mnist → usps)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 128
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 20
- results:
  - acc_clip_zeroshot: 0.629796
  - acc_clip_adapter: 0.771799
  - train_time_sec: 1.85
  - train_steps: 21
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_mnist2usps_mnist2usps_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: +0.1420 (higher is better).


## clipadapter_office31_amazon2webcam_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:20:12
- dataset: office31 (amazon → webcam)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 128
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 200
- results:
  - acc_clip_zeroshot: 0.914465
  - acc_clip_adapter: 0.898113
  - train_time_sec: 2.43
  - train_steps: 22
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_office31_amazon2webcam_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: -0.0164 (higher is better).


## clipadapter_svhn2mnist_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:20:25
- dataset: svhn2mnist (svhn → mnist)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 128
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 200
- results:
  - acc_clip_zeroshot: 0.616406
  - acc_clip_adapter: 0.692656
  - train_time_sec: 11.6
  - train_steps: 201
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_svhn2mnist_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: +0.0762 (higher is better).


## clipadapter_pacs_photo2sketch_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:22:56
- dataset: pacs (photo → sketch)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 128
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 200
- results:
  - acc_clip_zeroshot: 0.9155
  - acc_clip_adapter: 0.914737
  - train_time_sec: 1.22
  - train_steps: 11
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_pacs_photo2sketch_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: -0.0008 (higher is better).


## clipadapter_mnist2usps_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:26:15
- dataset: mnist2usps (mnist → usps)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 256
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 0
- results:
  - acc_clip_zeroshot: 0.630294
  - acc_clip_adapter: 0.857997
  - train_time_sec: 15.08
  - train_steps: 234
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_mnist2usps_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: +0.2277 (higher is better).


## clipadapter_svhn2mnist_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:26:21
- dataset: svhn2mnist (svhn → mnist)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 256
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 0
- results:
  - acc_clip_zeroshot: 0.6189
  - acc_clip_adapter: 0.7008
  - train_time_sec: 17.11
  - train_steps: 286
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_svhn2mnist_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: +0.0819 (higher is better).


## clipadapter_officehome_Art2Real_World_lr0.001_alpha0.2_b64
- time: 2026-04-16 15:26:44
- dataset: officehome (Art → Real World)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 64
  - batch_size: 128
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 200
- results:
  - acc_clip_zeroshot: 0.915997
  - acc_clip_adapter: 0.913473
  - train_time_sec: 16.37
  - train_steps: 15
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_officehome_Art2Real_World_lr0.001_alpha0.2_b64.pt
- observations:
  - Adapter vs. zero-shot: -0.0025 (higher is better).


## clipadapter_mnist2usps_lr1e-3_alpha0.5_b64
- time: 2026-04-16 15:37:00
- dataset: mnist2usps (mnist → usps)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.5
  - bottleneck: 64
  - batch_size: 256
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 0
- results:
  - acc_clip_zeroshot: 0.630294
  - acc_clip_adapter: 0.870453
  - train_time_sec: 14.71
  - train_steps: 234
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_mnist2usps_lr1e-3_alpha0.5_b64.pt
- observations:
  - Adapter vs. zero-shot: +0.2402 (higher is better).


## clipadapter_mnist2usps_lr1e-3_alpha0.2_b256
- time: 2026-04-16 15:37:00
- dataset: mnist2usps (mnist → usps)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 256
  - batch_size: 256
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 0
- results:
  - acc_clip_zeroshot: 0.630294
  - acc_clip_adapter: 0.890882
  - train_time_sec: 15.45
  - train_steps: 234
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_mnist2usps_lr1e-3_alpha0.2_b256.pt
- observations:
  - Adapter vs. zero-shot: +0.2606 (higher is better).


## clipadapter_mnist2usps_lr1e-3_alpha0.2_b16
- time: 2026-04-16 15:37:01
- dataset: mnist2usps (mnist → usps)
- config:
  - clip_model: ViT-B-32
  - clip_pretrained: laion2b_s34b_b79k
  - lr: 0.001
  - alpha: 0.2
  - bottleneck: 16
  - batch_size: 256
  - epochs: 1
  - precision: fp16
  - seed: 0
  - max_steps: 0
- results:
  - acc_clip_zeroshot: 0.630294
  - acc_clip_adapter: 0.885899
  - train_time_sec: 15.76
  - train_steps: 234
  - adapter_ckpt: ClipAdapter/checkpoints/clipadapter_mnist2usps_lr1e-3_alpha0.2_b16.pt
- observations:
  - Adapter vs. zero-shot: +0.2556 (higher is better).
