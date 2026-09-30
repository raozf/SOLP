# SOLP

Official release code for **Subspace-Orthogonal Latent Perturbation (SOLP)** for stable long-horizon video generation.

SOLP applies a PCA-defined orthogonal perturbation to the VAE conditioning latents before the DiT backbone. The optional spectral stabilizer is loaded separately at inference time. The repository keeps one training entry point (`train.py`) and one testing entry point (`test.py`); model weights and datasets are intentionally not included.

## Installation

```bash
git clone https://github.com/raozf/SOLP.git
cd SOLP
pip install -r requirements.txt
pip install -e .
```

The released scripts were developed for Wan2.1-I2V-14B-480P with CUDA and mixed-precision training. Check that the local PyTorch/CUDA, `diffusers`, `transformers`, `lightning`, and `peft` versions match the environment used for your run.

## Model and data layout

Download the Wan2.1-I2V-14B-480P model separately and arrange the files as follows:

```text
weights/Wan2.1-I2V-14B-480P/
├── diffusion_pytorch_model-00001-of-00007.safetensors
├── ...
├── diffusion_pytorch_model-00007-of-00007.safetensors
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── Wan2.1_VAE.pth
```

Training expects a directory of videos. A metadata CSV is optional; when present, place it at `data/train/metadata.csv`. Paths are supplied on the command line, so no machine-specific paths are embedded in the release.

## Training

`train.py` contains the SOLP perturbation, temporal PCA residual construction, LoRA fine-tuning, and spectral-stabilizer loss. A minimal single-GPU command is:

```bash
python train.py \\
  --dataset_path data/train \\
  --dit_path "weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors,weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors,weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors,weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors,weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors,weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors,weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors" \\
  --text_encoder_path weights/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \\
  --image_encoder_path weights/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \\
  --vae_path weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \\
  --output_path outputs/solp \\
  --devices 1 --num_nodes 1
```

Add `--pretrained_filter_path PATH` to resume a spectral-stabilizer checkpoint. The script saves `lora.safetensors`, `filter.safetensors`, and the parsed configuration under the output directory.

## Testing

`test.py` accepts a model directory, an optional LoRA checkpoint, and an optional spectral-stabilizer checkpoint. For one reference image and prompt file:

```bash
python test.py \\
  --dit_root weights/Wan2.1-I2V-14B-480P \\
  --ref_image_path data/test_samples/example/frame.png \\
  --prompt_path data/test_samples/example/prompt.txt \\
  --extra_module_root outputs/solp/lora.safetensors \\
  --filter_path outputs/solp/filter.safetensors \\
  --output outputs/videos
```

The filter checkpoint path is deliberately a command-line argument; no local or cluster-specific path is hard-coded in the repository. Use `--data_root` and `--test_samples` to process a directory of benchmark cases.

## Demonstration videos

The following six demonstrations are provided as 4×-speed previews. Each video contains a persistent `4x SPEED` label. The speed-up only shortens playback; it does not change the generated frames.

| Demo | Preview |
|---|---|
| Flower | <video controls muted loop width="320"><source src="assets/videos/01_flower_4x.mp4" type="video/mp4"></video><br>[Open video](assets/videos/01_flower_4x.mp4) |
| Natural short | <video controls muted loop width="320"><source src="assets/videos/02_natural_short_4x.mp4" type="video/mp4"></video><br>[Open video](assets/videos/02_natural_short_4x.mp4) |
| Dog | <video controls muted loop width="320"><source src="assets/videos/03_dog_4x.mp4" type="video/mp4"></video><br>[Open video](assets/videos/03_dog_4x.mp4) |
| Sunset | <video controls muted loop width="320"><source src="assets/videos/04_sunset_4x.mp4" type="video/mp4"></video><br>[Open video](assets/videos/04_sunset_4x.mp4) |
| Natural long | <video controls muted loop width="320"><source src="assets/videos/05_natural_long_4x.mp4" type="video/mp4"></video><br>[Open video](assets/videos/05_natural_long_4x.mp4) |
| Sea | <video controls muted loop width="320"><source src="assets/videos/06_sea_4x.mp4" type="video/mp4"></video><br>[Open video](assets/videos/06_sea_4x.mp4) |

## Reproducibility notes

The repository contains the code used for the manuscript experiments, but does not redistribute Wan2.1 weights, private training data, LoRA checkpoints, or spectral-stabilizer checkpoints. Results in the paper should therefore be interpreted as reported experimental records; running the public scripts with different checkpoints or data may produce different values.

## Citation

```bibtex
@article{rao2026solp,
  title   = {SOLP: Subspace-Orthogonal Latent Perturbation for Stable Long-Horizon Video Generation},
  author  = {Rao, Zhifeng and Zheng, Mingkai},
  journal = {Neural Networks},
  year    = {2026}
}
```
