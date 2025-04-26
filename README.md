# Wan2.1 Text-to-Video Model for Mac M-Series

This repository has the sole purpose of making Wan2.1 run efficiently on Mac M-Series chips. Distributed inference and various things are broken, but the memory savings are real.

## Original problems

Mac M-Series chips have Unified Memory, so the original method of offloading models to the CPU still costs memory.

The original repo also loads all models at startup, which takes a lot of memory. (umt5-xxl is a 13B model!!)

## Changes

- Load models only when needed. (T5, base model, and vae)
- Modify the offload_model method to delete the model from memory immediately after use.
- Add VAE tiling to reduce memory usage. From [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
- Add quantized T5 model to reduce memory usage.
- Enable mixed precision for MPS, reducing memory usage and increasing speed.
- Experimental support for FLF2V model.

## Installation

Follow the upstream instructions to install the dependencies and download the model.

Assuming you have Poetry installed, you can also install the dependencies with:

```bash
poetry install
```

If you want to use hugingface-cli or modelscope, you can install with:

```bash
poetry install --extras dev
```

Download the model with huggingface-cli or modelscope:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

```bash
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```

To use quantized T5 model, [download it](https://huggingface.co/HighDoping/umt5-xxl-encode-gguf/resolve/main/umt5-xxl-encode-only-Q4_K_M.gguf) from my [🤗 repo](https://huggingface.co/HighDoping/umt5-xxl-encode-gguf) or use huggingface-cli and put it in the same folder as wan model:

```bash
huggingface-cli download HighDoping/umt5-xxl-encode-gguf --local-dir ./Wan2.1-T2V-1.3B
```

[Models from city96](https://huggingface.co/city96/umt5-xxl-encoder-gguf) also works. Only needs to change the model name in ```wan\configs```

Then install llama.cpp from homebrew: (Note llama.cpp version b4882 to b4974 don't support T5Encoder model, check [issue #2](https://github.com/HighDoping/Wan2.1/issues/2) for workaround)

```bash
brew install llama.cpp
```

## Usage

To generate a video, use the following command:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-1.3B --size "832*480" --frame_num 17 --sample_steps 25 --tile_size 256 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_quant --device mps --sample_shift 8 --sample_guide_scale 6 --prompt "Penguins fighting a polar bear in the arctic." --save_file output_video.mp4
```

For 32GB M4 Mac Mini, everything runs without swap, Video generation takes about 10GB and VAE uses about 12GB. Time taken: 12m14s.

For ```--frame_num 45 --sample_steps 50 --tile_size 128```, time taken: 1h23m.

Without quantized T5 model and mixed precision:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-1.3B --size "832*480" --frame_num 17 --sample_steps 25 --tile_size 128 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --device mps --prompt "Penguins fighting a polar bear in the arctic." --save_file output_video.mp4
```

For 32GB M4 Mac Mini, T5 model needs swap, but the video generation stage only uses about 16GB of RAM, VAE uses about 5GB. Time taken: 20m3s.

For ```--frame_num 25 --sample_steps 50 --tile_size 256```, time taken: 56m.
