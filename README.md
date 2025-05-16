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
- Support for FLF2V model.
- Add disk offload for device with smaller RAM to run the 14B models.

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
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir ./Wan2.1-FLF2V-14B-720P

```

```bash
modelscope download Wan-AI/Wan2.1-T2V-1.3 --local_dir ./Wan2.1-T2V-1.3B
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
modelscope download Wan-AI/Wan2.1-I2V-14B-480P --local_dir ./Wan2.1-I2V-14B-480P
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./Wan2.1-I2V-14B-720P
modelscope download Wan-AI/Wan2.1-FLF2V-14B-720P --local_dir ./Wan2.1-FLF2V-14B-720P
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

### Text-to-Video with 1.3B model

To generate a video, use the following command:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-1.3B --size "832*480" --frame_num 17 --sample_steps 25  --ckpt_dir ./Wan2.1-T2V-1.3B --tile_size 256 --offload_model True --t5_quant --device mps --sample_shift 8 --sample_guide_scale 6 --prompt "Penguins fighting a polar bear in the arctic." --save_file output_video.mp4
```

```--t5_quant``` enables the quantized T5 model.

For 32GB M4 Mac Mini, everything runs without swap, Video generation takes about 10GB and VAE uses about 12GB. Time taken: 12m14s.

For ```--frame_num 45 --sample_steps 50 --tile_size 128```, time taken: 1h23m.

### Image-to-Video with 14B model

To generate a video, use the following command:

(For testing only, increase frame_num and sample_steps to get usable results.)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task i2v-14B --size "832*480" --frame_num 5 --sample_steps 2  --ckpt_dir ./Wan2.1-I2V-14B-480P --tile_size 256 --offload_model True --t5_quant --device mps --disk_offload --mps_ram 10GB --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --save_file output_video.mp4
```

For 32GB M4 Mac Mini with 10 Gbps external storage, Time taken: 13m19s.

### First-Last-Frame-to-Video with 14B model

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task flf2v-14B --size "1280*720" --frame_num 5 --sample_steps 2 --ckpt_dir ./Wan2.1-FLF2V-14B-720P  --tile_size 256 --offload_model True --t5_quant --device mps --disk_offload --mps_ram 10GB --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective." --save_file output_video.mp4
```

For 32GB M4 Mac Mini with 10 Gbps external storage, Time taken: 17m51s.

## About disk offloading

The disk offloading function uses Accelerate Big Model Inference mode. It allows device with smaller RAM to run the 14B models.

The compromise is time and disk. Each inference will writes about 60GB of cache, as accelerate seems to not support persistent cache yet.

Adding ```--disk_offload --mps_ram 10GB``` to the generation script to enable disk offloading and set the RAM limit.

Example:  

### T2V-14B

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-14B --size "832*480" --frame_num 5 --sample_steps 2 --tile_size 256 --ckpt_dir ./Wan2.1-T2V-14B --offload_model True --t5_quant --device mps --sample_shift 8 --sample_guide_scale 6 --prompt "Penguins fighting a polar bear in the arctic." --save_file output_video.mp4 --disk_offload --mps_ram 10GB
```

For 32GB M4 Mac Mini with 10 Gbps external storage, Time taken: 14m.
