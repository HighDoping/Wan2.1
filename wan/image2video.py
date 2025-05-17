# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def clear_cache():
    gc.collect()
    if torch.backends.mps.is_built():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        t5_quant=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        # Check if device_id is a torch.device instance
        if isinstance(device_id, torch.device):
            self.device = device_id
        elif device_id == "mps" or (isinstance(device_id, int) and
                                    device_id == -1):
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.t5_fsdp = t5_fsdp
        self.t5_quant = t5_quant

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        input_prompt,
        img,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        disk_offload=False,
        mps_ram="10GB",
        VAE_tile_size=None,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = (((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w //
                       (self.patch_size[1] * self.patch_size[2]))
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        logging.info("Loading text encoder model.")

        if self.t5_quant:
            # custom t5 quantization
            from .modules.t5_gguf import run_llama_embedding

            checkpoint_path = os.path.join(self.checkpoint_dir,
                                           self.config.t5_quant_checkpoint)
            context = [
                torch.from_numpy(
                    run_llama_embedding(checkpoint_path,
                                        input_prompt)).to(self.device)
            ]
            context_null = [
                torch.from_numpy(
                    run_llama_embedding(checkpoint_path,
                                        n_prompt)).to(self.device)
            ]
        else:
            # original t5
            from .modules.t5 import T5EncoderModel

            self.text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=os.path.join(self.checkpoint_dir,
                                             self.config.t5_checkpoint),
                tokenizer_path=os.path.join(self.checkpoint_dir,
                                            self.config.t5_tokenizer),
                shard_fn=None,
            )

            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
            else:
                context = self.text_encoder([input_prompt], torch.device("cpu"))
                context_null = self.text_encoder([n_prompt],
                                                 torch.device("cpu"))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            if offload_model:
                del self.text_encoder
                logging.info("Remove text encoder model.")
                clear_cache()

        logging.info("Loading CLIP model.")
        self.clip = CLIPModel(
            dtype=self.config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(self.checkpoint_dir,
                                         self.config.clip_checkpoint),
            tokenizer_path=os.path.join(self.checkpoint_dir,
                                        self.config.clip_tokenizer),
        )
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            del self.clip
            logging.info("Remove CLIP model.")
            clear_cache()

        logging.info("Loading VAE model.")
        self.vae = WanVAE(
            vae_pth=os.path.join(self.checkpoint_dir,
                                 self.config.vae_checkpoint),
            device=self.device,
        )
        logging.info("Encoding image.")
        y = self.vae.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(
                            img[None], size=(h, w), mode="bicubic").transpose(
                                0, 1),
                        torch.zeros(3, F - 1, h, w),
                    ],
                    dim=1,
                ).to(self.device)
            ],
            VAE_tile_size,
        )[0]
        if offload_model:
            del self.vae
            logging.info("Remove VAE model.")
            clear_cache()
        y = torch.concat([msk, y])

        logging.info("Loading WanModel")

        if disk_offload:
            logging.info("Use disk offload.")
            self.model = WanModel.from_pretrained(
                self.checkpoint_dir,
                device_map="auto",
                max_memory={
                    "mps": mps_ram,
                    "cpu": "0.5GB"
                },
                offload_folder="disk_offload",
                offload_state_dict=True,
            )
        else:
            self.model = WanModel.from_pretrained(self.checkpoint_dir)
            self.model.to(self.device)
        self.model.eval().requires_grad_(False)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # evaluation mode
        with (
                amp.autocast(
                    device_type=str(self.device), dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                "context": [context[0]],
                "clip_fea": clip_context,
                "seq_len": max_seq_len,
                "y": [y],
            }

            arg_null = {
                "context": context_null,
                "clip_fea": clip_context,
                "seq_len": max_seq_len,
                "y": [y],
            }

            # if offload_model:
            #     clear_cache()

            logging.info("Start generation loop.")
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent]
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                if offload_model:
                    clear_cache()
                del latent_model_input, timestep
            logging.info("End generation loop.")

            if offload_model:
                del self.model
                logging.info("Remove WanModel.")
                clear_cache()

            if self.rank == 0:
                logging.info("Loading VAE model.")
                self.vae = WanVAE(
                    vae_pth=os.path.join(self.checkpoint_dir,
                                         self.config.vae_checkpoint),
                    device=self.device,
                )
                logging.info("Decoding video.")
                videos = self.vae.decode(x0, VAE_tile_size)

        del noise, latent
        del sample_scheduler
        if offload_model:
            clear_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
