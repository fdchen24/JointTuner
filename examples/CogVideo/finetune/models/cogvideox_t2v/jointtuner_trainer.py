import copy
import os
import gc
from typing import Any, Dict, List, Tuple
import hashlib
import json
import logging
import math
from tqdm import tqdm
from datetime import timedelta
from pathlib import Path

import diffusers
import torch
import wandb
import transformers
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import T2VDatasetWithResize
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)

from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.export_utils import export_to_video

from ..utils import register

from .lora_handler import LoraHandler

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}

PROMPT_FILENAME_LEN=40

class JointTunerTrainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST = ["text_encoder", "vae"]

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        print('state dtype', self.state.weight_dtype)

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def save_pipe(
            self,
            global_step,
            accelerator
    ):
        save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
        transformer = copy.deepcopy(accelerator.unwrap_model(self.components.transformer.cpu(), keep_fp32_wrapper=False))

        self.lora_manager.save_lora_weights(model=copy.deepcopy(transformer.cpu()), save_path=save_path, step=global_step,flag=None)

        logger.info(f"Saved model at {save_path} on step {global_step}")

        del transformer
        torch.cuda.empty_cache()
        gc.collect()

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config


    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        self.dataset = T2VDatasetWithResize(
            **(self.args.model_dump()),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )
       

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            ...
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                component.requires_grad_(False)


    def prepare_optimizer(self, initial_global_step) -> None:        
        logger.info("Initializing optimizer and lr scheduler")

        weight_dtype = self.state.weight_dtype
        
        self.lora_manager = LoraHandler(replace_modules=["CogVideoXBlock"])
        
        lora_params, negation = self.lora_manager.add_lora_to_model(self.components.transformer, self.lora_manager.replace_modules, 0.1,
            str(self.args.resume_from_checkpoint) + '/lora/' if self.args.resume_from_checkpoint else "", r=self.args.rank, scale=self.args.lora_alpha, 
            weight_dtype=weight_dtype)
        
        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)
        
        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()
        
        cast_training_params([self.components.transformer], dtype=torch.float32)
        
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, self.components.transformer.parameters())
        )
        
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }


        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
                self.accelerator.state.deepspeed_plugin is not None
                and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
                self.accelerator.state.deepspeed_plugin is not None
                and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
            )
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True
            
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_videos = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.accelerator.init_trackers(tracker_name, self.args.model_dump(), {"wandb": {"mode":"offline"}})


    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
                self.args.batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
        )

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )

        self.prepare_optimizer(initial_global_step)
        
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    loss = self.compute_loss_fusion(batch)

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % self.args.checkpointing_steps == 0 and global_step > 0:
                        self.save_pipe(
                            global_step,
                            accelerator
                        )

                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

        accelerator.wait_for_everyone()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.save_pipe(
                global_step,
                accelerator,
            )
        if self.args.do_validation and global_step < self.args.train_steps:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        logging.info('Congratulations! The training is completed!')
        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=["transformer"]
            )
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]

            if image is not None:
                image = preprocess_image_with_resize(
                    image, self.state.train_height, self.state.train_width
                )
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step(
                {"prompt": prompt, "image": image, "video": video}, pipe
            )

            if (
                    self.state.using_deepspeed
                    and self.accelerator.deepspeed_plugin.zero_stage == 3
                    and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:PROMPT_FILENAME_LEN]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update(
                    {f"artifact_{i}": {"type": artifact_type, "value": artifact_value}}
                )
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST
            )
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,    # 226
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "datatype": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            datatype = sample["datatype"]  # 'image' or 'video'
            

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["datatype"].append(datatype)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])

        return ret
    
    @override
    def compute_loss_fusion(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        # for 49,480,720 -> [1, 13, 16, 60, 90]
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        predicted_noise = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        datatypes = batch["datatype"] 
        video_indices = [i for i, t in enumerate(datatypes) if t == 'video']

        if video_indices:
            offset_noise = torch.randn(len(video_indices), 1, 1, height, width, device=latent.device, dtype=predicted_noise.dtype)
            predicted_noise[video_indices]  = predicted_noise[video_indices] + offset_noise

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_added_noise, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss
    
    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin
    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(
                        self.components, name, component.to(self.accelerator.device, dtype=dtype)
                    )

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

register("cogvideox-t2v", "jointtuner", JointTunerTrainer)
