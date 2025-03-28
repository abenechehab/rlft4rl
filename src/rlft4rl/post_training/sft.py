from dataclasses import dataclass
from distutils.util import strtobool  # type: ignore
import os
import re
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    SFTConfig,
    get_peft_config,
    # setup_chat_format,
)
from datasets import load_dataset

from rlft4rl.utils import setup_logger, set_seed_everywhere

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_size: int
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None


########################
# Setup logging
########################
logger, _, _ = setup_logger(
    logger_name="SFT",
    log_dir="logs",
    env_id="post_training",
    exp_name="sft",
    create_ts_writer=False,
)

########################
# Helper functions
########################


def setup_model_for_spectrum(model, spectrum_config_path):
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    # get the unfrozen parameters from the yaml file
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(
            re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters
        ):
            param.requires_grad = True

    # COMMENT IN: for sanity check print the trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}")

    return model


#######################################################################################


def train_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig
):
    """Main training function."""
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith(".json"):
        train_dataset = load_dataset(
            "json", data_files=script_args.dataset_id_or_path, split="train"
        )
    else:
        train_dataset = load_dataset(
            script_args.dataset_id_or_path, split=script_args.dataset_splits
        )

    train_dataset = train_dataset.select(range(script_args.dataset_size))

    logger.info(
        f"Loaded dataset with {len(train_dataset)} samples and the following "
        f"features: {train_dataset.features}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if we use peft we need to make sure we use a chat template that is not using
    # special tokens as by default embedding layers will not be trainable
    tokenizer.padding_side = "right"  # to prevent warnings

    #######################
    # Load pretrained model
    #######################

    # define model kwargs
    model_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(
            torch, model_args.torch_dtype
        ),  # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True,  # Whether
        low_cpu_mem_usage=True
        if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
        else None,  # Reduces memory usage on CPU for loading the model
    )

    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    # load the model with our kwargs
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    training_args.distributed_state.wait_for_everyone()  # wait for all procs to load

    # for peft we need to make sure we use a chat template that is not using special
    # tokens as by default embedding layers will not be trainable

    # set chat template to OAI chatML, remove if you start from a fine-tuned model
    # model, tokenizer = setup_chat_format(model, tokenizer)

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    logger.info(f"*** Starting training for {training_args.num_train_epochs} epochs***")
    train_result = trainer.train()
    # log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all procs to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["sft", "rlft4rl", "abenechehab"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed_everywhere(training_args.seed)
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
