import os
from dataclasses import dataclass
from datetime import datetime

from distutils.util import strtobool  # type: ignore

import torch

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

from rlft4rl.utils import setup_logger, set_seed_everywhere
from rlft4rl.reward.functions import (
    format_reward_func_constructor,
)  # ,format_reward_func, equation_reward_func


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


########################
# Setup logging
########################
logger, _, _ = setup_logger(
    logger_name="GRPO",
    log_dir="logs",
    env_id="post_training",
    exp_name="grpo",
    create_ts_writer=False,
)

########################


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        # truncation=True,
        # padding=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        #########################
    # Load pretrained model
    #########################

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

    # add special tokens
    obs_start = "<observation>"
    obs_end = "</observation>"
    action_start = "<action>"
    action_end = "</action>"

    num_added_toks = tokenizer.add_tokens(
        [obs_start, obs_end, action_start, action_end],
        special_tokens=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"*** num_added_toks {num_added_toks} ***")

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith(".json"):
        dataset = load_dataset(
            "json", data_files=script_args.dataset_id_or_path, split="train"
        )
    else:
        dataset = load_dataset(
            script_args.dataset_id_or_path, split=script_args.dataset_splits
        )

    if script_args.dataset_size > 0:
        dataset_size = min(script_args.dataset_size, len(dataset))
        dataset = dataset.select(range(dataset_size))

    #############################
    # Prepare and format dataset
    #############################

    def convert_to_array(s: str):
        s = s.split(",")
        s = [float(x) for x in s]
        return np.array(s)

    system_prompt = "You are the controller for a HalfCheetah robot in a physics "
    "simulation. The HalfCheetah is a 2-dimensional robot consisting of 9 body parts "
    f"and 8 connecting joints. You will receive the observation between {obs_start}"
    f"and {obs_end} tags, which contains the robot's state. Your task is to generate an"
    f" action between {action_start} and {action_end} tags. The action is a "
    "comma-separated list of 6 numbers, each representing the torque applied to the "
    "robot's joints (back thigh, back shin, back foot, front thigh, front shin, front "
    "foot)."  # Example output: {action_start}-0.39555,-0.66661,-0.36855,0.91655,
    #-0.81651,1.16655{action_end}."

    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["observation"]},
                {"role": "assistant", "content": "<action>"},
            ],
            "observation": convert_to_array(
                s=x["observation"].split(obs_start)[-1].split(obs_end)[0]
            ),
            "action": convert_to_array(
                x["action"].split(action_start)[-1].split(action_end)[0]
            ),
        }
    )

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #########################
    # Instantiate GRPO trainer
    #########################

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func_constructor(
                log_dir=training_args.output_dir, num_action_dim=6, add_action_tag=True
            ),
        ],  # [format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )

    ###############
    # Training loop
    ###############

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
        f'{training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train()
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["grpo", "rlft4rl", "abenechehab"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed_everywhere(training_args.seed)
    set_seed(training_args.seed)

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
