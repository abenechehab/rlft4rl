from dataclasses import dataclass
import random
from distutils.util import strtobool  # type: ignore
import os

# import re
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    # EarlyStoppingCallback,
    TrainerCallback,
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
    eval_dataset_id_or_path: str = None
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

    #########################
    # Load datasets
    #########################
    if script_args.dataset_id_or_path.endswith(".json"):
        train_dataset = load_dataset(
            "json", data_files=script_args.dataset_id_or_path, split="train"
        )
    else:
        train_dataset = load_dataset(
            script_args.dataset_id_or_path, split=script_args.dataset_splits
        )

    if script_args.dataset_size > 0:
        dataset_size = min(script_args.dataset_size, len(train_dataset))
        train_dataset = train_dataset.select(range(dataset_size))

    if script_args.eval_dataset_id_or_path:
        if script_args.eval_dataset_id_or_path.endswith(".json"):
            eval_dataset = load_dataset(
                "json", data_files=script_args.eval_dataset_id_or_path, split="train"
            )
        else:
            eval_dataset = load_dataset(
                script_args.eval_dataset_id_or_path, split=script_args.dataset_splits
            )
    else:
        eval_dataset = None

    logger.info(
        f"Loaded dataset with {len(train_dataset)} samples and the following "
        f"features: {train_dataset.features}"
    )

    #########################
    # Load tokenizer
    #########################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if we use peft we need to make sure we use a chat template that is not using
    # special tokens as by default embedding layers will not be trainable
    tokenizer.padding_side = "right"  # to prevent warnings

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
    num_added_toks = tokenizer.add_tokens(
        ["<observation>", "</observation>", "<action>", "</action>"],
        special_tokens=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"*** num_added_toks {num_added_toks} ***")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    class LoggingCallback(TrainerCallback):
        def __init__(
            self, eval_dataset, tokenizer, log_file="eval_log.txt", num_samples=1
        ):
            self.eval_dataset = eval_dataset
            self.tokenizer = tokenizer
            self.log_file = log_file
            self.num_samples = num_samples
            self.log_file_path = os.path.join(training_args.output_dir, log_file)
            # Clear the log file at the beginning of training
            with open(self.log_file_path, "w") as f:
                f.write("Evaluation Log:\n")

        def on_evaluate(self, args, state, control, model, **kwargs):
            # Select random samples from the eval dataset
            indices = random.sample(range(len(self.eval_dataset)), self.num_samples)
            samples = [self.eval_dataset[i] for i in indices]

            with open(self.log_file_path, "a") as f:
                for sample in samples:
                    # Extract the query from the sample
                    query = "".join(
                        [
                            msg["content"]
                            for msg in sample["messages"]
                            if msg["role"] in ["system", "user"]
                        ]
                    )  # Extract user message

                    # Generate response from the model
                    input_ids = self.tokenizer.encode(query, return_tensors="pt").to(
                        model.device
                    )
                    with torch.no_grad():
                        output = model.generate(
                            input_ids,
                            max_length=128,
                            num_return_sequences=1,
                            early_stopping=True,
                            os_token_id=self.tokenizer.eos_token_id,
                        )  # Adjust max_length as needed
                    response = self.tokenizer.decode(
                        output[0], skip_special_tokens=True
                    )

                    # Log the query and response
                    f.write(f"Query: {query}\nResponse: {response}\n\n")
            logger.info(
                f"Wrote {self.num_samples} query/response pairs to {self.log_file_path}"
            )

    # Instantiate the logging callback
    # logging_callback = LoggingCallback(eval_dataset, tokenizer)
    # trainer.add_callback(logging_callback)

    #########################
    # Training loop
    #########################

    # Add EarlyStopping callback
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=5,
    #     # Number of evaluations with no improvement after which training will be
    # stopped
    #     early_stopping_threshold=0.01,
    #     # improvement over best objective is less than that amount, the training could
    #     # be stopped.
    # )
    # trainer.add_callback(early_stopping_callback)

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
