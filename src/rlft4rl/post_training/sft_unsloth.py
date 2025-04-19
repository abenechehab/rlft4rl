from unsloth import FastLanguageModel

from dataclasses import dataclass
import random
import os

# import re
from typing import Optional

import torch
from transformers import (
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    SFTConfig,
    setup_chat_format,
    DataCollatorForCompletionOnlyLM,
)
from datasets import load_dataset

from rlft4rl.utils import setup_logger, set_seed_everywhere
from rlft4rl.prompts import (
    OBS_START,
    OBS_END,
    ACTION_START,
    ACTION_END,
    SHORT_SYSTEM_PROMPT_HALFCHEETAH,
)

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
    setup_chat_format: bool = False
    early_stopping_callback: bool = False
    # Whether to use the early stopping callback
    logging_callback: bool = False
    # Whether to use the logging callback


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
###########################


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

    ##################################################
    # Load model and tokenizer with unsloth
    ##################################################
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        # dtype=model_args.torch_dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=model_args.load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = model.to("cuda")

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # if we use peft we need to make sure we use a chat template that is not using
    # special tokens as by default embedding layers will not be trainable
    tokenizer.padding_side = "right"  # to prevent warnings

    #########################
    # Load pretrained model
    #########################

    # define model kwargs
    # model_kwargs = dict(
    #     attn_implementation=model_args.attn_implementation,
    #     # What attention implementation to use, defaults to flash_attention_2
    #     torch_dtype=model_args.torch_dtype
    #     if model_args.torch_dtype in ["auto", None]
    #     else getattr(
    #         torch, model_args.torch_dtype
    #     ),  # What torch dtype to use, defaults to auto
    #     use_cache=False if training_args.gradient_checkpointing else True,  # Whether
    #     low_cpu_mem_usage=True
    #     if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
    #     else None,  # Reduces memory usage on CPU for loading the model
    # )

    # Check which training method to use and if 4-bit quantization is needed
    # if model_args.load_in_4bit:
    #     model_kwargs["quantization_config"] = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    #         bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    #     )

    if model_args.use_peft:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,  # Dropout = 0 is currently optimized
            bias="none",  # Bias = "none" is currently optimized
            use_gradient_checkpointing=False,
            random_state=training_args.seed,
        )

    training_args.distributed_state.wait_for_everyone()  # wait for all procs to load

    # for peft we need to make sure we use a chat template that is not using special
    # tokens as by default embedding layers will not be trainable

    # set chat template to OAI chatML, remove if you start from a fine-tuned model
    if script_args.setup_chat_format:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # add special tokens
    num_added_toks = tokenizer.add_tokens(
        [OBS_START, OBS_END, ACTION_START, ACTION_END],
        special_tokens=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Set action_end as the EOS token
    # tokenizer.eos_token = ACTION_END
    # tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(ACTION_END)

    logger.info(f"*** num_added_toks {num_added_toks} ***")
    logger.info(f"*** model {model} ***")

    ##################################
    # Completion only data collator
    ##################################
    # Format datasets
    train_dataset = train_dataset.map(
        lambda x: {
            "text": f"### User: {SHORT_SYSTEM_PROMPT_HALFCHEETAH + x['observation']}\n ### Controller: {x['action']}"
        }
    )

    eval_dataset = eval_dataset.map(
        lambda x: {
            "text": f"### User: {SHORT_SYSTEM_PROMPT_HALFCHEETAH + x['observation']}\n ### Controller: {x['action']}"
        }
    )

    response_template = " ### Controller:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    ###################################

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        # processing_class=tokenizer,
        # formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

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
                            max_new_tokens=100,
                            # num_return_sequences=1,
                            # early_stopping=True,
                            eos_token_id=self.tokenizer.eos_token_id,
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
    logging_callback = LoggingCallback(eval_dataset, tokenizer)
    if script_args.logging_callback:
        trainer.add_callback(logging_callback)

    #########################
    # Training loop
    #########################

    # Add EarlyStopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        # Number of evaluations with no improvement after which training will be stopped
        early_stopping_threshold=0.01,
        # improvement over best objective is less than that amount, the training could
        # be stopped.
    )
    if script_args.early_stopping_callback:
        trainer.add_callback(early_stopping_callback)

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
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all procs to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            {"tags": ["sft", "unsloth", "rlft4rl", "abenechehab"]}
        )
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
