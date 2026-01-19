# RLFT4RL

**Reinforcement Learning Fine-Tuning for Reinforcement Learning** - A research framework for fine-tuning Large Language Models (LLMs) to act as controllers/policies for RL environments.

## Overview

RLFT4RL investigates how LLMs can be adapted to solve reinforcement learning control tasks through post-training techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). The framework enables training LLMs on offline RL datasets (Minari/D4RL) to generate actions for continuous and discrete control environments.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rlft4rl

# Install dependencies
pip install -e .

# Optional: Install flash attention for faster training
pip install flash-attn==2.7.4.post1
```

### Requirements

- Python >= 3.9
- CUDA 12.1+ (for GPU support)
- Key dependencies: `transformers`, `trl`, `peft`, `torch`, `gymnasium`, `minari`

## Project Structure

```
rlft4rl/
├── src/rlft4rl/
│   ├── post_training/      # Training scripts (SFT, GRPO)
│   ├── policies/           # Policy implementations (LLM, MLP)
│   ├── reward/             # Reward functions and models
│   ├── data/               # Dataset handling (Minari/D4RL)
│   ├── prompts.py          # Prompt templates for environments
│   ├── sampling.py         # Action extraction utilities
│   └── utils.py            # General utilities
├── scripts/                # Evaluation scripts
├── config/                 # Training configuration files (YAML)
├── notebooks/              # Analysis notebooks
└── data/                   # Training datasets
```

## Evaluation Scripts

Located in `scripts/`, these scripts evaluate trained policies in RL environments.

### `eval_llm_policy.py`

Evaluates an LLM-based policy in a gymnasium environment.

```bash
python scripts/eval_llm_policy.py \
    --env_id "CartPole-v1" \
    --model "path/to/trained/model" \
    --episodes 10 \
    --temperature 0.3 \
    --n_examples 3
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--env_id` | `CartPole-v1` | Gymnasium environment ID |
| `--model` | `llama-3.3-70b-instruct` | LLM model name or path |
| `--episodes` | `10` | Number of evaluation episodes |
| `--temperature` | `0.3` | Sampling temperature |
| `--prediction_horizon` | `200` | Max tokens for action generation |
| `--n_examples` | `3` | Few-shot examples in system prompt |
| `--capture_video` | `False` | Record evaluation videos |
| `--use_vllm` | `False` | Use vLLM for inference |
| `--discretized` | `False` | Use discretized action representation |
| `--api_url` | `http://...` | API endpoint for remote LLM |

**Output:** Episode returns, lengths, and TensorBoard logs in `logs/`.

### `eval_random_policy.py`

Baseline evaluation with random, constant, or MLP policies.

```bash
# Random policy baseline
python scripts/eval_random_policy.py --env_id "HalfCheetah-v4" --mode random

# MLP policy (behavior cloning)
python scripts/eval_random_policy.py --env_id "HalfCheetah-v4" --mode mlp \
    --mlp_policy_path "models/mlp-policy/halfcheetah_medium/BC_test.pth"

# GRPO-trained MLP policy
python scripts/eval_random_policy.py --env_id "CartPole-v1" --mode grpo-mlp \
    --mlp_policy_path "models/grpo-mlp/cartpole/model.pth"
```

**Modes:**
- `random`: Random action sampling (baseline)
- `constant`: Zero/constant actions
- `mlp`: Trained MLP policy (behavior cloning)
- `grpo-mlp`: MLP policy trained with GRPO

## Post-Training Scripts

Located in `src/rlft4rl/post_training/`, these scripts fine-tune LLMs for control tasks.

### `sft.py` - Supervised Fine-Tuning

Trains an LLM to imitate expert demonstrations using standard language modeling loss.

```bash
accelerate launch src/rlft4rl/post_training/sft.py config/sft.yaml
```

**Configuration (`config/sft.yaml`):**
```yaml
# Model
model_name_or_path: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Dataset
dataset_id_or_path: "data/sft/mujoco_halfcheetah_medium-v0_1000.json"
eval_dataset_id_or_path: "data/sft/mujoco_halfcheetah_expert-v0_3.json"
dataset_size: 400000

# LoRA
use_peft: true
lora_r: 16
lora_alpha: 16
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_modules_to_save: ["lm_head", "embed_tokens"]

# Training
num_train_epochs: 40
per_device_train_batch_size: 32
learning_rate: 1e-3
```

**Features:**
- LoRA fine-tuning for parameter efficiency
- 4-bit quantization support (BitsAndBytes)
- Early stopping callback
- Evaluation logging callback
- Chat format setup for instruction-tuned models

### `grpo.py` - GRPO for Continuous Actions

Trains LLMs using Group Relative Policy Optimization with continuous action spaces (e.g., HalfCheetah).

```bash
accelerate launch src/rlft4rl/post_training/grpo.py config/grpo.yaml
```

**Key GRPO Parameters:**
```yaml
# GRPO specific
beta: 0.04                    # KL divergence weight
num_generations: 4            # Completions per prompt
max_completion_length: 10     # Max tokens for action
max_prompt_length: 512
scale_rewards: false
```

**Reward Functions:**
- `format_reward_func`: Validates `<action>...</action>` format
- `BC_reward_func`: Behavior cloning reward (similarity to expert)
- `log_rew_func`: Logs completions for analysis
- `reward_model_func`: Neural network reward model (optional)

### `grpo_d.py` - GRPO for Discrete Actions

Variant for discrete action environments (e.g., CartPole).

```bash
accelerate launch src/rlft4rl/post_training/grpo_d.py config/grpo.yaml \
    --dataset_id_or_path "data/sft/custom_cartpole_expert_ppo-v0_112.json"
```

**Differences from `grpo.py`:**
- Supports discrete reward functions (`format_reward_func_constructor_discrete`, `BC_reward_func_constructor_discrete`)
- Tokenized observation scaling for integer token representation
- Optional non-pretrained model initialization

### `sft_unsloth.py` - SFT with Unsloth Optimization

Optimized SFT using the Unsloth library for faster training.

### `sft_compl-only.py` - Completion-Only SFT

Trains only on the completion (action) portion, not the prompt.

### `merge_adapter_weights.py` - Merge LoRA Adapters

Merges LoRA adapter weights with the base model for inference.

```bash
python src/rlft4rl/post_training/merge_adapter_weights.py \
    --peft_model_id "models/grpo-lora8-16-d_qwen3_8b_cartpole" \
    --output_dir "models/merged-model" \
    --save_tokenizer True
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--peft_model_id` | Path to LoRA model |
| `--output_dir` | Output directory for merged model |
| `--save_tokenizer` | Save tokenizer alongside model |
| `--push_to_hub` | Upload to HuggingFace Hub |

## Supported Environments

**Continuous Action Spaces:**
- HalfCheetah (`HalfCheetah-v4`, `dm_control/cheetah-run-v0`)
- Other MuJoCo environments

**Discrete Action Spaces:**
- CartPole (`CartPole-v1`)
- Other Gymnasium classic control environments

## Workflow

```
┌─────────────────────────────────────────┐
│ Offline RL Dataset (Minari/D4RL)        │
│ - Expert/medium demonstrations          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Data Processing (d4rl.py)               │
│ - Convert to prompt format              │
│ - <observation>...</observation>        │
│ - <action>...</action>                  │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────────┐
│ SFT Training  │   │ GRPO Training     │
│ (sft.py)      │   │ (grpo.py/grpo_d.py)│
│               │   │                   │
│ Imitation     │   │ RL optimization   │
│ Learning      │   │ with rewards      │
└───────┬───────┘   └─────────┬─────────┘
        │                     │
        └─────────┬───────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Evaluation (eval_llm_policy.py)         │
│ - Test in environment                   │
│ - Compute returns                       │
└─────────────────────────────────────────┘
```

## Configuration

Training configurations are in `config/`:

- `grpo.yaml` - GRPO training settings
- `sft.yaml` - SFT training settings
- `deepspeed_zero3.yaml` - DeepSpeed optimization

## Logging

Training logs are written to:
- `logs/` - File and console logs
- TensorBoard logs in the output directory

View TensorBoard:
```bash
tensorboard --logdir models/your-model-dir
```

## License

MIT License

## Author

Abdelhakim Benechehab (abdelhakim.benechehab@gmail.com)
