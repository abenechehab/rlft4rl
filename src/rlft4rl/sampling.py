import logging
from typing import List, Optional, Dict
import re
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from rlft4rl.utils import trl_generate_completions
from rlft4rl.prompts import ACTION_START


def repeat_on_error(
    prompt: str,
    system_prompt: str,
    model_name: str,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    client: Optional[OpenAI] = None,
    temperature: float = 0.0,
    max_tokens: int = 75,
    n_action_dim: int = 6,
    logger: Optional[logging.Logger] = None,
    tol_repeat_gen: int = 5,
    logit_bias: Optional[Dict[str, int]] = None,
    good_tokens: List[str] = [],
):
    regex = r"^\s*([-+]?\d*\.\d+(?:,\s*[-+]?\d*\.\d+)*)\s*</action>"

    # messages = []
    # if system_prompt:
    #     messages.append({"content": system_prompt, "role": "system"})
    # messages.append({"content": prompt, "role": "user"})
    # messages.append({"content": "<action>", "role": "assistant"})

    messages = f"### Instructions: {system_prompt}\n ### User: {prompt}\n ### Controller: {ACTION_START}"

    count: int = 0
    n_try_gen: List[int] = []

    while True:
        if client is not None:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_tokens=max_tokens,
                logit_bias={id: logit_bias if logit_bias else 0 for id in good_tokens},
            )
            raw_response = response.choices[0].message.content
        else:
            generation_config = GenerationConfig(
                max_new_tokens=100,
                do_sample=True,
                temperature=0.9,
                num_return_sequences=1,
                pad_token_id=2,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            completions, _ = trl_generate_completions(
                model,
                [{"prompt": messages}],
                tokenizer,
                generation_config,
                device,
            )
            raw_response = completions[0]

        logger.debug(f"raw_response: {raw_response}")

        match = re.search(regex, raw_response, re.DOTALL)

        if match is None:
            n_try_gen.append(count + 1)
        else:
            numbers_str = match.group(1)
            numbers = [num.strip() for num in numbers_str.split(",")]
            if len(numbers) == n_action_dim:
                action = [float(x) for x in numbers]
                break
            else:
                n_try_gen.append(count + 1)
                logger.debug(f"Expected {n_action_dim} numbers, got {len(numbers)}")

        count += 1
        if count > tol_repeat_gen:
            raise Exception(
                f"Failed to get a valid response after {count} repetitions! "
                f"Expected action of dim {n_action_dim}, got: {raw_response}"
            )
    return action, n_try_gen


def complete_on_error(
    client: OpenAI,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 75,
    app_act_dim: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    tol_repeat_gen: int = 5,
    logit_bias: Optional[Dict[str, int]] = None,
    good_tokens: List[str] = [],
):
    if app_act_dim:
        app_act_dim = "<action> " + app_act_dim
    messages = []
    if system_prompt:
        messages.append({"content": system_prompt, "role": "system"})
    messages.append({"content": prompt, "role": "user"})
    messages.append({"content": app_act_dim, "role": "assistant"})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False,
        max_tokens=max_tokens,
        logit_bias={id: 30 if logit_bias else 0 for id in good_tokens},
    )

    raw_response = response.choices[0].message.content

    full_response = (
        app_act_dim + raw_response.split("<action>")[-1].split("</action>")[0]
    )
    full_response = full_response.split(",")

    if full_response[0] == "":
        full_response = full_response[1:]

    # Filter to keep only alphanumeric, -, and . characters
    filtered_response = [re.sub(r"[^0-9\-\.]", "", item) for item in full_response][:6]

    count: int = 0
    app_act_dim: str = ""
    n_try_gen: List[int] = []
    while True:
        # raw_response, full_response, filtered_response = generate_response(
        #     app_act_dim=app_act_dim
        # )
        logger.debug(f"raw_response: {raw_response}")
        if len(filtered_response) == 6:
            n_try_gen.append(count + 1)
            try:
                action = [float(x) for x in filtered_response]
                break
            except Exception as e:
                logger.debug(f"excpetion {e} has occured. repeating!")
        else:
            # if raw_response.count(",") == 5:
            # app_act_dim = ", ".join(filtered_response) + ", "
            # else:
            #     app_act_dim = ""
            pass
        count += 1
        if count > tol_repeat_gen:
            raise Exception(
                "Failed to get a valid response from the model! "
                f"Expected 6 action dimensions, got raw: {raw_response},"
                f"full_response: {full_response}, filtered: {filtered_response}"
            )
    return action
