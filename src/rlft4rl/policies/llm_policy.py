import logging
from typing import List, Optional, Dict
import numpy as np
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from rlft4rl.prompts import ENV_DESC, INSTRUCTIONS, SHORT_SYSTEM_PROMPT_HALFCHEETAH
from rlft4rl.sampling import repeat_on_error


class LLMPolicy:
    def __init__(
        self,
        api_url,
        api_key,
        model_name,
        temperature,
        prediction_horizon,
        logit_bias,
        good_tokens,
        env_name,
        tol_repeat_gen: int = 10,
        examples: Optional[Dict] = None,
        system_prompt: bool = True,
        use_vllm: bool = False,
        device: str = "cuda:0",
    ):
        if use_vllm:
            self.client = OpenAI(
                base_url=api_url,
                api_key=api_key,
            )
            self.model = None
        else:
            self.client = None
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        self.model_name = model_name
        self.temperature = temperature
        self.prediction_horizon = prediction_horizon
        self.logit_bias = logit_bias
        self.good_tokens = good_tokens

        self.init_prompt_template(
            env_name=env_name,
            examples=examples,
        )

        self.tol_repeat_gen = tol_repeat_gen
        self.n_try_gen: List[int] = []
        self.bool_system_prompt = system_prompt

    def init_prompt_template(
        self, env_name: str, examples: Optional[Dict] = None, use_short: bool = True
    ):
        if use_short:
            self.system_prompt = SHORT_SYSTEM_PROMPT_HALFCHEETAH
        else:
            desc = ENV_DESC[env_name]

            self.system_prompt = f"""
            You are the controller for a {env_name} robot in a physics simulation.

            Environment Description:
            """
            self.system_prompt += desc
            self.system_prompt += INSTRUCTIONS

            if examples:
                for i, (_, val) in enumerate(examples.items()):
                    self.system_prompt += f"""Example {i + 1}:
                    <observation> {val["obs"]} </observation>
                    <action> {val["action"]} </action>

                    """

    def obs_to_prompt(self, obs):
        # Convert obs list to a comma-separated string without brackets
        obs_string = ", ".join([f"{val:.5f}" for val in obs])
        prompt = f"""<observation> {obs_string} </observation>"""
        return prompt

    def act(self, obs, logger: logging.Logger):
        prompt = self.obs_to_prompt(obs)

        # compute max_tokens based on example
        ex_action = (
            "<action>-0.39555,-0.66661,-0.36855,0.91655,-0.81651,1.16655</action>"
        )
        max_tokens = len(self.tokenizer.tokenize(ex_action))

        logger.debug(f"max_tokens: {max_tokens}")

        tokens = set()
        for i in range(1000):
            toks = self.tokenizer.tokenize(str(i))
            tokens.update(toks)
        num_tk = self.tokenizer.convert_tokens_to_ids(list(tokens))  # number tokens
        tokens = set()
        tokens.update(self.tokenizer.tokenize("."))
        tokens.update(self.tokenizer.tokenize("-"))
        symbol_tk = self.tokenizer.convert_tokens_to_ids(list(tokens))  # symbol tokens
        tokens = set()
        tokens.update(self.tokenizer.tokenize("<action></action>"))
        tag_tk = self.tokenizer.convert_tokens_to_ids(list(tokens))  # tag tokens
        tokens = set()
        tokens.update(self.tokenizer.tokenize(","))
        sep_tk = self.tokenizer.convert_tokens_to_ids(list(tokens))  # separator tokens

        action, n_try_gen = repeat_on_error(
            model_name=self.model_name,
            client=self.client,
            prompt=prompt,
            system_prompt=self.system_prompt if self.bool_system_prompt else None,
            model=self.model,
            tokenizer=self.tokenizer,
            n_action_dim=6,  # TODO: set this automatically
            temperature=self.temperature,
            max_tokens=max_tokens,
            logger=logger,
            tol_repeat_gen=self.tol_repeat_gen,
            logit_bias=self.logit_bias,
            good_tokens=num_tk + symbol_tk + tag_tk + sep_tk,
        )
        self.n_try_gen = n_try_gen
        return np.array(action)

    def log(self, logger: logging.Logger):
        logger.info(f"[N tries gen] mean{np.mean(self.n_try_gen)}")
        logger.info(f"[N tries gen] std{np.std(self.n_try_gen)}")
        logger.info(f"[N tries gen] min{np.min(self.n_try_gen)}")
        logger.info(f"[N tries gen] max{np.max(self.n_try_gen)}")
