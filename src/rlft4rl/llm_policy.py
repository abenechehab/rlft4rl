import logging
from typing import List, Optional, Dict
import re
import numpy as np
from openai import OpenAI
from rlft4rl.prompts import ENV_DESC, INSTRUCTIONS


class LLMPolicy:
    def __init__(
        self,
        api_url,
        api_key,
        model,
        temperature,
        prediction_horizon,
        logit_bias,
        good_tokens,
        env_name,
        tol_repeat_gen: int = 10,
        examples: Optional[Dict] = None,
        system_prompt: bool = True,
    ):
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )
        self.model = model
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

    def init_prompt_template(self, env_name: str, examples: Optional[Dict] = None):
        desc = ENV_DESC[env_name]

        self.system_prompt = f"""
        You are the controller for a {env_name} robot in a physics simulation.

        Environment Description:
        """
        self.system_prompt += desc
        self.system_prompt += INSTRUCTIONS

        if examples:
            for i, (_, val) in enumerate(examples.items()):
                self.system_prompt += f"""Example {i+1}:
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

        def generate_response(app_act_dim: str = ""):
            if app_act_dim:
                app_act_dim = "<action> " + app_act_dim
            messages = []
            if self.bool_system_prompt:
                messages.append({"content": self.system_prompt, "role": "system"})
            messages.append({"content": prompt, "role": "user"})
            # messages.append({"content": app_act_dim, "role": "assistant"})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # temperature=self.temperature,
                stream=False,
                max_tokens=75,
                # logit_bias={
                #     id: 30 if self.logit_bias else 0 for id in self.good_tokens
                # },
            )

            # raw_response = ""
            # for chunk in stream:
            #     try:
            #         raw_response += chunk.choices[0].delta.content
            #     except TypeError:
            #         pass

            raw_response = response.choices[0].message.content

            # breakpoint()

            full_response = (
                app_act_dim + raw_response.split("<action>")[-1].split("</action>")[0]
            )
            full_response = full_response.split(",")

            if full_response[0] == "":
                full_response = full_response[1:]

            # Filter to keep only alphanumeric, -, and . characters
            filtered_response = [
                re.sub(r"[^0-9\-\.]", "", item) for item in full_response
            ][:6]
            return raw_response, full_response, filtered_response

        count = 0
        app_act_dim = ""
        while True:
            raw_response, full_response, filtered_response = generate_response(
                app_act_dim=app_act_dim
            )
            logger.debug(f"raw_response: {raw_response}")
            if len(filtered_response) == 6:
                self.n_try_gen.append(count + 1)
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
            if count > self.tol_repeat_gen:
                raise Exception(
                    "Failed to get a valid response from the model! "
                    f"Expected 6 action dimensions, got raw: {raw_response},"
                    f"full_response: {full_response}, filtered: {filtered_response}"
                )
        return np.array(action)

    def log(self, logger: logging.Logger):
        logger.info(f"[N tries gen] mean{np.mean(self.n_try_gen)}")
        logger.info(f"[N tries gen] std{np.std(self.n_try_gen)}")
        logger.info(f"[N tries gen] min{np.min(self.n_try_gen)}")
        logger.info(f"[N tries gen] max{np.max(self.n_try_gen)}")
