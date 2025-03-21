import re
import numpy as np
from openai import OpenAI
from rlft4rl.prompts import ENV_SPECS


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

        self.init_prompt_template(env_name=env_name)

    def init_prompt_template(self, env_name):
        spec = ENV_SPECS[env_name]

        self.system_prompt = f"""
        You are the controller for a {env_name} robot in a physics simulation.

        Environment Description:
        {spec["description"]}

        Observation Space:
        The observation is a {spec["obs_dim"]}-dimensional vector representing:
        {", ".join([f"{i+1}. {desc}" for i, desc in enumerate(spec["obs_desc"])])}
        Gym representation of the obs space:
        {spec["obs_space"]}

        Action Space:
        The action is a {spec["action_dim"]}-dimensional vector representing:
        {spec["action_desc"]}
        Gym representation of the action space:
        {spec["action_space"]}

        Task:
        {spec["task"]}

        Reward Structure:
        {spec["reward"]}

        Simulation Frequency:
        {spec["frequency"]}

        Instructions:
        1. Based on the current observation provided between <observation> and
            </observation>, determine the optimal action values.
        2. Your response must ONLY contain the numeric action values separated by
            commas.
        3. Do not include any explanations or additional text in your response.
        4. Format your response exactly as: "action_dimension_1, action_dimension_2,
            action_dimension_3, action_dimension_4, action_dimension_5,
            action_dimension_6 </action>"
        5. Each action dimension value should respect the range in the action space.

        Example:

        <observation> {spec["example_observation"]} </observation>

        <action> {spec["example_action"]} </action>

        """

    def obs_to_prompt(self, obs):
        prompt = f"""
        <observation> {obs} </observation>

        <action>
        """
        return prompt

    def act(self, obs):
        prompt = self.obs_to_prompt(obs)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            stream=True,
            max_tokens=2 * self.prediction_horizon,
            logit_bias={id: 30 if self.logit_bias else 0 for id in self.good_tokens},
        )

        raw_response = ""
        for chunk in stream:
            try:
                raw_response += chunk.choices[0].delta.content
            except TypeError:
                pass
        full_response = raw_response.split(" </action>")[0]
        full_response = full_response.split(",")

        if full_response[0] == "":
            full_response = full_response[1:]

        # Filter to keep only alphanumeric, -, and . characters
        filtered_response = [re.sub(r"[^0-9\-\.]", "", item) for item in full_response]

        assert (
            len(filtered_response) == 6
        ), f"Expected 6 action dimensions, got {filtered_response}"

        action = [float(x) for x in filtered_response]
        return np.array(action)
