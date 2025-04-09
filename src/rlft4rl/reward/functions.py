import os
from pathlib import Path
import random
import re


def debug_fn(completions, observation, action, **kwargs):
    rewards = []
    breakpoint()

    for _, _, _ in zip(completions, observation, action):
        rewards.append(0.0)
    return rewards


def format_reward_func_constructor(log_dir, num_action_dim, add_action_tag=False):
    def format_reward_func(prompts, completions, observation, action, **kwargs):
        """
        Format: <action>...</action>
        Args:
            completions (list[str]): Generated outputs

        Returns:
            list[float]: Reward scores
        """
        # Check if the format is correct
        regex = r"^<action>([-+]?\d*\.\d+(?:,\s*[-+]?\d*\.\d+)*)</action>$"
        rewards = []

        for completion in completions:
            response = completion[0]["content"]

            if add_action_tag:
                response = "<action>" + response

            breakpoint()

            # try:
            if random.random() < 0.1:  # 1% chance to write samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = Path(log_dir) / "completion_samples.txt"
                with open(log_file, "a") as f:
                    f.write("\n\n==============\n")
                    f.write(response)

            match = re.search(regex, response, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None:
                rewards.append(0.0)
            else:
                # Extract the numbers inside the <action> tag
                numbers_str = match.group(1)
                # Split the numbers by commas, remove any extra spaces, and count the
                # number of values
                numbers = [num.strip() for num in numbers_str.split(",")]

                # Check if the number of values matches the expected num_action_dim
                if len(numbers) == num_action_dim:
                    rewards.append(2.0)
                else:
                    rewards.append(1.0)
            # except Exception:
            #     rewards.append(0.0)
        return rewards

    return format_reward_func


def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for
            # the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses,
            # and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
                if (
                    random.random() < 0.10
                ):  # 10% chance to write fully successful samples into a file
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write("\n\n==============\n")
                        f.write(completion)
            else:
                rewards.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
    return rewards
