from jinja2 import Template


class PromptTemplate:
    """A class to represent a template for generating prompts with variables
    Attributes:
        template (str): The template string with variables
        input_variables (list): A list of the variable names in the template
    """

    def __init__(self, template, input_variables):
        self.template = Template(template)
        self.input_variables = input_variables

    def format(self, **kwargs):
        return self.template.render(**kwargs)


OBS_START = "<observation>"
OBS_END = "</observation>"
ACTION_START = "<action>"
ACTION_END = "</action>"

SHORT_SYSTEM_PROMPT_HALFCHEETAH = f"""You are the controller for a HalfCheetah robot in a physics simulation. The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 connecting joints. You will receive the observation between {OBS_START} and {OBS_END} tags, which contains the robot's state. Your task is to generate an action between {ACTION_START} and {ACTION_END} tags. The action is a comma-separated list of 6 numbers, each representing the torque applied to the robot's joints (back thigh, back shin, back foot, front thigh, front shin, front foot)."""  # Example output: {ACTION_START}-0.39555,-0.66661,-0.36855,0.91655,-0.81651,1.16655{ACTION_END}."""
SHORT_SYSTEM_PROMPT_HALFCHEETAH_TOK = f"""### Instructions: You are the controller for a HalfCheetah robot in a physics simulation. The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them. You will receive the observation between {OBS_START} and {OBS_END} tags, which contains 17 encoded values representing the robot's state. Each value is encoded as an integer from 0-999 where 0 corresponds to the minimal physically possible value and 999 the maximal one. Your task is to generate an action between {ACTION_START} and {ACTION_END} tags. The action should be exactly 6 integers from 0-999, separated by commas, each representing the encoded torque applied to the robot's joints in this order: back_thigh, back_shin, back_foot, front_thigh, front_shin, front_foot. 0 corresponds to the minimal physically possible action value and 999 the maximal one."""  # Example format: {ACTION_START}456,123,789,234,567,890{ACTION_END}"""
SHORT_SYSTEM_PROMPT_CARTPOLE = f"""### Instructions: You are the controller for a CartPole system in a physics simulation. The CartPole system consists of a cart that moves along a horizontal track and a pole standing upright on the cart. You will receive the observation between {OBS_START} and {OBS_END} tags, which contains 4 encoded values representing the system's state. The observation values correspond to the following physical quantities:
1. Cart Position: The horizontal position of the cart on the track.
2. Cart Velocity: The rate of change of the cart's position along the track.
3. Pole Angle: The angular deviation of the pole from the vertical position.
4. Pole Angular Velocity: The rate of change of the pole's angular deviation.

Your task is to generate an action between {ACTION_START} and {ACTION_END} tags. The action should be exactly 1 integer from the valid action list [0, 1]. The integer represents the direction to push the cart: 0 corresponds to pushing the cart to the left, and 1 corresponds to pushing the cart to the right. Example output: "{ACTION_START}0{ACTION_END}". Dont add any other text just output the action (0 or 1), the closing tag action {ACTION_END}, and then stop generating."""


INSTRUCTIONS = """Instructions:
1. Based on the current observation provided between <observation> and
    </observation>, determine the optimal action values.
2. Write your answer between <action> and </action> tags.
3. Your response must ONLY contain the numeric action values separated by
    commas.
4. Do not include any explanations or additional text in your response.
5. Format your response exactly as: "<action> action[0], action[1],
    action[2], action[3], action[4], action[5] </action>"
6. Each action value should respect the range in the action space.
"""

ENV_DESC = {
    "HalfCheetah-v4": """
The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints
connecting them (including two paws).
The cheetah’s head is fixed to its torso.
The HalfCheetah is capable of performing various locomotion tasks by applying torque to
6 joints over the front and back thighs (which connect to the torso), the shins
(which connect to the thighs), and the feet (which connect to the shins).

## Technical Specifications

### Observation Space: Box(-inf, inf, (17,), float32)
The environment observation is represented as a 1-dimensional NumPy array of shape
`(17,)`, containing comprehensive information about the HalfCheetah's current
configuration:

#### 1. Position and Angle
- `observation[0:2]`: Position (observation[0]) and angle (observation[1]) of head fixed
 to its torso
  - `observation[0]`: z-coordinate of the head in meters (0 when spawned).
  - `observation[1]`: angle of the head in radians (0 when spawned).

#### 2. Joint Angles
- `observation[2:8]`: Angles of joint in back leg (observation[2:5]) and front
leg(observation[5:8])
  - `observation[2]`: angle of the back thigh in radians.
  - `observation[3]`: angle of the back shin in radians.
  - `observation[4]`: angle of the back foot in radians.
  - `observation[5]`: angle of the front thigh in radians.
  - `observation[6]`: angle of the front shin in radians.
  - `observation[7]`: angle of the front foot in radians.

### Action Space: Box(-1.0, 1.0, (6,), float32)
The action space consists of a 1-dimensional NumPy array of shape `(6,)`,
controlling the torques applied to each of the HalfCheetah's actuated joints.

**Range**: All actions are bounded between [-1, 1]

**Control mapping**:
- `action[0]`: Torque applied on the back thigh rotor.
- `action[1]`: Torque applied on the back shin rotor.
- `action[2]`: Torque applied on the back foot rotor.
- `action[3]`: Torque applied on the front thigh rotor.
- `action[4]`: Torque applied on the front shin rotor.
- `action[5]`: Torque applied on the front foot rotor.
"""
}


ENV_SPECS = {
    "HalfCheetah-v4": {
        "description": "A 2D robot consisting of 9 links and 8 joints connecting them "
        "(including a slide joint between the cheetah and the ground).",
        "obs_dim": 17,
        "obs_desc": [
            "position of the center of mass (1D)",
            "orientation of the center of mass (1D)",
            "joint angles (6D)",
            "velocity of the center of mass (1D)",
            "angular velocity of the center of mass (1D)",
            "joint velocities (6D)",
        ],
        "action_dim": 6,
        "action_desc": "Joint torques applied at each of the 6 actuated joints",
        "task": "Move forward (positive x direction) as quickly as possible",
        "reward": "Velocity in the forward direction minus a control cost penalizing"
        " excessive action magnitude",
        "frequency": "50 Hz",
        "example_observation": "-0.10633214, -0.02079649, 0.01070453, 0.13634107, "
        "-0.09362565, 0.0554741, -0.06408735, 0.45652848, -0.0636955, 0.07744387, "
        "0.0090135, 0.01320691, -0.29848951, 0.2165644, -0.36278105, 0.59348507, "
        "-0.82543053",
        "example_action": "-0.03581921,  0.05720125, -0.04736278, 0.00533788, "
        "-0.00698648, 0.08435",
        "obs_space": "Box(-inf, inf, (17,), float32)",
        "action_space": "Box(-1.0, 1.0, (6,), float32)",
    },
    "Humanoid-v4": {
        "description": "A 3D humanoid robot with 17 joints and 21 actuators",
        "state_dim": 376,
        "state_desc": [
            "position of center of mass (3D)",
            "orientation quaternion (4D)",
            "joint angles (17D)",
            "velocity of center of mass (3D)",
            "angular velocity (3D)",
            "joint velocities (17D)",
            "actuator forces (21D)",
            "external contact forces (84D)",
            "vector to target (3D)",
            "actuator activations (21D)",
            "body positions, orientations, and velocities (200D)",
        ],
        "action_dim": 21,
        "action_desc": "Actuator activations for each muscle/motor in the humanoid",
        "task": "Move forward while maintaining balance and minimizing energy usage",
        "reward": "Forward velocity minus energy expenditure and costs for "
        "falling/imbalance",
        "frequency": "50 Hz",
    },
}


def build_system_prompt_from_wang23():
    system_prompt = """
    You are the controller of a quadrupedal robot (A1 robot) with 10 Hz.
    Please inference the output.

    The robot's state is represented by a 33-dimensional input space.
    The first 3 dimensions correspond to the robot's linear velocity.
    The next 3 dimensions denote the robot's angular velocity.
    The following 3 dimensions represent the gravity vector.
    The subsequent 12 dimensions represent the joint positions.
    The final 12 dimensions indicate the velocity of each joint.

    The output space is 12-dimension, which is the joint position.

    The order of the joints is [FRH, FRT, FRC, FLH, FLT, FLC, RRH, RRT, RRC, RLH, RLT,
        RLC].

    After we have the output, we will use 200 Hz PD controller to track it.

    The following are past and consecutive inputs and outputs.
    All numbers are normalized to non-negative integers by our special rule.
    The output would be impacted by the previous inputs.
    The trend of the outputs should be smooth.

    Your output is only one line and starts with "Output:", please do not output other
        redundant words.

    """

    return system_prompt


def build_env_prompt(env_name):
    """Build a system prompt for controlling a specific environment."""

    # Dictionary of environment specifications
    env_specs = {
        "HalfCheetah": {
            "description": "A 2D robot consisting of 9 links and 8 joints connecting "
            "them (including a slide joint between the cheetah and the ground).",
            "state_dim": 17,
            "state_desc": [
                "position of the center of mass (1D)",
                "orientation of the center of mass (1D)",
                "joint angles (6D)",
                "velocity of the center of mass (1D)",
                "angular velocity of the center of mass (1D)",
                "joint velocities (6D)",
            ],
            "action_dim": 6,
            "action_desc": "Joint torques applied at each of the 6 actuated joints",
            "task": "Move forward (positive x direction) as quickly as possible",
            "reward": "Velocity in the forward direction minus a control cost "
            "penalizing excessive action magnitude",
            "frequency": "50 Hz",
        },
        "Humanoid": {
            "description": "A 3D humanoid robot with 17 joints and 21 actuators",
            "state_dim": 376,
            "state_desc": [
                "position of center of mass (3D)",
                "orientation quaternion (4D)",
                "joint angles (17D)",
                "velocity of center of mass (3D)",
                "angular velocity (3D)",
                "joint velocities (17D)",
                "actuator forces (21D)",
                "external contact forces (84D)",
                "vector to target (3D)",
                "actuator activations (21D)",
                "body positions, orientations, and velocities (200D)",
            ],
            "action_dim": 21,
            "action_desc": "Actuator activations for each muscle/motor in the humanoid",
            "task": "Move forward while maintaining balance and minimizing energy "
            "usage",
            "reward": "Forward velocity minus energy expenditure and costs for "
            "falling/imbalance",
            "frequency": "50 Hz",
        },
    }

    # Get the specifications for the requested environment
    if env_name not in env_specs:
        raise ValueError(
            f"Environment {env_name} not supported. Available environments: "
            f"{list(env_specs.keys())}"
        )

    spec = env_specs[env_name]

    # Build the system prompt
    system_prompt = f"""
    You are the controller for a {env_name} robot in a physics simulation.

    Environment Description:
    {spec["description"]}

    State Space:
    The observation is a {spec["state_dim"]}-dimensional vector representing:
    {", ".join([f"{i + 1}. {desc}" for i, desc in enumerate(spec["state_desc"])])}

    Action Space:
    The action is a {spec["action_dim"]}-dimensional vector representing:
    {spec["action_desc"]}

    Task:
    {spec["task"]}

    Reward Structure:
    {spec["reward"]}

    Simulation Frequency:
    {spec["frequency"]}

    Instructions:
    1. Based on the current observation provided below, determine the optimal action
        values.
    2. Your response must ONLY contain the numeric action values separated by commas.
    3. Do not include any explanations or additional text in your response.
    4. Format your response exactly as: "Action: value1, value2, value3, ..."
    5. Each action value should typically be in the range [-1.0, 1.0].

    Current Observation:
    [OBSERVATION_PLACEHOLDER]
    """

    return system_prompt


def format_prompt_with_observation(system_prompt, observation):
    """Replace the observation placeholder with actual observation values."""
    return system_prompt.replace("[OBSERVATION_PLACEHOLDER]", str(observation.tolist()))


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Example for HalfCheetah
    prompt = build_env_prompt("HalfCheetah")

    # Mock observation (17 dimensions for HalfCheetah)
    mock_observation = np.random.randn(17)

    # Format the prompt with the observation
    formatted_prompt = format_prompt_with_observation(prompt, mock_observation)

    print("Sample formatted prompt for HalfCheetah:")
    print(formatted_prompt)
