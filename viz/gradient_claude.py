import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, Button, CheckButtons
# import matplotlib.patches as patches


class GradientVisualization:
    def __init__(self):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.35, right=0.85)

        # Initialize parameters
        self.num_samples = 8
        self.sigma = 0.3
        self.reward_scale = 2.0
        self.gradient_scale = 0.3
        self.sample_distribution = "gaussian"  # 'gaussian', 'uniform', 'mixture'

        # Fixed positions
        self.true_target = np.array([3.0, 2.0])
        self.current_prediction = np.array([1.5, 3.0])

        # Animation parameters
        self.frame = 0
        self.is_playing = False

        # Set up plot
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(0, 5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(
            "Gradient Update Comparison: MSE vs GRPO", fontsize=14, fontweight="bold"
        )

        # Initialize plot elements
        self.setup_plot_elements()
        self.setup_controls()

        # Animation
        self.ani = None

    def setup_plot_elements(self):
        # Uncertainty ellipse
        self.uncertainty_ellipse = Ellipse(
            self.current_prediction,
            width=4 * self.sigma,
            height=3 * self.sigma,
            alpha=0.2,
            color="blue",
            linestyle="--",
        )
        self.ax.add_patch(self.uncertainty_ellipse)

        # Target and current prediction
        self.target_point = self.ax.scatter(
            self.true_target[0],
            self.true_target[1],
            c="green",
            s=100,
            marker="*",
            edgecolors="white",
            linewidth=2,
            label="True Target (y)",
            zorder=5,
        )

        self.current_point = self.ax.scatter(
            self.current_prediction[0],
            self.current_prediction[1],
            c="blue",
            s=80,
            marker="o",
            edgecolors="white",
            linewidth=2,
            label="Current μ(x)",
            zorder=5,
        )

        # Sample points (will be updated)
        self.sample_points = self.ax.scatter(
            [], [], c=[], s=[], alpha=0.7, cmap="Reds", zorder=4
        )

        # Gradient arrows (will be updated)
        self.mse_arrow = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=3, color="green"),
        )
        self.grpo_arrow = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=3, color="red"),
        )

        # Labels
        self.ax.text(
            0.02,
            0.98,
            "Green Arrow: MSE Gradient",
            transform=self.ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            verticalalignment="top",
            fontsize=10,
        )
        self.ax.text(
            0.02,
            0.88,
            "Red Arrow: GRPO Gradient",
            transform=self.ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
            verticalalignment="top",
            fontsize=10,
        )

        # Legend
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))

    def setup_controls(self):
        # Sliders
        slider_height = 0.03
        slider_width = 0.2

        # Number of samples slider
        ax_samples = plt.axes([0.1, 0.25, slider_width, slider_height])
        self.slider_samples = Slider(
            ax_samples, "Samples", 1, 20, valinit=self.num_samples, valstep=1
        )
        self.slider_samples.on_changed(self.update_samples)

        # Sigma slider
        ax_sigma = plt.axes([0.1, 0.20, slider_width, slider_height])
        self.slider_sigma = Slider(
            ax_sigma, "Sigma", 0.1, 1.0, valinit=self.sigma, valfmt="%.2f"
        )
        self.slider_sigma.on_changed(self.update_sigma)

        # Reward scale slider
        ax_reward = plt.axes([0.1, 0.15, slider_width, slider_height])
        self.slider_reward = Slider(
            ax_reward,
            "Reward Scale",
            0.5,
            5.0,
            valinit=self.reward_scale,
            valfmt="%.1f",
        )
        self.slider_reward.on_changed(self.update_reward_scale)

        # Gradient scale slider
        ax_grad = plt.axes([0.1, 0.10, slider_width, slider_height])
        self.slider_grad = Slider(
            ax_grad,
            "Gradient Scale",
            0.1,
            1.0,
            valinit=self.gradient_scale,
            valfmt="%.2f",
        )
        self.slider_grad.on_changed(self.update_gradient_scale)

        # Buttons
        ax_play = plt.axes([0.4, 0.25, 0.08, 0.04])
        self.button_play = Button(ax_play, "Play/Pause")
        self.button_play.on_clicked(self.toggle_animation)

        ax_reset = plt.axes([0.5, 0.25, 0.08, 0.04])
        self.button_reset = Button(ax_reset, "Reset")
        self.button_reset.on_clicked(self.reset_animation)

        # Distribution selector
        ax_dist = plt.axes([0.4, 0.15, 0.15, 0.08])
        self.check_dist = CheckButtons(
            ax_dist, ["Gaussian", "Uniform", "Mixture"], [True, False, False]
        )
        self.check_dist.on_clicked(self.update_distribution)

    def generate_samples(self):
        """Generate samples based on current distribution and parameters"""
        if self.sample_distribution == "gaussian":
            # Gaussian samples around current prediction
            samples = np.random.multivariate_normal(
                self.current_prediction, np.eye(2) * self.sigma**2, self.num_samples
            )
        elif self.sample_distribution == "uniform":
            # Uniform samples in a square around current prediction
            samples = np.random.uniform(
                self.current_prediction - self.sigma * 2,
                self.current_prediction + self.sigma * 2,
                (self.num_samples, 2),
            )
        else:  # mixture
            # Mixture of two gaussians
            n_mix = self.num_samples // 2
            samples1 = np.random.multivariate_normal(
                self.current_prediction + np.array([0.2, 0.2]),
                np.eye(2) * self.sigma**2,
                n_mix,
            )
            samples2 = np.random.multivariate_normal(
                self.current_prediction - np.array([0.2, 0.2]),
                np.eye(2) * self.sigma**2,
                self.num_samples - n_mix,
            )
            samples = np.vstack([samples1, samples2])

        # Add some temporal variation
        angle_offset = self.frame * 0.05
        rotation = np.array(
            [
                [np.cos(angle_offset), -np.sin(angle_offset)],
                [np.sin(angle_offset), np.cos(angle_offset)],
            ]
        )
        samples = samples @ rotation.T

        return samples

    def calculate_rewards(self, samples):
        """Calculate rewards based on distance to true target"""
        distances = np.linalg.norm(samples - self.true_target, axis=1)
        rewards = -(distances**2)  # Negative squared distance
        return rewards

    def calculate_gradients(self, samples, rewards):
        """Calculate MSE and GRPO gradients"""
        # MSE gradient: points toward true target
        mse_grad = (self.true_target - self.current_prediction) * self.gradient_scale

        # GRPO gradient: weighted average toward samples
        weights = np.exp(rewards * self.reward_scale)
        weights = weights / np.sum(weights)  # Normalize

        sample_directions = samples - self.current_prediction
        grpo_grad = (
            np.sum(weights[:, np.newaxis] * sample_directions, axis=0)
            * self.gradient_scale
        )

        return mse_grad, grpo_grad

    def update_plot(self, frame):
        """Update the plot for animation"""
        self.frame = frame

        # Generate new samples
        samples = self.generate_samples()
        rewards = self.calculate_rewards(samples)
        mse_grad, grpo_grad = self.calculate_gradients(samples, rewards)

        # Update sample points
        colors = (rewards - np.min(rewards)) / (
            np.max(rewards) - np.min(rewards) + 1e-8
        )
        sizes = 50 + 100 * colors  # Size based on reward

        self.sample_points.set_offsets(samples)
        self.sample_points.set_array(colors)
        self.sample_points.set_sizes(sizes)

        # Update arrows
        self.mse_arrow.set_position(
            (self.current_prediction[0], self.current_prediction[1])
        )
        self.mse_arrow.xy = self.current_prediction + mse_grad

        self.grpo_arrow.set_position(
            (self.current_prediction[0], self.current_prediction[1])
        )
        self.grpo_arrow.xy = self.current_prediction + grpo_grad

        # Update uncertainty ellipse
        self.uncertainty_ellipse.center = self.current_prediction
        self.uncertainty_ellipse.width = 4 * self.sigma
        self.uncertainty_ellipse.height = 3 * self.sigma

        # Update title with current statistics
        mse_magnitude = np.linalg.norm(mse_grad)
        grpo_magnitude = np.linalg.norm(grpo_grad)
        angle_diff = np.arccos(
            np.clip(
                np.dot(mse_grad, grpo_grad) / (mse_magnitude * grpo_magnitude + 1e-8),
                -1,
                1,
            )
        )

        self.ax.set_title(
            f"Gradient Comparison | MSE: {mse_magnitude:.2f}, GRPO: {grpo_magnitude:.2f}, "
            f"Angle Diff: {np.degrees(angle_diff):.1f}°",
            fontsize=12,
            fontweight="bold",
        )

        return [
            self.sample_points,
            self.mse_arrow,
            self.grpo_arrow,
            self.uncertainty_ellipse,
        ]

    def update_samples(self, val):
        self.num_samples = int(self.slider_samples.val)

    def update_sigma(self, val):
        self.sigma = self.slider_sigma.val

    def update_reward_scale(self, val):
        self.reward_scale = self.slider_reward.val

    def update_gradient_scale(self, val):
        self.gradient_scale = self.slider_grad.val

    def update_distribution(self, label):
        if label == "Gaussian":
            self.sample_distribution = "gaussian"
        elif label == "Uniform":
            self.sample_distribution = "uniform"
        elif label == "Mixture":
            self.sample_distribution = "mixture"

        # Update checkboxes to show only one selected
        for i, checkbox_label in enumerate(["Gaussian", "Uniform", "Mixture"]):
            self.check_dist.set_active(i, checkbox_label == label)

    def toggle_animation(self, event):
        if self.ani is None:
            self.ani = animation.FuncAnimation(
                self.fig,
                self.update_plot,
                frames=range(1000),
                interval=100,
                blit=False,
                repeat=True,
            )
        else:
            if self.is_playing:
                self.ani.pause()
            else:
                self.ani.resume()

        self.is_playing = not self.is_playing

    def reset_animation(self, event):
        self.frame = 0
        if self.ani is not None:
            self.ani.pause()
            self.is_playing = False
        self.update_plot(0)
        plt.draw()

    def show(self):
        """Display the visualization"""
        self.update_plot(0)
        plt.show()


# Create and run the visualization
if __name__ == "__main__":
    viz = GradientVisualization()
    viz.show()

    # Additional utility functions for experimentation
    def run_experiment(num_samples_list, sigma_list, reward_scales):
        """Run experiments with different parameter combinations"""
        results = []

        for n_samples in num_samples_list:
            for sigma in sigma_list:
                for reward_scale in reward_scales:
                    viz = GradientVisualization()
                    viz.num_samples = n_samples
                    viz.sigma = sigma
                    viz.reward_scale = reward_scale

                    # Generate samples and calculate gradients
                    samples = viz.generate_samples()
                    rewards = viz.calculate_rewards(samples)
                    mse_grad, grpo_grad = viz.calculate_gradients(samples, rewards)

                    # Calculate metrics
                    mse_magnitude = np.linalg.norm(mse_grad)
                    grpo_magnitude = np.linalg.norm(grpo_grad)
                    angle_diff = np.arccos(
                        np.clip(
                            np.dot(mse_grad, grpo_grad)
                            / (mse_magnitude * grpo_magnitude + 1e-8),
                            -1,
                            1,
                        )
                    )

                    results.append(
                        {
                            "n_samples": n_samples,
                            "sigma": sigma,
                            "reward_scale": reward_scale,
                            "mse_magnitude": mse_magnitude,
                            "grpo_magnitude": grpo_magnitude,
                            "angle_diff": np.degrees(angle_diff),
                        }
                    )

        return results

    # Example experiment
    results = run_experiment([5, 10, 20], [0.2, 0.5, 0.8], [1.0, 2.0, 3.0])
    print("Sample results:")
    for r in results[:5]:
        print(
            f"Samples: {r['n_samples']}, Sigma: {r['sigma']:.1f}, "
            f"Angle Diff: {r['angle_diff']:.1f}°"
        )
