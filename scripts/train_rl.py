import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
print("Set TF_USE_LEGACY_KERAS=1 to force Keras 2 usage.")

import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import tensorflow as tf
import tyro
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment, tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.system.system_multiprocessing import handle_main


from balance.observation_processing import EncoderWrapper

from balance.env import (
    BehaviorSettings,
    ResetSettings,
    SegwayEnv,
    SimulationSettings,
)
from balance.rl_model import create_ppo_networks
from balance.utils import load_robot_model

@dataclass
class PPOTrainingSettings:
    """Hyperparameters for PPO training."""

    # Agent params
    learning_rate: float = 3e-4
    num_epochs: int = 10  # Number of PPO epochs per data collection iteration
    entropy_regularization: float = 0.0
    value_pred_loss_coef: float = 0.5
    importance_ratio_clipping: float = 0.2
    use_gae: bool = True
    lambda_value: float = 0.95
    discount_factor: float = 0.99  # Agent discount factor (gamma)

    actor_fc_layers: tuple[int, ...] = (32, 32)
    value_fc_layers: tuple[int, ...] = (256, 256)

    collect_steps_per_iteration: int = 1000
    replay_buffer_capacity: int = collect_steps_per_iteration + 1

    num_parallel_environments: int = 4  # Number of environments to run in parallel

    num_iterations: int = 100_000  # Total number of training iterations
    log_interval: int = 10  # Log metrics every N iterations
    eval_interval: int = 1_000  # Evaluate policy every N iterations
    num_eval_episodes: int = 10  # Number of episodes for evaluation
    checkpoint_interval: int = 100  # Save checkpoint every N iterations

    root_dir: str = "ppo_training_results"  # Directory to save results


@dataclass
class WorldModelEncoderSettings:
    """Settings for using the world model encoder as observation preprocessor."""
    use_encoder: bool = False # Set to True to enable the encoder wrapper
    # Path to the *specific checkpoint file prefix* (e.g., ckpt-X) from world model training
    checkpoint_path: Optional[str] = None
    latent_dim: int = 32 # Must match the trained world model
    # Input features dim = IMU(6) + Action(2) used during WM training
    input_features: int = 8

    def __post_init__(self):
        if self.use_encoder:
            if self.checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided if use_encoder is True.")
            # Check if the index file exists, indicating a valid checkpoint prefix
            if not Path(f"{self.checkpoint_path}.index").exists():
                 raise FileNotFoundError(
                     f"Checkpoint index file not found for '{self.checkpoint_path}'. "
                     "Ensure path points to the checkpoint prefix (e.g., ckpt-X)."
                 )

@dataclass
class Settings:
    sim: SimulationSettings
    behavior: BehaviorSettings
    reset: ResetSettings
    ppo: PPOTrainingSettings
    world_model: WorldModelEncoderSettings


def train_eval(
    sim_settings: SimulationSettings,
    behavior_settings: BehaviorSettings,
    reset_settings: ResetSettings,
    ppo_settings: PPOTrainingSettings,
    world_model_settings: WorldModelEncoderSettings,
):
    """Main training and evaluation function."""

    root_dir = os.path.expanduser(ppo_settings.root_dir)
    train_dir = os.path.join(root_dir, "train")
    eval_dir = os.path.join(root_dir, "eval")
    saved_model_dir = os.path.join(root_dir, "policy")

    train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=10000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=10000)
    train_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=ppo_settings.num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=ppo_settings.num_eval_episodes
        ),
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()

    def env_factory():
        model = load_robot_model()
        gym_env =  SegwayEnv(model, sim_settings, behavior_settings, reset_settings)
        return gym_env

    # --- Create Training Environment (Pass world_model_settings) ---
    train_tf_env = make_tf_env(
        env_factory,
        ppo_settings.num_parallel_environments,
        world_model_settings # Pass the settings here
    )

    # --- Create and Conditionally Wrap Evaluation Environment ---
    eval_py_env = env_factory()  # Create base eval env

    # Conditionally wrap the evaluation PyEnvironment
    if world_model_settings.use_encoder:
        print("Wrapping evaluation environment with world model encoder.")
        try:
            eval_py_env = EncoderWrapper(
                environment=eval_py_env,
                encoder_checkpoint_path=world_model_settings.checkpoint_path,
                latent_dim=world_model_settings.latent_dim,
                input_features=world_model_settings.input_features
            )
        except Exception as e:
            print(f"FATAL: Failed to initialize EncoderWrapper for evaluation: {e}")
            raise e
    else:
        print("Using raw observations for evaluation environment.")

    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env) # Convert potentially wrapped env
    print(f"Evaluation Observation Spec: {eval_tf_env.observation_spec()}")
    # -----------------------------------------------------------

    # --- Agent and Network Setup ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=ppo_settings.learning_rate)

    actor_net, value_net = create_ppo_networks(
        train_tf_env.observation_spec(),
        train_tf_env.action_spec(),
        actor_fc_layers=ppo_settings.actor_fc_layers,
        value_fc_layers=ppo_settings.value_fc_layers,
    )

    agent = ppo_agent.PPOAgent(
        time_step_spec=train_tf_env.time_step_spec(),
        action_spec=train_tf_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=ppo_settings.num_epochs,
        gradient_clipping=0.5,
        entropy_regularization=ppo_settings.entropy_regularization,
        importance_ratio_clipping=ppo_settings.importance_ratio_clipping,
        value_pred_loss_coef=ppo_settings.value_pred_loss_coef,
        use_gae=ppo_settings.use_gae,
        lambda_value=ppo_settings.lambda_value,
        discount_factor=ppo_settings.discount_factor,
        train_step_counter=global_step,
        debug_summaries=False,
        summarize_grads_and_vars=False,
    )
    agent.initialize()

    # --- Replay Buffer and Data Collection ---
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, # Uses potentially wrapped spec
        batch_size=train_tf_env.batch_size,
        max_length=ppo_settings.replay_buffer_capacity,
    )
    environment_steps_metric = tf_metrics.EnvironmentSteps()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_tf_env, # Uses potentially wrapped env
        agent.collect_policy,
        observers=[
            replay_buffer.add_batch,
            environment_steps_metric,
        ],
        num_steps=ppo_settings.collect_steps_per_iteration,
    )

    # --- Checkpointing and Saving ---
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, "train_metrics"),
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "policy"),
        policy=agent.policy,
        global_step=global_step,
    )
    model_saver = policy_saver.PolicySaver(agent.policy, train_step=global_step)

    train_checkpointer.initialize_or_restore()

    # --- Evaluation Function ---
    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]  # Extract scalar value

    # --- Optimize Training with tf.function ---
    collect_driver.run = common.function(collect_driver.run)
    agent.train = common.function(agent.train)

    # --- Training Loop ---
    print(f"Starting training for {ppo_settings.num_iterations} iterations...")
    start_time = time.time()
    for iteration in range(ppo_settings.num_iterations):
        iter_start_time = time.time()

        # --- Collect Data ---
        collect_driver.run()

        # --- Train Agent ---
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience=experience)
        replay_buffer.clear()

        step = agent.train_step_counter.numpy()
        iter_time = time.time() - iter_start_time

        # --- Logging ---
        if step % ppo_settings.log_interval == 0:
            print(
                f"Iteration {step}: Loss = {train_loss.loss.numpy():.4f}, Time = {iter_time:.2f}s"
            )
            tf.summary.scalar("Agent/loss", train_loss.loss, step=step)
            tf.summary.scalar("Timing/Iteration_Time", iter_time, step=step)
            env_steps = environment_steps_metric.result()
            tf.summary.scalar("Environment/Steps", env_steps, step=step)

        # --- Evaluation ---
        if step % ppo_settings.eval_interval == 0:
            eval_start_time = time.time()
            avg_return = compute_avg_return(
                eval_tf_env, agent.policy, ppo_settings.num_eval_episodes
            )
            eval_time = time.time() - eval_start_time
            print(
                f"Iteration {step}: Average Return = {avg_return:.2f}, Eval Time = {eval_time:.2f}s"
            )
            with eval_summary_writer.as_default():
                tf.summary.scalar("Metrics/AverageReturn", avg_return, step=step)

        # --- Checkpointing ---
        if step % ppo_settings.checkpoint_interval == 0:
            train_checkpointer.save(global_step=step)
            policy_checkpointer.save(global_step=step)
            saved_model_path = os.path.join(saved_model_dir, f"policy_step_{step}")
            model_saver.save(saved_model_path)
            print(f"Checkpoint saved at iteration {step}")

    # --- Final Save ---
    train_checkpointer.save(global_step=step)
    policy_checkpointer.save(global_step=step)
    saved_model_path = os.path.join(saved_model_dir, f"policy_final_step_{step}")
    model_saver.save(saved_model_path)

    print(f"Training finished in {(time.time() - start_time):.2f} seconds.")


# Modify the signature and add wrapping logic
def make_tf_env(
    env_factory,
    parallel_environments: int,
    world_model_settings: WorldModelEncoderSettings # Add this parameter
) -> tf_py_environment.TFPyEnvironment:
    """Creates a TFPyEnvironment, potentially parallel and wrapped."""

    # Create the base PyEnvironment(s)
    if parallel_environments > 1:
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [env_factory] * parallel_environments
        )
    else:
        py_env = env_factory() # Single base environment

    # --- Conditionally Wrap the PyEnvironment(s) ---
    if world_model_settings.use_encoder:
        print("Wrapping environment with world model encoder.")
        try:
            if parallel_environments > 1:
                 raise ValueError(
                     "EncoderWrapper currently supports only num_parallel_environments=1. "
                     "Set ppo.num_parallel_environments=1 when using world_model.use_encoder=True."
                 )

            py_env = EncoderWrapper(
                environment=py_env, # Pass the single base py_env
                encoder_checkpoint_path=world_model_settings.checkpoint_path,
                latent_dim=world_model_settings.latent_dim,
                input_features=world_model_settings.input_features
            )
            print(f"Using EncoderWrapper.")
        except Exception as e:
            print(f"FATAL: Failed to initialize EncoderWrapper: {e}")
            raise e
    else:
        print("Using raw observations from environment.")
    # -------------------------------------------------

    # Convert the potentially wrapped PyEnvironment to TFPyEnvironment
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    print(f"Final Observation Spec: {tf_env.observation_spec()}")
    return tf_env


def main():
    # tf_agent needs a multiprocessing wrapper for main. it uses
    # https://abseil.io/docs/python/guides/app
    # which seems to be related with bazel and how python apps are run.
    # ultimately we get an extra positional parameter with the script relative path.
    # Unsure what to do with it so for now ignoring it.

    settings = tyro.cli(Settings)
    # Pass the world_model settings to train_eval
    train_eval(
        sim_settings=settings.sim,
        behavior_settings=settings.behavior,
        reset_settings=settings.reset,
        ppo_settings=settings.ppo,
        world_model_settings=settings.world_model
    )

if __name__ == "__main__":
    main()
