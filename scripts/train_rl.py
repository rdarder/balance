# /home/rdarder/dev/balance/scripts/train_ppo.py

import os
import time
from dataclasses import dataclass

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

    num_iterations: int = 1_000_000  # Total number of training iterations
    log_interval: int = 500  # Log metrics every N iterations
    eval_interval: int = 10_000  # Evaluate policy every N iterations
    num_eval_episodes: int = 10  # Number of episodes for evaluation
    checkpoint_interval: int = 10_000  # Save checkpoint every N iterations

    root_dir: str = "ppo_training_results"  # Directory to save results

@dataclass
class Settings:
    sim: SimulationSettings
    behavior: BehaviorSettings
    reset: ResetSettings
    ppo: PPOTrainingSettings

def train_eval(
    sim_settings: SimulationSettings,
    behavior_settings: BehaviorSettings,
    reset_settings: ResetSettings,
    ppo_settings: PPOTrainingSettings,
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

    train_tf_env = make_tf_env(env_factory, ppo_settings.num_parallel_environments)
    eval_py_env = env_factory()  # Separate env for evaluation
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    print("Observation Spec:", train_tf_env.observation_spec())
    print("Action Spec:", train_tf_env.action_spec())
    print("TimeStep Spec:", train_tf_env.time_step_spec())

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
        # normalize_observations=True, # Consider adding normalization layers
        # normalize_rewards=True,      # Consider reward normalization
        train_step_counter=global_step,
        debug_summaries=False,  # Set True for more detailed TensorBoard logs
        summarize_grads_and_vars=False,
    )
    agent.initialize()

    # --- Replay Buffer and Data Collection ---
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_tf_env.batch_size,  # Matches num_parallel_environments
        max_length=ppo_settings.replay_buffer_capacity,
    )
    environment_steps_metric = tf_metrics.EnvironmentSteps()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_tf_env,
        agent.collect_policy,
        observers=[
            replay_buffer.add_batch,
            environment_steps_metric,
        ],
        num_steps=ppo_settings.collect_steps_per_iteration,
    )

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
        collect_time_step = train_tf_env.current_time_step()  # Get initial state if needed
        collect_driver.run()  # Fills the replay buffer

        # --- Train Agent ---
        # Sample all data collected in this iteration
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience=experience)
        replay_buffer.clear()  # Clear buffer for next on-policy iteration

        step = agent.train_step_counter.numpy()
        iter_time = time.time() - iter_start_time

        # --- Logging ---
        if step % ppo_settings.log_interval == 0:
            print(
                f"Iteration {step}: Loss = {train_loss.loss.numpy():.4f}, Time = {iter_time:.2f}s"
            )
            # Log training loss and other agent metrics
            tf.summary.scalar("Agent/loss", train_loss.loss, step=step)
            # Log time per iteration
            tf.summary.scalar("Timing/Iteration_Time", iter_time, step=step)
            # Log number of environment steps collected
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
            # Log evaluation metrics
            with eval_summary_writer.as_default():
                tf.summary.scalar("Metrics/AverageReturn", avg_return, step=step)
                # You could add AverageEpisodeLength here too if needed

        # --- Checkpointing ---
        if step % ppo_settings.checkpoint_interval == 0:
            train_checkpointer.save(global_step=step)
            policy_checkpointer.save(global_step=step)
            # Save the policy in SavedModel format for deployment/inference
            saved_model_path = os.path.join(saved_model_dir, f"policy_step_{step}")
            model_saver.save(saved_model_path)
            print(f"Checkpoint saved at iteration {step}")

    # --- Final Save ---
    train_checkpointer.save(global_step=step)
    policy_checkpointer.save(global_step=step)
    saved_model_path = os.path.join(saved_model_dir, f"policy_final_step_{step}")
    model_saver.save(saved_model_path)

    print(f"Training finished in {(time.time() - start_time):.2f} seconds.")


def make_tf_env(env_factory, parallel_environments):
    if parallel_environments > 1:
        return tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [env_factory] * parallel_environments
            )
        )
    else:
        return  tf_py_environment.TFPyEnvironment(env_factory())


def main(script_path: str):
    # tf_agent needs a multiprocessing wrapper for main. it uses
    # https://abseil.io/docs/python/guides/app
    # which seems to be related with bazel and how python apps are run.
    # ultimately we get an extra positional parameter with the script relative path.
    # Unsure what to do with it so for now ignoring it.

    settings= tyro.cli(Settings)
    train_eval(sim_settings=settings.sim, behavior_settings=settings.behavior,
               reset_settings=settings.reset, ppo_settings=settings.ppo)

if __name__ == "__main__":
    handle_main(main)
