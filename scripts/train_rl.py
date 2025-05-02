import os

import mujoco

from balance.tf_agents_utils import run_with_tyro_and_tfagents_mp

os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
import time
from dataclasses import dataclass

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment, tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from balance.observation_processing import EncoderWrapper, WorldModelEncoderSettings

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
    num_epochs: int = 10
    entropy_regularization: float = 0.0
    value_pred_loss_coef: float = 0.5
    importance_ratio_clipping: float = 0.2
    use_gae: bool = True
    lambda_value: float = 0.95
    discount_factor: float = 0.99

    actor_fc_layers: tuple[int, ...] = (64, 64)
    value_fc_layers: tuple[int, ...] = (128, 128)

    collect_steps_per_iteration: int = 1000
    replay_buffer_capacity: int = collect_steps_per_iteration + 1

    num_parallel_environments: int = 1 # Default to 1, check added in train_eval

    num_iterations: int = 100_000  # Total number of training iterations
    log_interval: int = 100  # Log metrics every N iterations
    eval_interval: int = 1_000  # Evaluate policy every N iterations
    num_eval_episodes: int = 10  # Number of episodes for evaluation
    checkpoint_interval: int = 100  # Save checkpoint every N iterations

    root_dir: str = "ppo_training_results"  # Directory to save results


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

    # --- Define Metrics ---
    # Metrics for data collection phase (logged every log_interval)
    train_step_counter = tf.compat.v1.train.get_or_create_global_step() # Use global_step alias
    # Use a buffer size reflecting recent episodes within the log interval
    train_buffer_size = ppo_settings.num_eval_episodes # Or adjust as needed
    train_return_metric = tf_metrics.AverageReturnMetric(
        buffer_size=train_buffer_size, name='TrainAverageReturn', batch_size=ppo_settings.num_parallel_environments
    )
    train_length_metric = tf_metrics.AverageEpisodeLengthMetric(
        buffer_size=train_buffer_size, name='TrainAverageEpisodeLength', batch_size=ppo_settings.num_parallel_environments
    )
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    # Number of episodes completed during collection
    train_episodes_metric = tf_metrics.NumberOfEpisodes(name='TrainNumberOfEpisodes')

    # Metrics for evaluation phase (logged every eval_interval)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=ppo_settings.num_eval_episodes, name='EvalAverageReturn'),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=ppo_settings.num_eval_episodes, name='EvalAverageEpisodeLength'),
    ]
    # ----------------------

    def env_factory():
        model = load_robot_model()
        model_data = mujoco.MjData(model)
        gym_env =  SegwayEnv(model, model_data, sim_settings, behavior_settings, reset_settings)
        return gym_env

    # --- Create Training Environment ---
    train_tf_env = make_tf_env(
        env_factory,
        ppo_settings.num_parallel_environments,
        world_model_settings
    )

    # --- Create and Conditionally Wrap Evaluation Environment ---
    eval_py_env = env_factory()
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
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
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
        train_step_counter=train_step_counter, # Use the counter
        debug_summaries=False,
        summarize_grads_and_vars=False,
    )
    agent.initialize()

    # --- Replay Buffer and Data Collection ---
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_tf_env.batch_size,
        max_length=ppo_settings.replay_buffer_capacity,
    )

    # --- Define Collection Observers ---
    collect_observers = [
        replay_buffer.add_batch,
        environment_steps_metric,
        train_episodes_metric,
        train_return_metric,
        train_length_metric,
    ]
    # ---------------------------------

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_tf_env,
        agent.collect_policy,
        observers=collect_observers, # Pass the full list
        num_steps=ppo_settings.collect_steps_per_iteration,
    )

    # --- Checkpointing ---
    # Group metrics to be saved in the checkpoint
    checkpoint_metrics = metric_utils.MetricsGroup(
        [environment_steps_metric, train_episodes_metric, train_return_metric, train_length_metric] + eval_metrics,
        name="checkpoint_metrics"
    )
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=agent,
        global_step=train_step_counter,
        metrics=checkpoint_metrics # Save all metrics
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "policy"),
        policy=agent.policy,
        global_step=train_step_counter,
    )
    # model_saver = policy_saver.PolicySaver(agent.policy, train_step=train_step_counter) # Keep commented out

    train_checkpointer.initialize_or_restore()

    # --- Optimize Training with tf.function ---
    collect_driver.run = common.function(collect_driver.run)
    agent.train = common.function(agent.train)

    # --- Training Loop ---
    print(f"Starting training for {ppo_settings.num_iterations} iterations...")
    start_time = time.time()
    for iteration in range(ppo_settings.num_iterations):
        iter_start_time = time.time()

        collect_driver.run()
        experience = replay_buffer.gather_all()
        if tf.nest.is_nested(experience) and not tf.nest.flatten(experience):
             print(f"Warning: Iteration {iteration}: No experience gathered, skipping training.")
             replay_buffer.clear()
             continue
        train_loss = agent.train(experience=experience)
        replay_buffer.clear()
        step = agent.train_step_counter.numpy()
        iter_time = time.time() - iter_start_time

        # --- Logging (every log_interval) ---
        if step % ppo_settings.log_interval == 0:
            # Get results from training metrics
            train_avg_return = train_return_metric.result()
            train_avg_length = train_length_metric.result()
            train_num_episodes = train_episodes_metric.result()
            env_steps = environment_steps_metric.result()

            print(
                f"Iteration {step}: Loss = {train_loss.loss.numpy():.4f}, "
                f"Train Return = {train_avg_return:.2f}, Train Length = {train_avg_length:.2f}, "
                f"Episodes = {train_num_episodes}, Env Steps = {env_steps}, Time = {iter_time:.2f}s"
            )
            with train_summary_writer.as_default():
                tf.summary.scalar("Agent/loss", train_loss.loss, step=step)
                tf.summary.scalar("Timing/Iteration_Time", iter_time, step=step)
                tf.summary.scalar("Environment/Steps", env_steps, step=step)
                # Log training metrics to TensorBoard
                tf.summary.scalar("Train/AverageReturn", train_avg_return, step=step)
                tf.summary.scalar("Train/AverageEpisodeLength", train_avg_length, step=step)
                tf.summary.scalar("Train/NumberOfEpisodes", train_num_episodes, step=step)

            # Reset training metrics for the next interval
            train_return_metric.reset()
            train_length_metric.reset()
            train_episodes_metric.reset()
            # Do NOT reset environment_steps_metric, it's cumulative
        # ------------------------------------

        # --- Evaluation (every eval_interval) ---
        if step % ppo_settings.eval_interval == 0:
            eval_start_time = time.time()
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                agent.policy,
                num_episodes=ppo_settings.num_eval_episodes,
                train_step=step,
                summary_writer=eval_summary_writer,
                summary_prefix='Eval', # Changed prefix to Eval/
            )
            eval_time = time.time() - eval_start_time

            # Log results to console
            eval_avg_return = results['EvalAverageReturn'].numpy() # Use updated name
            eval_avg_length = results['EvalAverageEpisodeLength'].numpy() # Use updated name
            print(
                f"--- EVALUATION Iteration {step}: Average Return = {eval_avg_return:.2f}, Average Length = {eval_avg_length:.2f}, Eval Time = {eval_time:.2f}s ---"
            )

            # Reset evaluation metrics for the next evaluation run
            for metric in eval_metrics:
                metric.reset()
        # -----------------------------------------

        # --- Checkpointing (every checkpoint_interval) ---
        if step % ppo_settings.checkpoint_interval == 0:
            train_checkpointer.save(global_step=step)
            policy_checkpointer.save(global_step=step)
            saved_model_path = os.path.join(saved_model_dir, f"policy_step_{step}")
            # model_saver.save(saved_model_path)
            print(f"Checkpoint saved at iteration {step}")

    # --- Final Save ---
    train_checkpointer.save(global_step=step)
    policy_checkpointer.save(global_step=step)
    saved_model_path = os.path.join(saved_model_dir, f"policy_final_step_{step}")
    # model_saver.save(saved_model_path)

    print(f"Training finished in {(time.time() - start_time):.2f} seconds.")


def make_tf_env(
    env_factory,
    parallel_environments: int,
    world_model_settings: WorldModelEncoderSettings
) -> tf_py_environment.TFPyEnvironment:
    """Creates a TFPyEnvironment, potentially parallel and wrapped."""

    if parallel_environments > 1:
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [env_factory for _ in range(parallel_environments)]
        )
    else:
        py_env = env_factory()

    if world_model_settings.use_encoder:
        print("Wrapping environment with world model encoder.")
        try:
            py_env = EncoderWrapper(
                environment=py_env,
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

    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    print(f"Final Observation Spec: {tf_env.observation_spec()}")
    return tf_env


def main(settings: Settings):
    train_eval(
        sim_settings=settings.sim,
        behavior_settings=settings.behavior,
        reset_settings=settings.reset,
        ppo_settings=settings.ppo,
        world_model_settings=settings.world_model
    )

if __name__ == "__main__":
    run_with_tyro_and_tfagents_mp(settings_cls=Settings, main_func=main)
