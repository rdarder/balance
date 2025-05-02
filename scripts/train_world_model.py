# /home/rdarder/dev/balance/scripts/train_world_model.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
print("Set TF_USE_LEGACY_KERAS=1 to force Keras 2 usage.")

import tensorflow as tf
import numpy as np
import tyro
from dataclasses import dataclass, field
from pathlib import Path
import time
import random
from typing import List, Tuple, Generator, Dict

# Assuming models are in balance.world_model
from balance.world_model import TrainingEncoderNetwork, PredictorMLP
# Assuming env spec might be useful, though not strictly required if data defines dims
# from balance.env import SegwayEnv, SimulationSettings # Example if needed

# --- Configuration ---

@dataclass
class DataSettings:
    """Data loading and preparation settings."""
    data_files: List[str] = field(default_factory=lambda: ["simulation_data/simulated_data_*.npz"]) # Glob pattern
    validation_split: float = 0.1 # Fraction of episodes for validation
    warmup_steps: int = 10      # Steps to feed encoder before prediction starts
    prediction_steps: int = 15  # Number of future steps to predict and include in loss

@dataclass
class ModelSettings:
    """Model architecture settings."""
    latent_dim: int = 32
    predictor_hidden_dim: int = 32
    # Assuming input_features = obs(8) + action(2) = 10
    # Assuming action_dim = 2

@dataclass
class TrainingSettings:
    """Training hyperparameters and settings."""
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_train_steps: int = 20_000 # Total training iterations
    log_interval: int = 100       # Steps between logging training metrics
    eval_interval: int = 1000     # Steps between running validation
    checkpoint_interval: int = 2000 # Steps between saving checkpoints
    log_dir: str = "world_model_logs"

@dataclass
class Settings:
    """Main settings container."""
    data: DataSettings
    model: ModelSettings
    train: TrainingSettings


# --- Data Loading and Preparation ---

def load_and_prepare_data(
    data_files_patterns: List[str],
    min_length: int,
    validation_split: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Loads data from npz files, filters by length, and splits."""
    all_episodes = []
    print("Loading data...")
    for pattern in data_files_patterns:
        # Use pathlib's glob
        for file_path in Path().glob(pattern):
            try:
                with np.load(str(file_path), allow_pickle=False) as data:
                    # data is an NpzFile object, iterate through arrays ('arr_0', 'arr_1', ...)
                    for key in data.files:
                        all_episodes.append(data[key])
                print(f"  Loaded {len(data.files)} episodes from {file_path}")
            except Exception as e:
                print(f"  Warning: Could not load or read {file_path}: {e}")

    print(f"Total episodes loaded: {len(all_episodes)}")

    # Filter episodes that are too short
    valid_episodes = [ep for ep in all_episodes if len(ep) >= min_length]
    print(f"Episodes after filtering (min_length={min_length}): {len(valid_episodes)}")

    if not valid_episodes:
        raise ValueError("No valid episodes found after filtering. Check data or parameters.")

    # Shuffle and split
    random.shuffle(valid_episodes)
    num_validation = int(len(valid_episodes) * validation_split)
    num_training = len(valid_episodes) - num_validation

    train_episodes = valid_episodes[:num_training]
    val_episodes = valid_episodes[num_training:]

    print(f"Training episodes: {len(train_episodes)}")
    print(f"Validation episodes: {len(val_episodes)}")

    return train_episodes, val_episodes

def build_dataset(
    episodes: List[np.ndarray],
    warmup_steps: int,
    prediction_steps: int,
    batch_size: int,
    input_features: int, # obs_dim + action_dim
    action_dim: int
) -> tf.data.Dataset:
    """Builds a tf.data.Dataset that samples subsequences."""

    sequence_length = warmup_steps + prediction_steps

    def generator() -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        while True:
            # Select a random episode
            episode = random.choice(episodes)
            episode_len = len(episode)

            # Select a random start index for the subsequence
            start_idx = random.randint(0, episode_len - sequence_length)
            end_idx = start_idx + sequence_length

            subsequence = episode[start_idx:end_idx] # Shape: (sequence_length, input_features)

            # Separate inputs (obs+prev_action) and the actions *taken during* the sequence
            # Input for step t is (obs_t, action_{t-1})
            # Action for step t is action_t (used by predictor)
            # The last 'action_dim' columns of the subsequence are prev_actions
            # We need the actions *corresponding* to the observations in the sequence

            # Example: subsequence[i] = [obs_i, prev_action_{i-1}]
            # We need action_i for the predictor when predicting step i+1 from state i

            # Shift actions: action[t] is stored in subsequence[t+1]'s last columns
            # Need actions from index `warmup_steps` up to `sequence_length - 1`
            actions_for_prediction = subsequence[warmup_steps:, -action_dim:]

            # Ensure correct types
            subsequence_tensor = tf.convert_to_tensor(subsequence, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions_for_prediction, dtype=tf.float32)

            yield subsequence_tensor, actions_tensor

    # Determine output signature based on expected shapes
    output_signature = (
        tf.TensorSpec(shape=(sequence_length, input_features), dtype=tf.float32),
        tf.TensorSpec(shape=(prediction_steps, action_dim), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# --- Training Step ---

@tf.function
def train_step(
    encoder: TrainingEncoderNetwork,
    predictor: PredictorMLP,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    subsequences: tf.Tensor, # Shape: (batch, warmup+pred, features)
    actions: tf.Tensor,      # Shape: (batch, pred, action_dim)
    warmup_steps: int,
    prediction_steps: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Performs one training step."""

    with tf.GradientTape() as tape:
        # Encode the full sequence
        # sequence_states shape: (batch, warmup+pred, latent_dim)
        # final_state shape: (batch, latent_dim) -> not used directly here
        sequence_states, _ = encoder(subsequences, training=True)

        # Get initial state for prediction and target states
        initial_pred_state = sequence_states[:, warmup_steps - 1, :] # Shape: (batch, latent_dim)
        target_states = sequence_states[:, warmup_steps:, :]      # Shape: (batch, pred, latent_dim)

        # Stop gradient flow to targets - predictor learns to match encoder, not change it here
        target_states = tf.stop_gradient(target_states)

        # Multi-step prediction loop
        predicted_states = []
        current_state = initial_pred_state
        for i in range(prediction_steps):
            current_action = actions[:, i, :] # Shape: (batch, action_dim)
            next_state_pred = predictor(current_state, current_action, training=True)
            predicted_states.append(next_state_pred)
            current_state = next_state_pred # Use prediction for next step

        # Stack predictions: list of (batch, latent) -> (batch, pred, latent)
        predicted_states_tensor = tf.stack(predicted_states, axis=1)

        # Calculate loss
        loss = loss_fn(target_states, predicted_states_tensor)

    # Compute and apply gradients
    trainable_vars = encoder.trainable_variables + predictor.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Calculate latent state std dev as a proxy for collapse
    latent_std = tf.math.reduce_std(initial_pred_state, axis=0) # Std dev across batch for each latent dim
    mean_latent_std = tf.reduce_mean(latent_std) # Average std dev over latent dimensions

    return loss, mean_latent_std

# --- Validation Step ---

@tf.function
def validation_step(
    encoder: TrainingEncoderNetwork,
    predictor: PredictorMLP,
    loss_fn: tf.keras.losses.Loss,
    subsequences: tf.Tensor,
    actions: tf.Tensor,
    warmup_steps: int,
    prediction_steps: int
) -> tf.Tensor:
    """Calculates validation loss for one batch."""
    # Encode
    sequence_states, _ = encoder(subsequences, training=False)
    initial_pred_state = sequence_states[:, warmup_steps - 1, :]
    target_states = sequence_states[:, warmup_steps:, :]
    # No stop_gradient needed for validation loss calculation itself

    # Predict
    predicted_states = []
    current_state = initial_pred_state
    for i in range(prediction_steps):
        current_action = actions[:, i, :]
        next_state_pred = predictor(current_state, current_action, training=False)
        predicted_states.append(next_state_pred)
        current_state = next_state_pred

    predicted_states_tensor = tf.stack(predicted_states, axis=1)

    # Loss
    loss = loss_fn(target_states, predicted_states_tensor)
    return loss


# --- Main Training Function ---

def run_training(settings: Settings):
    """Main function to run the world model training."""

    # --- Setup ---
    log_dir = Path(settings.train.log_dir)
    summary_writer = tf.summary.create_file_writer(str(log_dir / "train"))
    val_summary_writer = tf.summary.create_file_writer(str(log_dir / "validation"))

    ACTION_DIM = 2
    OBS_DIM = 6 # We're only saving IMU readings, discarding desired speed/turn.
    INPUT_FEATURES = OBS_DIM + ACTION_DIM

    # --- Load Data ---
    min_req_length = settings.data.warmup_steps + settings.data.prediction_steps
    train_episodes, val_episodes = load_and_prepare_data(
        settings.data.data_files, min_req_length, settings.data.validation_split
    )

    # --- Build Datasets ---
    train_dataset = build_dataset(
        train_episodes,
        settings.data.warmup_steps,
        settings.data.prediction_steps,
        settings.train.batch_size,
        INPUT_FEATURES,
        ACTION_DIM
    )
    val_dataset = build_dataset(
        val_episodes,
        settings.data.warmup_steps,
        settings.data.prediction_steps,
        settings.train.batch_size, # Use same batch size for consistency
        INPUT_FEATURES,
        ACTION_DIM
    )
    # Create iterators
    train_iterator = iter(train_dataset)
    val_iterator = iter(val_dataset)


    # --- Create Models and Optimizer ---
    encoder = TrainingEncoderNetwork(latent_dim=settings.model.latent_dim)
    predictor = PredictorMLP(
        latent_dim=settings.model.latent_dim,
        action_dim=ACTION_DIM,
        hidden_dim=settings.model.predictor_hidden_dim
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.train.learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # --- Build models (optional but good practice) ---
    # Create dummy input shapes to build the models explicitly
    dummy_seq_shape = (settings.train.batch_size, min_req_length, INPUT_FEATURES)
    dummy_action_shape = (settings.train.batch_size, settings.data.prediction_steps, ACTION_DIM)
    dummy_latent_shape = (settings.train.batch_size, settings.model.latent_dim)
    encoder.build(dummy_seq_shape)
    predictor.build([dummy_latent_shape, (settings.train.batch_size, ACTION_DIM)]) # Predictor takes single action


    # --- Checkpointing ---
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0),
        optimizer=optimizer,
        encoder=encoder,
        predictor=predictor
    )
    manager = tf.train.CheckpointManager(
        checkpoint, str(checkpoint_dir), max_to_keep=3
    )
    # Restore latest checkpoint if exists
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")


    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time()
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    latent_std_metric = tf.keras.metrics.Mean(name='latent_std')

    # Use tf.range for the loop if using @tf.function on the outer loop (not done here)
    for step in range(int(checkpoint.step), settings.train.num_train_steps):
        # Get next batch
        subsequences, actions = next(train_iterator)

        # Perform training step
        loss, mean_latent_std = train_step(
            encoder, predictor, optimizer, loss_fn,
            subsequences, actions,
            settings.data.warmup_steps, settings.data.prediction_steps
        )

        # Update metrics
        train_loss_metric(loss)
        latent_std_metric(mean_latent_std)
        checkpoint.step.assign_add(1)

        # --- Logging ---
        if step % settings.train.log_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = settings.train.log_interval / elapsed_time
            print(
                f"Step {step}/{settings.train.num_train_steps}, "
                f"Loss: {train_loss_metric.result():.4f}, "
                f"LatentStd: {latent_std_metric.result():.4f}, "
                f"Steps/sec: {steps_per_sec:.2f}"
            )
            with summary_writer.as_default(step=step):
                tf.summary.scalar('loss', train_loss_metric.result())
                tf.summary.scalar('latent_std_dev', latent_std_metric.result())
                tf.summary.scalar('steps_per_sec', steps_per_sec)

            # Reset metrics and timer for the next interval
            train_loss_metric.reset_state()
            latent_std_metric.reset_state()
            start_time = time.time()

        # --- Validation ---
        if step % settings.train.eval_interval == 0:
            print(f"--- Running Validation at Step {step} ---")
            val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
            # Typically run validation over a fixed number of batches or the whole val set
            num_val_batches = 50 # Example: run on 50 batches
            for _ in range(num_val_batches):
                try:
                    val_subsequences, val_actions = next(val_iterator)
                    val_loss = validation_step(
                        encoder, predictor, loss_fn,
                        val_subsequences, val_actions,
                        settings.data.warmup_steps, settings.data.prediction_steps
                    )
                    val_loss_metric(val_loss)
                except StopIteration:
                    # Reset iterator if validation set is exhausted
                    val_iterator = iter(val_dataset)
                    break # Stop validation loop if dataset ends

            avg_val_loss = val_loss_metric.result()
            print(f"--- Validation Loss at Step {step}: {avg_val_loss:.4f} ---")
            with val_summary_writer.as_default(step=step):
                tf.summary.scalar('loss', avg_val_loss)
            val_loss_metric.reset_state() # Reset for next eval


        # --- Checkpointing ---
        if step % settings.train.checkpoint_interval == 0 and step > 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {step}: {save_path}")

    print("Training finished.")
    # Save final checkpoint
    save_path = manager.save()
    print(f"Saved final checkpoint: {save_path}")

    # TODO: Consider saving the models in SavedModel format as well for easier deployment
    # encoder.save(...)
    # predictor.save(...)


# --- Entry Point ---

if __name__ == "__main__":
    # Use tyro to parse command-line arguments into the Settings dataclass
    tyro.cli(run_training)