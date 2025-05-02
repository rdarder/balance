# /home/rdarder/dev/balance/balance/observation_processing.py
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from pathlib import Path
import warnings

# Assuming world model classes are here
from balance.world_model import InferenceEncoderNetwork, TrainingEncoderNetwork, PredictorMLP

class EncoderWrapper(py_environment.PyEnvironment):
    """
    A PyEnvironment wrapper that uses a pre-trained InferenceEncoderNetwork
    to transform observations before passing them to the agent.
    """

    def __init__(self,
                 environment: py_environment.PyEnvironment,
                 encoder_checkpoint_path: str,
                 latent_dim: int,
                 input_features: int):
        """
        Args:
            environment: The underlying PyEnvironment (e.g., SegwayEnv).
            encoder_checkpoint_path: Path to the world model checkpoint directory
                                     (containing encoder weights).
            latent_dim: The dimensionality of the encoder's latent space.
            input_features: The dimensionality of the raw input to the encoder
                            (obs_dim + action_dim).
        """
        super().__init__()
        self._environment = environment
        self._latent_dim = latent_dim
        self._input_features = input_features

        # --- Create and Load Inference Encoder ---
        # Batch size 1 for step-by-step processing in PyEnvironment
        self._encoder = InferenceEncoderNetwork(latent_dim=latent_dim, batch_size=1)

        # Build the encoder with expected input shape (batch=1, steps=1, features)
        try:
            self._encoder.build((1, 1, self._input_features))
        except ValueError as e:
            # Catch potential build errors early
            raise ValueError(f"Error building InferenceEncoderNetwork: {e}") from e

        # Load weights from the world model checkpoint
        print(f"Attempting to load encoder weights from: {encoder_checkpoint_path}")
        # Create a checkpoint object that ONLY expects the encoder part
        # The key 'encoder' must match the key used when saving the WM checkpoint
        # The layer name 'encoder_gru' inside the models must also match.
        ckpt = tf.train.Checkpoint(encoder=self._encoder)

        # Use restore().expect_partial() to only load matching weights (i.e., the encoder)
        # and ignore optimizer state, predictor weights, etc. from the WM checkpoint.
        try:
            status = ckpt.restore(encoder_checkpoint_path).expect_partial()
            # status.assert_existing_objects_matched() # Use if you want strict checking
            status.assert_nontrivial_match() # Check that *something* was loaded
            print("Encoder weights loaded successfully.")
        except tf.errors.NotFoundError:
             raise FileNotFoundError(
                 f"Checkpoint not found at {encoder_checkpoint_path}. "
                 "Ensure the path points to a valid TF checkpoint file (e.g., ckpt-X), not just the directory."
             )
        except Exception as e:
            raise RuntimeError(f"Error loading encoder weights: {e}") from e


        # --- Freeze Encoder Weights ---
        self._encoder.trainable = False
        print("Encoder weights frozen.")
        # Verify freezing (optional)
        # assert len(self._encoder.trainable_variables) == 0

        # --- Define the new observation spec ---
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._latent_dim,),
            dtype=np.float32,
            minimum=-np.inf, # Latent space bounds are unknown/unbounded
            maximum=np.inf,
            name='latent_observation'
        )

        # --- Internal state for previous action ---
        # Action spec is needed to know the shape/dtype of the previous action
        self._action_spec_shape = self._environment.action_spec().shape
        self._action_spec_dtype = self._environment.action_spec().dtype
        self._previous_action = np.zeros(self._action_spec_shape, dtype=self._action_spec_dtype)


    # --- Implement PyEnvironment abstract methods ---

    def observation_spec(self):
        return self._observation_spec # Return the new latent spec

    def action_spec(self):
        return self._environment.action_spec() # Action spec remains the same

    def _reset(self) -> ts.TimeStep:
        # 1. Reset the underlying environment
        time_step = self._environment.reset()

        # 2. Reset the stateful encoder
        self._encoder.reset_states()
        print("Encoder state reset.")

        # 3. Reset previous action (important!)
        self._previous_action = np.zeros(self._action_spec_shape, dtype=self._action_spec_dtype)

        # 4. Process the initial observation
        latent_observation = self._process_observation(time_step.observation)

        # 5. Return the first TimeStep with the latent observation
        return ts.restart(latent_observation) # Use ts.restart

    def _step(self, action) -> ts.TimeStep:
        # 1. Step the underlying environment with the action
        time_step = self._environment.step(action)

        # 2. Process the resulting observation using the *previous* action
        latent_observation = self._process_observation(time_step.observation)

        # 3. Store the current action for the *next* step's processing
        self._previous_action = action.astype(self._action_spec_dtype) # Ensure correct type

        # 4. Return the TimeStep with latent observation and original reward/discount
        # Need to check if the original timestep was terminal/truncated
        if time_step.is_last():
             # Use termination to signal end, discount is handled by ts.termination
             return ts.termination(latent_observation, time_step.reward)
        else:
             return ts.transition(latent_observation, time_step.reward, time_step.discount)


    def _process_observation(self, observation: np.ndarray) -> np.ndarray:
        """Combines observation with previous action and passes through encoder."""
        # Combine raw observation and previous action
        # NOTE: Assumes the raw observation contains only the features needed
        #       by the encoder (e.g., 6 IMU dims if desired speed/turn were excluded
        #       during WM training, or all 8 if they were included).
        #       Adjust slicing if necessary based on how WM was trained.
        #       The provided train_world_model.py used OBS_DIM=6.
        imu_observation = observation[:6]
        combined_input = np.concatenate(
            [imu_observation, self._previous_action]
        ).astype(np.float32)

        # Reshape for the stateful encoder (batch=1, steps=1, features)
        encoder_input = tf.reshape(combined_input, (1, 1, self._input_features))

        # Pass through the encoder
        latent_state = self._encoder(encoder_input) # training=False is default

        # Remove batch/time dimensions -> (latent_dim,)
        latent_state_np = tf.squeeze(latent_state, axis=0).numpy()

        return latent_state_np

    # --- Forward other methods if needed ---
    def __getattr__(self, name):
        """Forward attribute access to the underlying environment."""
        return getattr(self._environment, name)

    def close(self):
        return self._environment.close()

    # get_info, get_state, set_state could be forwarded or adapted if necessary
    def get_info(self):
         # Forward info from underlying env
         return self._environment.get_info()

    # get_state/set_state are tricky with the wrapper's internal state (encoder state, prev_action)
    # Best to disable or implement carefully if needed for checkpointing the *entire* RL state.
    def get_state(self):
        raise NotImplementedError("get_state not implemented for EncoderWrapper due to internal state complexity.")

    def set_state(self, state):
        raise NotImplementedError("set_state not implemented for EncoderWrapper.")


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
