import tensorflow as tf

class TrainingEncoderNetwork(tf.keras.Model):
    """
    Encoder RNN designed for training the world model.
    Processes sequences and returns the full sequence of hidden states,
    suitable for multi-step prediction loss calculation.
    """
    def __init__(self, latent_dim: int, name="TrainingEncoderNetwork", **kwargs):
        """
        Args:
            latent_dim: The dimensionality of the GRU's hidden/output state.
            name: Model name.
        """
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim

        # GRU Layer configured for training:
        # - return_sequences: True to get hidden state for each step.
        # - return_state: True to also get the final state explicitly (useful for consistency).
        self.gru_layer = tf.keras.layers.GRU(
            units=self.latent_dim,
            return_sequences=True,
            return_state=True,
            name="encoder_gru" # Keep the layer name consistent for weight loading
        )
        print(f"EncoderRNN_Training initialized with latent_dim={latent_dim}")

    def call(self, inputs, initial_state=None, training=None):
        """
        Processes the input sequence.

        Args:
            inputs: A tensor of shape (batch_size, sequence_length, input_features)
                    containing the sequence of (observation, previous_action) pairs.
            initial_state: Optional initial hidden state for the GRU.
                           Shape (batch_size, latent_dim).
            training: Boolean indicating training mode.

        Returns:
            sequence_of_states: The hidden state for each time step.
                                Shape (batch_size, sequence_length, latent_dim).
            final_state: The final hidden state after processing the sequence.
                         Shape (batch_size, latent_dim).
        """
        sequence_of_states, final_state = self.gru_layer(
            inputs,
            initial_state=initial_state,
            training=training
        )
        return sequence_of_states, final_state

    def build(self, input_shape):
         # input_shape expected: (batch_size, sequence_length, input_features)
         self.gru_layer.build(input_shape)
         self.built = True
         print(f"EncoderRNN_Training built with input shape: {input_shape}")

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class InferenceEncoderNetwork(tf.keras.Model):
    """
    Encoder RNN designed for step-by-step inference (e.g., in simulation).
    Uses a stateful GRU to maintain the hidden state internally across calls.
    Processes one time step at a time.
    Requires a fixed batch_size specified during initialization.
    """
    def __init__(self, latent_dim: int, batch_size: int, name="InferenceEncoderNetwork", **kwargs):
        """
        Args:
            latent_dim: The dimensionality of the GRU's hidden state.
                        Must match the trained EncoderRNN_Training model.
            batch_size: The fixed batch size this model will be used with during inference.
            name: Model name.
        """
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.batch_size = batch_size # Store batch size

        # GRU Layer configured for stateful inference:
        # - stateful: True to maintain state between calls.
        # - return_sequences: False, as we process one step and need one output state.
        # - batch_input_shape: Required for stateful RNNs. Specifies (batch, steps, features).
        #                      Here, steps=1 because we feed one step at a time.
        # Note: The internal weights (kernels, biases) are compatible with the
        #       non-stateful GRU layer in EncoderRNN_Training if latent_dim matches.
        self.gru_layer = tf.keras.layers.GRU(
            units=self.latent_dim,
            return_sequences=False, # Output is the state for the single step
            return_state=False,    # State is managed internally, output *is* the state
            stateful=True,
            # Specify batch_input_shape: (batch_size, timesteps, features)
            # Features dimension will be determined when build is called or from input.
            # We set timesteps=1 as we process one step at a time.
            # batch_input_shape=(self.batch_size, 1, input_features) # Let build handle features dim
            name="encoder_gru" # Keep the layer name consistent for weight loading
        )
        print(f"EncoderRNN_Inference initialized with latent_dim={latent_dim}, batch_size={batch_size}")

    def call(self, inputs, training=None):
        """
        Processes a single time step input.

        Args:
            inputs: A tensor of shape (batch_size, 1, input_features)
                    containing the (observation, previous_action) pair for the current step.
                    The batch_size must match the one specified during initialization.
            training: Boolean indicating training mode (usually False for inference).

        Returns:
            current_state: The hidden state after processing the input step.
                           Shape (batch_size, latent_dim).
        """
        # The stateful GRU updates its internal state and returns the output for this step.
        current_state = self.gru_layer(inputs, training=training)
        return current_state

    def build(self, input_shape):
         # input_shape expected: (batch_size, 1, input_features)
         # We need to ensure the GRU layer is built with the correct batch_input_shape
         # for stateful operation.
         if input_shape[0] is not None and input_shape[0] != self.batch_size:
              raise ValueError(f"Input batch size ({input_shape[0]}) does not match "
                               f"model's expected batch size ({self.batch_size})")
         if input_shape[1] != 1:
              raise ValueError(f"Input sequence length ({input_shape[1]}) must be 1 for "
                               f"step-by-step inference.")

         # Construct the explicit batch_input_shape for the GRU layer
         batch_input_shape = (self.batch_size, 1, input_shape[2])
         self.gru_layer.build(batch_input_shape)
         self.built = True
         print(f"EncoderRNN_Inference built with batch_input_shape: {batch_input_shape}")

    def reset_states(self):
        """Resets the internal state of the stateful GRU layer."""
        self.gru_layer.reset_states()
        print("EncoderRNN_Inference states reset.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "batch_size": self.batch_size # Include batch_size in config
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PredictorMLP(tf.keras.Model):
    """
    Predicts the next latent state given the current latent state and action.
    Uses a simple MLP architecture.
    (This class remains unchanged as its function is independent of training/inference mode)
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 32, name="PredictorMLP", **kwargs):
        """
        Args:
            latent_dim: Dimensionality of the latent state (input and output).
            action_dim: Dimensionality of the action vector.
            hidden_dim: Number of units in the hidden layer.
            name: Model name.
        """
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.hidden_layer = tf.keras.layers.Dense(
            units=self.hidden_dim,
            activation='relu',
            name="predictor_hidden"
        )
        self.output_layer = tf.keras.layers.Dense(
            units=self.latent_dim,
            activation=None,
            name="predictor_output"
        )
        print(f"PredictorMLP initialized with latent_dim={latent_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")

    def call(self, latent_state, action, training=None):
        """
        Predicts the next latent state.

        Args:
            latent_state: The current latent state tensor. Shape (batch_size, latent_dim).
            action: The action taken at the current step. Shape (batch_size, action_dim).
            training: Boolean indicating training mode.

        Returns:
            predicted_next_latent: The predicted latent state for the next step.
                                   Shape (batch_size, latent_dim).
        """
        combined_input = tf.concat([latent_state, action], axis=-1)
        hidden_output = self.hidden_layer(combined_input, training=training)
        predicted_next_latent = self.output_layer(hidden_output, training=training)
        return predicted_next_latent

    def build(self, input_shape):
        latent_state_shape, action_shape = input_shape
        combined_dim = latent_state_shape[-1] + action_shape[-1]
        self.hidden_layer.build((None, combined_dim))
        self.output_layer.build((None, self.hidden_dim))
        self.built = True
        print(f"PredictorMLP built with input shapes: {input_shape}")

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
