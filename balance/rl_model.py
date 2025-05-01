import tensorflow as tf
from tf_agents.networks import actor_distribution_network, network, value_network
from tf_agents.specs import is_continuous


def create_ppo_networks(
    observation_spec: tf.TensorSpec,
    action_spec: tf.TensorSpec,
    actor_fc_layers: tuple,
    value_fc_layers: tuple,
    activation_fn=tf.keras.activations.relu,
    dtype=tf.float32,
) -> tuple[network.Network, network.Network]:
    """
    Creates Actor and Value networks for PPO.

    Args:
        observation_spec: A tf.TensorSpec representing the observation space.
        action_spec: A tf.TensorSpec representing the action space.
        actor_fc_layers: Tuple of integers representing hidden layers for the actor network.
        value_fc_layers: Tuple of integers representing hidden layers for the value network.
        activation_fn: Activation function for hidden layers.
        dtype: Data type for network parameters.

    Returns:
        A tuple containing the (actor_network, value_network).
    """
    if not is_continuous(action_spec):
        raise ValueError("This network setup assumes a continuous action space.")

    # --- Actor Network ---
    # Outputs parameters for a distribution (e.g., MultivariateNormalDiag)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        activation_fn=activation_fn,
        dtype=dtype,
        # For continuous actions, TF-Agents defaults to NormalProjectionNetwork
        # which creates a MultivariateNormalDiag distribution.
        # You can customize projection_network_ctor if needed.
    )

    # --- Value Network ---
    # Outputs a single value estimate for the given observation
    value_net = value_network.ValueNetwork(
        observation_spec,
        fc_layer_params=value_fc_layers,
        activation_fn=activation_fn,
        dtype=dtype,
    )

    return actor_net, value_net
