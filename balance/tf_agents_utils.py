# --- Encapsulated Runner Function (Modified) ---
import functools
import sys
from typing import TypeVar, Type, Callable

import tyro
from absl import app
import tensorflow as tf

from tf_agents.policies import TFPolicy
from tf_agents.system.default import multiprocessing_core as mpc

T = TypeVar("T") # Generic type for settings dataclass

def run_with_tyro_and_tfagents_mp(
    settings_cls: Type[T],
    main_func: Callable[[T], None],
):
    """
    Parses arguments using tyro, then runs the main function, initializing
    TF-Agents multiprocessing context via handle_main.
    This is horrific and very likely doesn't work for all cases, but I didn't find
    another way. I regret moving over to
    tf-agents.

    Args:
        settings_cls: The dataclass type to parse arguments into (e.g., Settings).
        main_func: The main function to execute, which accepts an instance of
                   settings_cls.
    """
    settings = tyro.cli(settings_cls)
    original_argv = sys.argv
    clean_argv = [original_argv[0]]
    def absl_main_wrapper(argv_from_absl):
        # Call the user-provided main function with the captured 'settings'.
        main_func(settings)
    app_run_configured = functools.partial(app.run, absl_main_wrapper)
    try:
        sys.argv = clean_argv
        mpc.handle_main(app_run_configured)
    finally:
        sys.argv = original_argv


def print_policy_names(policy: TFPolicy):
    try:
        # Ensure variables are created if not already
        if not policy.variables():
            _ = policy.get_initial_state(batch_size=1) # Trigger variable creation
        for var in policy.variables():
            print(f"  Name: {var.name}, Shape: {var.shape}")
    except Exception as e:
        print(f"  Error inspecting tf_policy variables: {e}")


def inspect_checkpoint_variables(ckpt_dir: str):
    """Finds the latest checkpoint and prints the variable names within it."""
    print(f"Looking for latest checkpoint in: {ckpt_dir}")
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if not latest_ckpt:
        raise Exception(f"  ERROR: No checkpoint found in {ckpt_dir}")
    print(f"Found latest checkpoint prefix: {latest_ckpt}")
    print("\n--- Variables Found in Checkpoint File ---")

    variables_in_ckpt = tf.train.list_variables(latest_ckpt)
    if not variables_in_ckpt:
        print("  WARNING: No variables found in the checkpoint file!")
    else:
        for name, shape in variables_in_ckpt:
            print(f"  Name: {name}, Shape: {shape}")
