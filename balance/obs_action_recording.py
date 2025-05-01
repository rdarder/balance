from abc import ABCMeta, abstractmethod
from pathlib import Path
from datetime import datetime

import numpy as np
from tf_agents.trajectories import TimeStep

class EpisodeRecorder(metaclass=ABCMeta):
    @abstractmethod
    def record_step(self, time_step: TimeStep, previous_action: np.ndarray):
        pass

    @abstractmethod
    def finalize(self):
        """Prints the final recording statistics if recording was enabled."""
        pass

class NullEpisodeRecoder(EpisodeRecorder):
    def record_step(self, time_step, previous_action):
        pass
    def finalize(self):
        pass


class FileEpisodeRecorder(EpisodeRecorder):
    """Manages the overall recording process across multiple episodes."""

    def __init__(self,
                 output_path: Path,
                 min_episode_length: int):
        """
        Initializes the RecordingManager.

        Args:
            output_path: The base directory to save episode files.
            min_episode_length: Minimum steps for an episode to be saved.
        """
        self._output_path = output_path
        self._min_length = min_episode_length

        # Aggregate statistics
        self._discarded_count = 0
        self._collected = []
        self._buffer = []

        # Create directory once at the start
        self._output_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording enabled. Output directory: {self._output_path.resolve()}")

    def record_step(self, time_step: TimeStep, previous_action: np.ndarray):
        """Records a single step using the active recorder, if any."""
        combined_input = np.concatenate([time_step.observation, previous_action]).astype(np.float32)
        self._buffer.append(combined_input)
        if time_step.is_last():
            self._save_and_reset()

    def finalize(self):
        """Prints the final recording statistics if recording was enabled."""

        self._save_all_episodes()

        total_steps = sum(len(c) for c in self._collected)
        print("\n--- Recording Summary ---")
        print(f"Total episodes run: {len(self._collected) + self._discarded_count}")
        print(f"Episodes saved:     {len(self._collected)}")
        print(f"Episodes discarded: {self._discarded_count}")
        print(f"Total steps saved:  {total_steps}")
        self._collected = []


    def _save_all_episodes(self)-> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulated_data_{timestamp}.npz"
        filepath = self._output_path / filename
        np.savez_compressed(filepath, *self._collected)

    def _save_and_reset(self) -> None:
        episode_length = len(self._buffer)
        if episode_length >= self._min_length:
            episode_array = np.array(self._buffer)
            self._collected.append(episode_array)
            self._buffer = []
        else:
            self._discarded_count += 1

