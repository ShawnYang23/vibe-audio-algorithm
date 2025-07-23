import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

class SileroVAD:
    def __init__(self, sampling_rate=16000, threshold=0.5, window_size=512):
        self.model = load_silero_vad()
        self.SAMPLE_RATE = sampling_rate
        self.threshold = threshold
        self.window_size = window_size

    def vad_segments(self, audio: np.ndarray):
        """Detects speech segments using Silero VAD.

        Args:
            audio (np.ndarray): Audio array, expected shape (N,)

        Returns:
            list of dicts with 'start' and 'end' keys (in samples)
        """
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # [1, N]

        timestamps = get_speech_timestamps(audio, self.model, sampling_rate=self.SAMPLE_RATE)
        return timestamps  # [{'start': int, 'end': int}, ...]

    def is_speech(self, frame: np.ndarray) -> bool:
        """Checks if a given audio frame contains speech."""
        if len(frame) < self.window_size:
            return False
        frame = frame.astype(np.float32)
        frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # shape: [1, N]
        with torch.no_grad():
            speech_prob = self.model(frame_tensor, self.SAMPLE_RATE)
        return speech_prob.item() > self.threshold


    def is_speech_segment(self, segment: np.ndarray, min_voiced_frames: int = 1) -> bool:
        """Check if segment contains enough speech frames."""
        count = 0
        segment = segment.astype(np.float32)
        for i in range(0, len(segment) - self.window_size + 1, self.window_size):
            frame = segment[i:i + self.window_size]
            if self.is_speech(frame):
                count += 1
                if count >= min_voiced_frames:
                    return True
        return False
