import numpy as np
import padasip as pa
from scipy.signal import stft, istft, get_window
from scipy.fft import rfft, irfft
from scipy.signal.windows import hann
import ctypes
from ctypes.util import find_library
from dtd_mod import FastDoubleTalkDetector
from collections import deque

class NLMSFilter:
    def __init__(self, filter_len=128, mu=0.1):
        self.filter_len = filter_len
        self.mu = mu
        self.filter = pa.filters.FilterNLMS(n=filter_len, mu=mu)
        self.ref_history = np.zeros(filter_len, dtype=np.float32)
        self._output = 0.0
        self._echo_estimate = 0.0

    def process(self, mic_signal: float, ref_signal: float) -> float:
        """
        Process one sample using padasip's NLMS filter.

        Args:
            mic_signal (float): Microphone signal with echo.
            ref_signal (float): Far-end reference signal.

        Returns:
            float: Echo-cancelled signal (residual error).
        """
        # Update reference history
        self.ref_history = np.roll(self.ref_history, 1)
        self.ref_history[0] = ref_signal

        # Predict echo
        y = self.filter.predict(self.ref_history)

        # Compute error (echo-cancelled output)
        e = mic_signal - y

        # Update weights
        self.filter.adapt(mic_signal, self.ref_history)

        # Store values
        self._output = e
        self._echo_estimate = y
        return e

    def reset(self):
        """Reset filter state."""
        self.filter = pa.filters.FilterNLMS(n=self.filter_len, mu=self.mu)
        self.ref_history.fill(0)
        self._output = 0.0
        self._echo_estimate = 0.0

    @property
    def output(self):
        """Last output sample."""
        return self._output

    @property
    def echo_estimate(self):
        """Last estimated echo."""
        return self._echo_estimate

    @property
    def weights(self):
        """Current filter weights."""
        return self.filter.w.copy()

class KalmanAECSpectral:
    def __init__(self, frame_len=256, fft_len=None, overlap=0.5):
        self.frame_len = frame_len
        self.fft_len = fft_len or frame_len
        self.overlap = overlap
        self.hop_len = int(frame_len * (1 - overlap))
        self.freq_bins = self.fft_len // 2 + 1

        # Kalman parameters
        self.H = np.ones(self.freq_bins, dtype=np.complex64)
        self.P = np.ones(self.freq_bins) * 1e-2
        self.Q = 1e-4
        self.R = 1e-2

        # Hann window
        self.window = hann(self.frame_len, sym=False)

        # Overlap-add buffer
        self.output_tail = np.zeros(self.frame_len - self.hop_len, dtype=np.float32)

        self.pending_output = deque()

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        """Process one frame using frequency-domain Kalman AEC with windowing and overlap-add."""

        # 1. Windowing
        mic_win = mic_frame * self.window
        ref_win = ref_frame * self.window

        # 2. FFT
        mic_fft = rfft(mic_win, n=self.fft_len)
        ref_fft = rfft(ref_win, n=self.fft_len)

        # 3. Kalman filter per bin
        err_fft = np.zeros_like(mic_fft, dtype=np.complex64)

        for k in range(self.freq_bins):
            x = ref_fft[k]
            y = mic_fft[k]

            y_hat = self.H[k] * x
            e = y - y_hat

            Px = self.P[k] * np.conj(x)
            denom = np.real(x * Px) + self.R
            if denom < 1e-8:
                K = 0.0
            else:
                K = Px / denom

            self.H[k] += K * e
            self.P[k] = (1 - K * x) * self.P[k] + self.Q

            err_fft[k] = e

        # 4. IFFT
        err_time = irfft(err_fft, n=self.fft_len)
        err_time = np.real(err_time[:self.frame_len])

        # 5. Overlap-add output
        full_output = np.zeros(self.frame_len, dtype=np.float32)
        full_output[:self.output_tail.shape[0]] += self.output_tail
        full_output += err_time

        # 6. Save tail for next overlap-add
        self.output_tail = err_time[self.hop_len:]

        # 7. Return valid hop_len segment
        return full_output[:self.hop_len].astype(np.float32)
    
    
    def feed_frame(self, mic_frame, ref_frame) -> np.ndarray:
        """
        每次喂入 frame_len 数据，返回 frame_len 点输出；
        其中只有 hop_len 是真实的，其他点可能是 0（补偿）
        """
        out_hop = self.process(mic_frame, ref_frame)
        self.pending_output.extend(out_hop)

        # 每次补足 frame_len 长度输出
        if len(self.pending_output) >= self.frame_len:
            out = [self.pending_output.popleft() for _ in range(self.frame_len)]
            return np.array(out, dtype=np.float32)
        else:
            # 前期填 0
            return np.zeros(self.frame_len, dtype=np.float32)


class KalmanAECTime:
    def __init__(self, filter_len=128):
        self.filter_len = filter_len
        self.weights = np.zeros(filter_len, dtype=np.float32)
        self.P = np.eye(filter_len, dtype=np.float32) * 1e-2  # 初始化协方差矩阵
        self.Q = np.eye(filter_len, dtype=np.float32) * 1e-5  # 过程噪声
        self.R = 1e-1  # 观测噪声
        self.x_buf = np.zeros(filter_len, dtype=np.float32)  # 输入历史

    def process(self, mic_sample, ref_sample):
        # 滑动 ref 信号进入 x_buf
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = ref_sample

        x = self.x_buf.copy()
        x = x.reshape(-1, 1)  # 列向量

        # 预测
        echo_hat = np.dot(self.weights, self.x_buf)
        e = mic_sample - echo_hat  # 误差

        # Kalman 增益
        Px = self.P @ x
        K = Px / (x.T @ Px + self.R)

        # 更新权重
        self.weights += (K.flatten() * e)
        self.P = (np.eye(self.filter_len) - K @ x.T) @ self.P + self.Q

        return e, echo_hat, self.weights.copy()


class SpeexAEC:
    def __init__(self, frame_size=128, filter_len=1024, sample_rate=16000):
        self.frame_size = frame_size
        self.filter_len = filter_len

        lib_path = find_library('speexdsp')
        if not lib_path:
            raise RuntimeError("Could not find speexdsp library")

        self.lib = ctypes.CDLL(lib_path)

        # Create echo canceller
        self.state = self.lib.speex_echo_state_init(frame_size, filter_len)
        if not self.state:
            raise RuntimeError("Failed to create speex echo canceller")

        # Set sampling rate
        self.lib.speex_echo_ctl(self.state, 24, ctypes.byref(ctypes.c_int(sample_rate)))  # SPEEX_ECHO_SET_SAMPLING_RATE

        # Input buffers
        self.in_buf = np.zeros((frame_size,), dtype=np.int16)
        self.ref_buf = np.zeros((frame_size,), dtype=np.int16)
        self.out_buf = np.zeros((frame_size,), dtype=np.int16)

    def process(self, mic_sample, ref_sample):
        """单点处理，聚集成一帧后调用 Speex 逐帧处理"""
        # Shift buffer and add new sample
        self.in_buf = np.roll(self.in_buf, -1)
        self.in_buf[-1] = mic_sample

        self.ref_buf = np.roll(self.ref_buf, -1)
        self.ref_buf[-1] = ref_sample

        # Once buffer is full, process it
        if not np.any(self.out_buf):  # means we haven't processed yet
            self.lib.speex_echo_cancellation(
                self.state,
                self.in_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                self.ref_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                self.out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
            )
            self.out_index = 0

        result = self.out_buf[self.out_index]
        self.out_index += 1
        if self.out_index >= self.frame_size:
            self.out_buf[:] = 0  # reset output buffer
            self.out_index = 0

        return result / 32768.0  # normalize to float32

    def reset(self):
        self.lib.speex_echo_state_reset(self.state)

    def __del__(self):
        if self.state:
            self.lib.speex_echo_state_destroy(self.state)


class AECEngine:
    def __init__(self, mode: str = "nlms", filter_len: int = 128, mu: float = 0.1):
        assert mode in ["nlms", "kalman-t", "kalman-f", "speex"]
        self.mode = mode
        self.filter_len = filter_len
        self.mu = mu
        self._output = None
        self._echo_estimate = None
        self._weights = None
        self.queue = deque(maxlen=1024)
        
        if filter_len <= 0:
            print(f"Warning: filter_len {filter_len} must be positive, using default.")
            if mode == "nlms":
                filter_len = 128
            elif mode == "kalman-t":
                filter_len = 128
            elif mode == "kalman-f":
                filter_len = 256
            elif mode == "speex":
                filter_len = 128  

        self.filter_len = filter_len
        if self.mode == "nlms":
            self.filter = NLMSFilter(filter_len=filter_len, mu=mu)
        elif self.mode == "kalman-t":
            self.filter = KalmanAECTime(filter_len=filter_len)
        elif self.mode == "kalman-f":
            self.filter = KalmanAECSpectral(frame_len=filter_len)
        elif self.mode == "speex":
            self.filter = SpeexAEC(frame_size=filter_len)
        else:
            raise NotImplementedError(f"AEC mode '{self.mode}' not implemented.")
            
        print(f"Initialized AECEngine with mode: {self.mode}, filter_len: {filter_len}, mu: {mu}")

    def process(self, mic_signal: np.ndarray, ref_signal: np.ndarray) -> np.ndarray:
        assert len(mic_signal) == len(ref_signal)
        output = np.zeros_like(mic_signal)

        if self.mode == "nlms":
            for i in range(len(mic_signal)):
                mic = mic_signal[i]
                ref = ref_signal[i]
                err = self.filter.process(mic, ref)
                output[i] = err

            self._output = output
            self._echo_estimate = mic_signal - output
            self._weights = self.filter.weights
            return output

        elif self.mode == "kalman-t":
            echo = np.zeros_like(mic_signal)
            w = self.filter.weights  # default in case all frames are double-talk
            for i in range(len(mic_signal)):
                mic = mic_signal[i]
                ref = ref_signal[i]
                e, y, w = self.filter.process(mic, ref)
                output[i] = e
            self._output = output
            self._echo_estimate = echo
            self._weights = w
            return output

        elif self.mode == "kalman-f":
            output = np.zeros_like(mic_signal)
            output_list = []
            for i in range(0, len(mic_signal), self.filter_len):
                mic_frame = mic_signal[i:i+self.filter_len]
                ref_frame = ref_signal[i:i+self.filter_len]

                out_frame = self.filter.feed_frame(mic_frame, ref_frame)
                output_list.append(out_frame)  


            output = np.concatenate(output_list)
            self._output = output
            self._echo_estimate = mic_signal - output
            return output

        elif self.mode == "speex":
            for i in range(len(mic_signal)):
                mic = mic_signal[i]
                ref = ref_signal[i]
                e = self.filter.process(mic, ref)
                output[i] = e
              
            self._output = output
            self._echo_estimate = mic_signal - output
            return output
        else:
            raise NotImplementedError(f"AEC mode '{self.mode}' not implemented.")

if __name__ == "__main__":
    # Example usage
    engine = AECEngine(mode="kalman", filter_len=128, mu=0.1)
    file_path = "./data/250718_bot_room1_48k_src_all.wav"
    mic_signal, ref_signal = engine.read_wave_file(file_path, mic_chn=0)
    output = engine.process(mic_signal, ref_signal)