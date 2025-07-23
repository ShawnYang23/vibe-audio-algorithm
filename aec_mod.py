import numpy as np
import padasip as pa
from filterpy.kalman import KalmanFilter
from scipy.signal import stft, istft, get_window
import ctypes
import numpy as np
from numpy.fft import rfft, irfft
from ctypes.util import find_library
import webrtc_audio_processing

class KalmanAECSpectral:
    def __init__(self, frame_len=256, fft_len=None):
        self.frame_len = frame_len
        self.fft_len = fft_len or frame_len
        self.freq_bins = self.fft_len // 2 + 1

        # 初始化频域 Kalman 参数（逐频 bin）
        self.H = np.ones(self.freq_bins, dtype=np.complex64)   # 增益
        self.P = np.ones(self.freq_bins) * 1e-2                # 协方差
        self.Q = 1e-5                                           # 过程噪声
        self.R = 1e-1                                           # 观测噪声

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        """逐帧处理 AEC, 频域 Kalman"""

        # 1. FFT
        mic_fft = rfft(mic_frame, n=self.fft_len)
        ref_fft = rfft(ref_frame, n=self.fft_len)

        # 2. 初始化输出误差频域
        err_fft = np.zeros_like(mic_fft)

        # 3. 对每个频率 bin 做 Kalman 滤波
        for k in range(self.freq_bins):
            x = ref_fft[k]
            y = mic_fft[k]

            # 预测
            y_hat = self.H[k] * x
            e = y - y_hat

            # Kalman 增益
            Px = self.P[k] * np.conj(x)
            denom = (x * Px + self.R)
            K = Px / denom if denom != 0 else 0.0

            # 更新
            self.H[k] += K * e
            self.P[k] = (1 - K * x) * self.P[k] + self.Q

            # 输出误差信号
            err_fft[k] = e

        # 4. IFFT 得到时域误差
        err_time = irfft(err_fft, n=self.fft_len)[:self.frame_len]
        return np.real(err_time).astype(np.float32)


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
class WebRTCAECTime:
    def __init__(self, sample_rate=16000, frame_size=160):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._aec = webrtc_audio_processing.AudioProcessing(
            enable_aec=True,
            enable_agc=False,
            enable_ns=False,
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._mic_buffer = []
        self._ref_buffer = []

    def process(self, mic_sample, ref_sample):
        """逐点缓冲，按 frame_size 帧处理，输出单点数据"""
        self._mic_buffer.append(mic_sample)
        self._ref_buffer.append(ref_sample)

        if len(self._mic_buffer) >= self.frame_size:
            mic_frame = np.array(self._mic_buffer[-self.frame_size:], dtype=np.float32).reshape(1, -1)
            ref_frame = np.array(self._ref_buffer[-self.frame_size:], dtype=np.float32).reshape(1, -1)

            out_frame = self._aec.process_stream(mic_frame, ref_frame)
            out_sample = out_frame[0, -1]  # 取最后一帧最后一点输出（近似逐点）
            return out_sample
        else:
            return mic_sample  # 无法处理时，透传

class AECEngine:
    def __init__(self, mode: str = "nlms", filter_len: int = 128, mu: float = 0.1):
        assert mode in ["nlms", "kalman-t", "kalman-f", "speex"]
        self.mode = mode
        self.filter_len = filter_len
        self.mu = mu
        self._output = None
        self._echo_estimate = None
        self._weights = None
        
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
            elif mode == "webrtc":
                filter_len = 160
        self.filter_len = filter_len
        if self.mode == "nlms":
            self.filter = pa.filters.FilterNLMS(n=filter_len, mu=mu)
            self.ref_history = np.zeros(filter_len)
        elif self.mode == "kalman-t":
            self.filter = KalmanAECTime(filter_len=filter_len)
        elif self.mode == "kalman-f":
            self.filter = KalmanAECSpectral(frame_len=filter_len)
        elif self.mode == "speex":
            self.filter = SpeexAEC(frame_size=filter_len)
        elif self.mode == "webrtc":
            self.filter = WebRTCAECTime(sample_rate=16000, frame_size=filter_len)
            
        print(f"Initialized AECEngine with mode: {self.mode}, filter_len: {filter_len}, mu: {mu}")

    def process(self, mic_signal: np.ndarray, ref_signal: np.ndarray) -> np.ndarray:
        """逐点处理，每次返回与 mic_signal 等长的误差信号"""
        assert len(mic_signal) == len(ref_signal)
        output = np.zeros_like(mic_signal)

        if self.mode == "nlms":
            for i in range(len(mic_signal)):
                # 滑动更新参考信号历史
                self.ref_history = np.roll(self.ref_history, -1)
                self.ref_history[-1] = ref_signal[i]

                if i < self.filter_len - 1:
                    output[i] = mic_signal[i]  # 暂时无法计算，直接输出原始
                else:
                    x = self.ref_history.copy()
                    y = np.dot(self.filter.w, x)
                    e = mic_signal[i] - y
                    norm = np.dot(x, x) + 1e-8
                    self.filter.w += 2 * self.mu * e * x / norm
                    output[i] = e

            self._output = output
            self._echo_estimate = mic_signal - output
            self._weights = self.filter.w.copy()
            return output

        elif self.mode == "kalman-t":
            output = np.zeros_like(mic_signal)
            echo = np.zeros_like(mic_signal)
            for i in range(len(mic_signal)):
                e, y, w = self.filter.process(mic_signal[i], ref_signal[i])
                output[i] = e
                echo[i] = y
            self._output = output
            self._echo_estimate = echo
            self._weights = w
            return output
        elif self.mode == "kalman-f":
            output = np.zeros_like(mic_signal)
            for i in range(0, len(mic_signal), self.filter_len):
                mic_frame = mic_signal[i:i + self.filter_len]
                ref_frame = ref_signal[i:i + self.filter_len]

                if len(mic_frame) < self.filter_len or len(ref_frame) < self.filter_len:
                    continue

                e = self.filter.process(mic_frame, ref_frame)
                output[i:i + self.filter_len] = e

            self._output = output
            self._echo_estimate = mic_signal - output
            return output
        elif self.mode == "speex":
            output = np.zeros_like(mic_signal)
            for i in range(len(mic_signal)):
                output[i] = self.filter.process(mic_signal[i], ref_signal[i])
            self._output = output
            self._echo_estimate = mic_signal - output
            return output
        elif self.mode == "webrtc":
            output = np.zeros_like(mic_signal)
            for i in range(len(mic_signal)):
                output[i] = self.filter.process(mic_signal[i], ref_signal[i])
            self._output = output
            self._echo_estimate = mic_signal - output
            return output

if __name__ == "__main__":
    # Example usage
    engine = AECEngine(mode="kalman", filter_len=128, mu=0.1)
    file_path = "/home/shawn/AudioData/Room-Bot-Audio/250718_bot_room1_48k_src_all.wav"
    mic_signal, ref_signal = engine.read_wave_file(file_path, mic_chn=0)
    output = engine.process(mic_signal, ref_signal)