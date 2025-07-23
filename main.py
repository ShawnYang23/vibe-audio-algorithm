
from aec_mod import AECEngine
from vad_mod import SileroVAD
import utils_mod as utils
import numpy as np

lenth = -1 # default filter length
AEC = AECEngine(mode="kalman-f", filter_len=lenth, mu=0.1)
VAD = SileroVAD(sampling_rate=16000, threshold=0.3, window_size=512)

in_path = "./data/250718_bot_room1_48k_src_all.wav"
mic_signal, ref_signal = utils.read_wave_file(in_path, mic_chn=0)

# Ensure float32 processing
mic_signal = mic_signal.astype(np.float32)
ref_signal = ref_signal.astype(np.float32)

# VAD + AEC
vad_segments = VAD.vad_segments(ref_signal)
vad_data = np.zeros_like(mic_signal)
output = mic_signal
frame_len = AEC.filter_len 

for segment in vad_segments:
    start = int(segment['start'])
    end = int(segment['end'])

    if end - start < frame_len:
        continue

    if not VAD.is_speech_segment(mic_signal[start:end], min_voiced_frames=1):
        continue
    
    print(f"Processing segment from {start} to {end}, length: {end - start}")
    vad_data[start:end] = 0.9  # Mark VAD segment
    output[start:end] = 0  # Reset output segment
    for i in range(start, end - frame_len + 1, frame_len):
        mic_frame = mic_signal[i:i + frame_len]
        ref_frame = ref_signal[i:i + frame_len]

        if len(mic_frame) < frame_len or len(ref_frame) < frame_len:
            continue
        
        out_frame = AEC.process(mic_frame, ref_frame)
        print("out_frame shape:", out_frame.shape)
        out_frame = out_frame.reshape(frame_len)

        if out_frame is None or len(out_frame) != frame_len:
            print(f"Warning: AEC output frame length mismatch at {i}, expected {frame_len}, got {len(out_frame)}")
            continue

        print(f"Processing frame {i // frame_len + 1} in segment, length: {len(mic_frame)}")
        output[i:i + frame_len] = out_frame

# clip to prevent overflow
# output = np.clip(output, -1.0, 1.0)

# 保存输出
out_path = "/mnt/c/Users/jmysy/Music/Room-Bot-Audio/tmp/out.wav"
cov_path = "/mnt/c/Users/jmysy/Music/Room-Bot-Audio/tmp/cov.wav"
vad_path = "/mnt/c/Users/jmysy/Music/Room-Bot-Audio/tmp/vad.wav"
utils.write_wave_file(vad_path, vad_data, channels=1, sample_rate=16000)
utils.write_wave_file(cov_path, ref_signal, channels=1, sample_rate=16000)
utils.write_wave_file(out_path, output, channels=1, sample_rate=16000)

