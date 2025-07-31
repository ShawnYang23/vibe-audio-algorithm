
from aec_mod import AECEngine
from vad_mod import SileroVAD
import utils_mod as utils
import numpy as np
from dtd_mod import FastDoubleTalkDetector
import matplotlib.pyplot as plt

output_path = "/mnt/c/Users/jmysy/Music/Room-Bot-Audio/tmp/"
input_path = "./data/250718_bot_room6_48k_src_all.wav"

lenth = -1 # default filter length
AEC = AECEngine(mode="nlms", filter_len=lenth, mu=0.45) #nlms，kalman-t, kalman-f, speex
VAD = SileroVAD(sampling_rate=16000, threshold=0.3, window_size=512)
DTD = FastDoubleTalkDetector(detect_lenth=2048, method="shape")
mic_supression_factor = 0.01  # Factor to suppress mic signal

mic_signal, ref_signal = utils.read_wave_file(input_path, mic_chn=0)

# Ensure float32 processing
mic_signal = mic_signal.astype(np.float32)
ref_signal = ref_signal.astype(np.float32)

# Save original signals for reference
mic_path = f"{output_path}" + "mic.wav"
ref_path = f"{output_path}" + "ref.wav"
utils.write_wave_file(mic_path, mic_signal, channels=1, sample_rate=16000)
utils.write_wave_file(ref_path, ref_signal, channels=1, sample_rate=16000)

# VAD + AEC
vad_segments = VAD.vad_segments(ref_signal)
vad_data = np.zeros_like(mic_signal)
output = mic_signal.copy()  # Start with mic signal as output
output = output.astype(np.float32)  # Ensure float32 for processing
frame_len = AEC.filter_len 
ERLE = []


for segment in vad_segments:
    start = int(segment['start'])
    end = int(segment['end'])

    if end - start < frame_len:
        continue

    if not VAD.is_speech_segment(mic_signal[start:end], min_voiced_frames=1):
        continue
 
    print(f"Processing segment from {start} to {end}, length: {end - start}")
    vad_data[start:end] = 0.6  # Mark VAD segment
    output[start:end] = 0  # Reset output segment
    # dtd_flag = DTD.is_double_talk_flag(mic_signal[start:end], ref_signal[start:end])
    
    for i in range(start, end - frame_len + 1, frame_len):
        mic_frame = mic_signal[i:i + frame_len] * mic_supression_factor
        ref_frame = ref_signal[i:i + frame_len]
        
        # normalize frames
        gain = np.std(mic_signal) / (np.std(ref_signal) + AEC.epsilon)
        ref_signal *= gain
        
        if len(mic_frame) < frame_len or len(ref_frame) < frame_len:
            continue
        
        # TODO: fast DTD detection
        # is_dtd = DTD.is_double_talk(mic_frame, ref_frame)
        # is_dtd = np.mean(dtd_flag[i-start:i-start+ frame_len -1]) > 0.6 
        # is_dtd = False
        # if is_dtd:
        #     print("Double talk detected, skipping AEC processing.")
        #     vad_data[i:i + frame_len] = 0.9  # Mark as double talk
        #     out_frame = mic_frame  # Skip AEC processing
        # else:
        
        # Process AEC
        out_frame = AEC.process(mic_frame, ref_frame)
        if out_frame is None:
            continue
        
        out_frame = out_frame.reshape(frame_len)
        
        # unti normalize output frame
        # out_frame = out_frame * np.std(mic_frame)  # Scale back to original
        # out_frame = np.clip(out_frame, -1.0, 1.0)

        if out_frame is None or len(out_frame) != frame_len:
            print(f"Warning: AEC output frame length mismatch at {i}, expected {frame_len}, got {len(out_frame)}")
            continue

        # print(f"Processing frame {i // frame_len + 1} in segment, length: {len(mic_frame)}")
        output[i:i + frame_len] = out_frame
    erle_segment = AEC.calculate_erle(mic_signal[start:end], output[start:end])
    ERLE.append(erle_segment)
# clip to prevent overflow
# output = np.clip(output, -1.0, 1.0)
print(f"Total segments processed: {len(vad_segments)}")
erle_mean = np.mean(ERLE)
print(f"Mean ERLE for the entire signal: {erle_mean:.2f} dB")
# Save output and VAD data
out_path = f"{output_path}"+"" + f"{AEC.mode}.wav"
vad_path = f"{output_path}"+"vad.wav"
utils.write_wave_file(vad_path, vad_data, channels=1, sample_rate=16000)

utils.write_wave_file(out_path, output, channels=1, sample_rate=16000)

