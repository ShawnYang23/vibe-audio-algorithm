
from scipy.io import wavfile
import numpy as np
import os

def read_wave_file(file_path: str, mic_chn) -> np.ndarray:
        """ Reads a wave file and extracts microphone and loopback data.
        Only supports 2D data 2/8 channels int16 data formats.
        Only supports > 16000 Hz sample rate file.
        Processing sample rate is 16000 Hz.
        # Assumed 2D: 0 microphone channels, 1 for loopback
        # Assummed 8D: 0-5 microphone channels, 7 for loopback, 6 is invalid
        Args:
            file_path (str): Path to the wave file.
            mic_chn (int): Microphone channel to extract, -1 for average of all channels.
        Returns:
            tuple: (mic_data, lp_data) where mic_data is the microphone signal and lp_data is the loopback signal.
        """
        sample_rate, data = wavfile.read(file_path)
        if sample_rate < 16000:
            print(f"Warning: file {file_path} sample rate {sample_rate} is less than 16000 Hz, returning None.")
            return None, None
        #downsample to 16000 Hz if needed
        if sample_rate > 16000:
            factor = sample_rate // 16000
            data = data[::factor]
            sample_rate = 16000
            
        print(f"mic_chn: {mic_chn}, data shape: {data.shape}， data ndim: {data.ndim}, data dtype: {data.dtype}")
        mic_data = None
        lp_data = None
        if data.ndim != 2:
            print(f"Warning: unsupported data format {data.ndim}D, returning None.")
            return None, None
        
        if data.shape[1] == 2:
            mic_data = data[:, 0]
            lp_data = data[:, 1] 
            
        elif data.shape[1] == 8:
            if data.ndim < mic_chn + 1:
                print(f"Warning: file {file_path} channel {mic_chn} does not exist, returning None.")
                return None, None
            else:
                if mic_chn >= 0:
                    mic_data = data[:, mic_chn]
                else:
                    # average all channels
                    mic_data = np.mean(data[0:5], axis=1)
            lp_data = data[:, 7] 
        else:
            print(f"Warning: unsupported data format {data.ndim}D, returning None.")
            return None, None
        
        if data.dtype == np.int16:
            mic_data = mic_data.astype(np.float32) / 32768.0
            lp_data = lp_data.astype(np.float32) / 32768.0
        elif data.dtype == np.int24:
            mic_data = mic_data.astype(np.float32) / 2147483648.0
            lp_data = lp_data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.int32:
            mic_data = mic_data.astype(np.float32) / 2147483648.0
            lp_data = lp_data.astype(np.float32) / 2147483648.0
        else:
            print(f"Warning: unsupported data type {data.dtype}, returning None.")
            return None, None
        return mic_data, lp_data
    
def write_wave_file(file_path: str, data: np.ndarray, channels: int = 1, sample_rate: int = 16000):
    """ Writes a numpy array to a wave file.
    Only supports int16 and float32 data formats.
    Args:
        file_path (str): Path to the output wave file.
        data (np.ndarray): Data to write, should be 1D or 2D.
        channels (int): Number of channels, default is 1.
        sample_rate (int): Sample rate, default is 16000.
    """
    if data.ndim == 1:
        data = data.reshape(-1, channels)
    elif data.shape[1] != channels:
        raise ValueError("Data must be 1D or 2D with the correct number of channels.")
    
    if data.dtype == np.float32:
       data = (data * 32768.0).astype(np.int16)
    elif data.dtype == np.int16:
       pass
    else:
        raise ValueError("Data must be either int16 or float32.")
    
    if not file_path.endswith('.wav'):
        file_path += '.wav'
        
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        
    wavfile.write(file_path, sample_rate, data)

    print(f"Wave file written to {file_path}")
    return file_path