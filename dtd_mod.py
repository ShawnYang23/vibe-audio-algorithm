import numpy as np

class FastDoubleTalkDetector:
    class EnergyRatioDTD:
        def __init__(self, threshold=0.6):
            self.threshold = threshold  # 近端/远端 能量比阈值

        def is_double_talk(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> bool:
            err = mic_frame - ref_frame
            mic_energy = np.mean(mic_frame ** 2) + 1e-8
            ref_energy = np.mean(ref_frame ** 2) + 1e-8
            if ref_energy <= 1e-8 or mic_energy <= 1e-8:
                return False  
            ratio = mic_energy / ref_energy
            if mic_energy > 1e-8 and ratio > self.threshold:
                print(f"Energy ratio: {ratio:.7f}, threshold: {self.threshold}")

            return ratio > self.threshold

    class EnvelopeMismatchDTD:
        def __init__(self, corr_th=0.8):
            self.corr_th = corr_th

        def is_double_talk(self, mic_frame, ref_frame):
            mic_env = np.abs(mic_frame)
            ref_env = np.abs(ref_frame)
            corr = np.corrcoef(mic_env, ref_env)[0,1]
            if corr < self.corr_th:
                print(f"Envelope mismatch detected: {corr:.4f} < {self.corr_th}")
            return corr < self.corr_th

        
    class ZCR_Energy_DTD:
        def __init__(self, energy_th=1e-4, zcr_th=0.1):
            self.energy_th = energy_th
            self.zcr_th = zcr_th

        def is_double_talk(self, mic_frame: np.ndarray) -> bool:
            energy = np.mean(mic_frame ** 2)
            zcr = np.mean(np.abs(np.diff(np.sign(mic_frame)))) / 2
            if  energy > self.energy_th and zcr > self.zcr_th:
                print(f"ZCR: {zcr:.4f}, Energy: {energy:.7f}, ZCR threshold: {self.zcr_th}, Energy threshold: {self.energy_th}")
            return energy > self.energy_th and zcr > self.zcr_th

    def __init__(self, detect_lenth=1024, method="all"):
        self.detect_len = detect_lenth
        self.energy_ratio_dtd = self.EnergyRatioDTD(threshold=0.3)
        self.shape_dtd = self.EnvelopeMismatchDTD(corr_th=0.1)
        self.zcr_energy_dtd = self.ZCR_Energy_DTD()
        self.method = method.lower()
    
    def is_duble_talk(self, mic_frame, ref_frame):
        if "energy" in self.method:
            ret_flag = self.energy_ratio_dtd.is_double_talk(mic_frame, ref_frame)
        if "shape" in self.method:
            ret_flag = self.shape_dtd.is_double_talk(mic_frame, ref_frame)
        if "zcr" in self.method:
            ret_flag = self.zcr_energy_dtd.is_double_talk(mic_frame)
        else:
            ret_flag = (self.energy_ratio_dtd.is_double_talk(mic_frame, ref_frame) and 
                        self.shape_dtd.is_double_talk(mic_frame, ref_frame) and 
                        self.zcr_energy_dtd.is_double_talk(mic_frame))
        return ret_flag

    def is_double_talk_flag(self, mic_frame, ref_frame):
        # vibe mic volume is too low, so we amplify it 25dB
        # gain = 20
        # mic_frame = mic_frame * (10 ** (gain / 20))
        out_flag = np.zeros_like(mic_frame, dtype=int)
        ret_flag = False
        
        if len(mic_frame) <= self.detect_len:
            ret_flag = self.is_duble_talk(mic_frame, ref_frame)
            if ret_flag:
                out_flag[:] = 1
        else:
            for i in range(0, len(mic_frame) - self.detect_len + 1, self.detect_len):
                mic_segment = mic_frame[i:i + self.detect_len]
                ref_segment = ref_frame[i:i + self.detect_len]
                ret_flag = self.is_duble_talk(mic_segment, ref_segment)
         
                if ret_flag:           
                    out_flag[i:i + self.detect_len] = 1
                else:
                    out_flag[i:i + self.detect_len] = 0
            if len(mic_frame) % self.detect_len != 0:
                mic_segment = mic_frame[-self.detect_len:]
                ref_segment = ref_frame[-self.detect_len:]
                ret_flag = self.is_duble_talk(mic_segment, ref_segment)
                if ret_flag:
                    out_flag[-self.detect_len:] = 1
                else:
                    out_flag[-self.detect_len:] = 0
                
        return out_flag
    

        
        
