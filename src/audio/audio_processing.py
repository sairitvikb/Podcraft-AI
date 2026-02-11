import numpy as np
from typing import List, Optional, Tuple
import logging
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except:
            audio_segment = AudioSegment.from_file(file_path)
            audio = np.array(audio_segment.get_array_of_samples())
            if audio_segment.channels == 2:
                audio = audio.reshape((-1, 2)).mean(axis=1)
            sr = audio_segment.frame_rate
            return audio, sr
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        intervals = librosa.effects.split(audio, top_db=20, frame_length=frame_length, hop_length=hop_length)
        audio_trimmed = np.concatenate([audio[interval[0]:interval[1]] for interval in intervals])
        return audio_trimmed
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
        magnitude_denoised = magnitude - noise_profile
        magnitude_denoised = np.maximum(magnitude_denoised, 0)
        
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised)
        
        return audio_denoised
    
    def concatenate_audio(self, audio_files: List[str], output_path: str):
        combined = AudioSegment.empty()
        
        for audio_file in audio_files:
            audio = AudioSegment.from_file(audio_file)
            combined += audio
        
        combined.export(output_path, format="mp3")
        logger.info(f"Concatenated audio saved to {output_path}")
    
    def add_fade(self, audio: np.ndarray, fade_in_ms: int = 500, fade_out_ms: int = 500) -> np.ndarray:
        fade_in_samples = int(fade_in_ms * self.sample_rate / 1000)
        fade_out_samples = int(fade_out_ms * self.sample_rate / 1000)
        
        fade_in = np.linspace(0, 1, fade_in_samples)
        fade_out = np.linspace(1, 0, fade_out_samples)
        
        audio[:fade_in_samples] *= fade_in
        audio[-fade_out_samples:] *= fade_out
        
        return audio
