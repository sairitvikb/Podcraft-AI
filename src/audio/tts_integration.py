from elevenlabs import generate, set_api_key, save
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    def __init__(self, api_key: Optional[str] = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        if api_key:
            set_api_key(api_key)
        self.api_key = api_key
        self.voice_id = voice_id
        logger.info("ElevenLabs TTS initialized")
    
    def generate(self, text: str, output_path: str, voice_id: Optional[str] = None) -> str:
        voice = voice_id or self.voice_id
        
        try:
            audio = generate(
                text=text,
                voice=voice,
                model="eleven_multilingual_v2"
            )
            
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            save(audio, output_path)
            
            logger.info(f"Generated audio: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise


class BatchTTSGenerator:
    def __init__(self, tts_engine, batch_size: int = 5):
        self.tts_engine = tts_engine
        self.batch_size = batch_size
    
    def generate_batch(self, texts: List[str], output_dir: str) -> List[str]:
        output_paths = []
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"segment_{i}.mp3")
            path = self.tts_engine.generate(text, output_path)
            output_paths.append(path)
        
        return output_paths
    
    def concatenate_audio(self, audio_files: List[str], output_path: str):
        from pydub import AudioSegment
        
        combined = AudioSegment.empty()
        for audio_file in audio_files:
            audio = AudioSegment.from_mp3(audio_file)
            combined += audio
        
        combined.export(output_path, format="mp3")
        logger.info(f"Concatenated audio: {output_path}")
        return output_path
