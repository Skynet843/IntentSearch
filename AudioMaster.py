import os
import whisper
from typing import Optional, Dict
import torch
from QueryParser import QueryRewriter
class AudioSearchPipeline:
    def __init__(
        self,
        whisper_model_size: str = "base",
        rewriter_model_name: str = "gemini-1.5-flash",
        use_gpu: bool = True
    ):
        """
        Initializes the audio search pipeline.
        
        Args:
            whisper_model_size (str): Size of Whisper model ("tiny", "base", "small", etc.)
            rewriter_model_name (str): Gemini model to use for query rewriting
            use_gpu (bool): Whether to run Whisper on GPU
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
        self.query_rewriter = QueryRewriter(model_name=rewriter_model_name)

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes audio to text using Whisper.
        """
        result = self.whisper_model.transcribe(audio_path)
        return result.get("text", "").strip()

    def process(self, audio_path: str) -> Dict:
        """
        End-to-end pipeline: audio → transcription → query structuring.

        Returns:
            Dict: structured query with keys like `query`, `category`, `price_max`, `intent`
        """
        try:
            print(f"🎙️ Transcribing audio: {audio_path}")
            text_query = self.transcribe_audio(audio_path)
            print(f"📝 Transcribed Text: {text_query}")
            # structured_query = self.query_rewriter.rewrite(text_query)
            return text_query
        except Exception as e:
            return None


# Example Usage
if __name__ == "__main__":
    pipeline = AudioSearchPipeline()

    # Replace this with your audio file
    audio_file = "test_queries/yoga_query.wav"

    result = pipeline.process(audio_file)
    print("\n🔍 Final Structured Query:")
    print(result)