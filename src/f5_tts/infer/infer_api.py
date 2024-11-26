from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process
)
from cached_path import cached_path
import numpy as np
import soundfile as sf
import tempfile
from typing import Optional
import base64
import os

app = FastAPI()

# Load models once at startup
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16,
                       ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT,
    F5TTS_model_cfg,
    str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
)


class TTSRequest(BaseModel):
    audio_url: str  # Base64 encoded audio file
    reference_text: Optional[str] = None
    text_to_generate: str


class TTSResponse(BaseModel):
    audio: str  # Base64 encoded audio file
    sample_rate: int


@app.post("/generate_speech", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    try:
        # Decode base64 audio and save temporarily
        audio_data = base64.b64decode(request.audio_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        # Process reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(
            temp_audio_path,
            request.reference_text
        )

        # Generate speech
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            request.text_to_generate,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=0.15,
            speed=1.0
        )

        # Save generated audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_out:
            sf.write(temp_out.name, final_wave, final_sample_rate)

            # Read and encode to base64
            with open(temp_out.name, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode()

        # Clean up temp files
        os.unlink(temp_audio_path)
        os.unlink(temp_out.name)

        return TTSResponse(
            audio=audio_base64,
            sample_rate=final_sample_rate
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
