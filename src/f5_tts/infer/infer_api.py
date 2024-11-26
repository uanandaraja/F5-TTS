from fastapi import FastAPI, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
import requests
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process
)
from cached_path import cached_path
import soundfile as sf
import tempfile
import os

app = FastAPI()

print("Loading models...")
vocoder = load_vocoder()
F5TTS_model_cfg = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4
)
F5TTS_ema_model = load_model(
    DiT,
    F5TTS_model_cfg,
    str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
)
print("Models loaded successfully")


@app.post("/tts")
async def generate_speech(
    background_tasks: BackgroundTasks,
    audio_url: str = Form(...),
    reference_text: str = Form(None),
    text_to_generate: str = Form(...)
):
    try:
        print("Downloading audio from URL...")
        response = requests.get(audio_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name

        print("Processing reference audio and text...")
        ref_audio, ref_text = preprocess_ref_audio_text(
            temp_audio_path,
            reference_text
        )

        print("Generating speech...")
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            text_to_generate,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=0.15,
            speed=1.0
        )

        print("Saving output audio...")
        output_path = tempfile.mktemp(suffix='.wav')
        sf.write(output_path, final_wave, final_sample_rate)

        os.unlink(temp_audio_path)
        background_tasks.add_task(os.unlink, output_path)

        return FileResponse(
            output_path,
            media_type='audio/wav',
            filename='generated_speech.wav'
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
