from fastapi import FastAPI, HTTPException, Form
import requests
import asyncio
import boto3
from botocore.config import Config
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
from dotenv import load_dotenv
from datetime import datetime
import uuid

load_dotenv()

app = FastAPI()

# R2 setup
r2 = boto3.client(
    service_name='s3',
    endpoint_url=os.getenv('R2_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'),
)

BUCKET_NAME = os.getenv('R2_BUCKET_NAME')

# Load models at startup
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


async def process_tts(audio_url: str, reference_text: str, text_to_generate: str, callback_url: str, generated_audio_id: str):
    temp_audio_path = None
    output_path = None

    try:
        # Download audio
        print("Downloading audio...")
        response = requests.get(audio_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name

        # Process audio
        print("Processing audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(
            temp_audio_path,
            reference_text
        )

        # Generate speech
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

        # Save output
        print("Saving output...")
        output_path = tempfile.mktemp(suffix='.wav')
        sf.write(output_path, final_wave, final_sample_rate)

        # Upload to R2
        print("Uploading to R2...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_key = f"tts/{timestamp}_{uuid.uuid4()}.wav"
        r2.upload_file(output_path, BUCKET_NAME, file_key)

        # Generate URL (valid for 6 months)
        # TODO: this not really secure i think
        url = r2.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': file_key},
            ExpiresIn=15552000
        )

        # Send callback
        print("Sending callback...")
        requests.post(callback_url, json={
            "status": "completed",
            "audio_url": url,
            "file_key": file_key,
            "generated_audio_id": generated_audio_id
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        requests.post(callback_url, json={
            "status": "failed",
            "generated_audio_id": generated_audio_id,
            "error": str(e)
        })

    finally:
        # Cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)


@app.post("/tts")
async def generate_speech(
    audio_url: str = Form(...),
    reference_text: str = Form(None),
    text_to_generate: str = Form(...),
    callback_url: str = Form(...),
    generated_audio_id: str = Form(...),
):
    asyncio.create_task(process_tts(
        audio_url, reference_text, text_to_generate, callback_url, generated_audio_id))
    return {"status": "processing", "message": "TTS generation started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

