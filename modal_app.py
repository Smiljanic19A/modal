import modal
import time
import json
import whisperx
import os
import torchaudio


CACHE_PATH = "/build_cache"
REMOTE_PATH = "/home/learning/audio"

# Function to pre-download whisperX model
def pre_download_model():
    import whisperx
    import gc
    compute_type = "float16"
    whisperx.load_model("large-v2", device="cuda", download_root=CACHE_PATH)

def pre_download_align_model():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    model_name = "facebook/wav2vec2-large-960h"
    device = "cuda"
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=CACHE_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_PATH).to(device)

def pre_download_diarization_model():
    from pyannote.audio import Pipeline
    import torch
    model_name = "pyannote/speaker-diarization-3.0"
    device = torch.device("cuda")
    diarization_model = Pipeline.from_pretrained(model_name, use_auth_token="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")
    diarization_model.to(device)

# Setting up the image
whisperX_image = (
    modal.Image.from_registry(  
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.11",
    )
    .apt_install("git", "libsndfile1")
    .pip_install("git+https://github.com/nikola1975/whisperXfork.git")
    .pip_install(
        "torch",
        "torchaudio",
        "SoundFile",
        "diffusers",
        "transformers",
        "numpy",
        "pydub",
        "ffmpeg-python",
        "pyannote.audio==3.0.1",
    )
    .run_commands(
        [
            "apt-get update",
            "apt-get install --yes ffmpeg ",
        ]
    )
    # Pre-download models
    .run_function(pre_download_model, gpu="t4")
    .run_function(pre_download_align_model, gpu="t4")
    .run_function(pre_download_diarization_model, gpu="t4")
)

# Initialize the stub
stub = modal.Stub(name="whisperX-app", image=whisperX_image)
# stub.volume = modal.NetworkFileSystem.persisted("Qira_volume")

@stub.function(
        gpu="T4", 
        # mounting local folder so I can read from it
        mounts=[modal.Mount.from_local_dir("audio", remote_path="/home/learning/audio")],
        timeout=360, 
    )

def whisper_function():   
    print(os.listdir("/home/learning/audio"))
    audio_file = f"/home/learning/audio/audio.mp3"
    device = "cuda" 
    batch_size = 16
    compute_type = "float16"

    print("Starting transcription")
    # Transcription
    start_time_transcription = time.time()
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=CACHE_PATH)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    end_time_transcription = time.time()
    print(result["segments"])
    metadata_path = os.path.join(REMOTE_PATH, "transcription_result.json")
    with open(metadata_path, "w") as f:
        json.dump(result["segments"], f, indent=4)

    print("Starting alignment")
    # Alignment
    start_time_alignment = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    end_time_alignment = time.time()
    print(result["segments"])
    # write_alignment_to_json(result["segments"])
    
    print("Starting diarization")
    # Diarization
    start_time_diarization = time.time()
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    end_time_diarization = time.time()
    # print(diarize_segments)
    print(result["segments"])
    with open(f'{REMOTE_PATH}/diarization_data.json', 'w') as f:
        json.dump(result["segments"], f, indent=4)

    print(f"Time taken for Transcription: {end_time_transcription - start_time_transcription} seconds")
    print(f"Time taken for Alignment: {end_time_alignment - start_time_alignment} seconds")
    print(f"Time taken for Diarization: {end_time_diarization - start_time_diarization} seconds")


