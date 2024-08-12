import modal
import json
import whisperx
import os
import time
import httpx
import logging
from typing import List
from pydantic import BaseModel, HttpUrl
from fastapi import FastAPI, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from modal import Image, Stub, asgi_app, NetworkFileSystem
#from whisperx import load_align_model, align, SubtitlesProcessor
import whisperx
import Levenshtein
import pysrt
import webvtt
import requests
from whisperx.SubtitlesProcessor import SubtitlesProcessor

CACHE_PATH = "/build_cache" 
logger = logging.getLogger(__name__)
MODEL_DIR = "/models" 
HF_AUTH_TOKEN = "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"
os.environ["TRANSFORMERS_CACHE"] = "/build_cache"

# Define persisted network file systems for models and output files
#volume = NetworkFileSystem.persisted("dataset-cache-vol")
volume = NetworkFileSystem.from_name("dataset-cache-vol", create_if_missing=True)
#Tolerance For Levenshtein Distance
transcription_threshold = 90

# Dictionary mapping language codes to model names. Add languages to avoid downloading model on every Modal container launch
LANGUAGE_MODEL_MAP = {
    "en": "facebook/wav2vec2-large-960h",
    "sr": "voidful/wav2vec2-xlsr-multilingual-56",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "hr": "voidful/wav2vec2-xlsr-multilingual-56",
    "bs": "voidful/wav2vec2-xlsr-multilingual-56",
    "sl": "bekirbakar/wav2vec2-large-xls-r-300m-slovenian",
}

test_audio_url = "https://s3.eu-central-2.wasabisys.com/qira/714/2024/8/WhatsApp_Audio_2024-08-08_at_14.41.4_1723125385333/WhatsApp_Audio_2024-08-08_at_14.41.4_1723125385333.mp3"


def transcribe_audio(audio_url):
    post_url = "https://api.gladia.io/v2/transcription"
    headers = {
        "Content-Type": "application/json",
        "x-gladia-key": "f6698a07-6db5-4c37-84a4-9e4e85c71086"
    }
    payload = {"audio_url": audio_url}

    try:
        response = requests.post(post_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        print("Response:", response_data)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

    result_url = response_data.get('result_url')
    id = response_data.get('id')

    if not id:
        print("No valid ID returned from the initial request.")
        return None

    get_url = f"https://api.gladia.io/v2/transcription/{id}"
    headers = {"x-gladia-key": "f6698a07-6db5-4c37-84a4-9e4e85c71086"}

    while True:
        try:
            response = requests.get(get_url, headers=headers, timeout=60)
            response_data = response.json()

            if response_data.get('status') == 'done':
                break

            print("Status is 'queued'. Retrying in 10 seconds...")
            time.sleep(10)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while polling: {e}")
            time.sleep(10)

    return response_data

def cyrillic_to_latin(text):
    cyrillic_to_latin_map = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I',
        'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R',
        'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i',
        'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
    }
    return ''.join(cyrillic_to_latin_map.get(char, char) for char in text)

def parse_webvtt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        parsed_text = []

        for line in lines:
            if '-->' in line:
                continue
            if line.strip() == "WEBVTT" or line.strip() == "":
                continue
            parsed_text.append(line.strip())

        combined_text = ' '.join(parsed_text)
        return cyrillic_to_latin(combined_text)

    except FileNotFoundError:
        return "File does not exist."

def calc_levenshtein_distance(string1, string2):
    distance = Levenshtein.distance(string1, string2)
    max_length = max(len(string1), len(string2))
    return (1 - distance / max_length) * 100

def format_timestamp(seconds: float, is_vtt: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    separator = '.' if is_vtt else ','
    hours_marker = f"{hours:02d}:"
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"
    )


# Functions to pre-download whisperX, alignment, and diarization models
def pre_download_model():
    import whisperx
    compute_type = "float16"
    #whisperx.load_model("large-v2", device="cuda", download_root=CACHE_PATH)
    whisperx.load_model("large-v2", device="cpu", download_root=CACHE_PATH)


def pre_download_align_model(language_code="en"):   
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    logger.info("Starting the pre-download of alignment model...")
    model_name = LANGUAGE_MODEL_MAP.get(language_code, "facebook/wav2vec2-large-960h")

    # Ensure CACHE_PATH exists
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # Check for specific model files in the cache directory
    model_files_path = os.path.join(CACHE_PATH, model_name.replace("/", "__"))
    model_files_exist = os.path.exists(os.path.join(model_files_path, "pytorch_model.bin"))

    if not model_files_exist:
        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=CACHE_PATH)
        #model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_PATH).to("cuda")
        model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_PATH).to("cpu")

    else:
        print(f"Model for {language_code} already downloaded.")

def pre_download_diarization_model():
    from pyannote.audio import Pipeline
    import torch
    YOUR_AUTH_TOKEN = "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"
    model_name = "pyannote/speaker-diarization-3.1"
    #device = "cuda"
    device = "cpu"


    diarization_model = Pipeline.from_pretrained(model_name, use_auth_token=YOUR_AUTH_TOKEN)
    device = torch.device('cpu')
    diarization_model.to(device)
    logger.info("Model loaded successfully and moved to CUDA.")

# Setting up the image
whisperX_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", 
        add_python="3.11",
    )
    .apt_install("git", "libsndfile1") 
    .pip_install("git+https://github.com/nikola1975/whisperXfork.git@bc2a64106761ff8c3602f8eaf8dd7e7b58656a89")
    .pip_install(
        "torch",
        # "whisper==1.1.10",
        # "faster-whisper",
        "torchaudio",
        "SoundFile",
        "diffusers",
        "transformers",
        "numpy",
        "pydub",
        "ffmpeg-python",
        "httpx",
        "pyannote.audio==3.1.1",
    )
    .pip_install("git+https://github.com/SYSTRAN/faster-whisper@65551c081f17f30e9e73debbbdd148233de67ac7")
    .env({
        "TORCH_HOME": "/build_cache",
        "HF_HOME": "/build_cache",
        "TRANSFORMERS_CACHE": "/build_cache",
        "HF_AUTH_TOKEN": "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"
    })
    .run_commands(
        [
            "apt-get update",
            "apt-get install --yes ffmpeg ",
            # Add -c commands for each model
            "python3 -c 'import torchaudio; bundle = torchaudio.pipelines.__dict__[\"WAV2VEC2_ASR_BASE_960H\"]; align_model = bundle.get_model(); labels = bundle.get_labels()'",
            "python3 -c 'import torchaudio; bundle = torchaudio.pipelines.__dict__[\"VOXPOPULI_ASR_BASE_10K_FR\"]; align_model = bundle.get_model(); labels = bundle.get_labels()'",
            "python3 -c 'import torchaudio; bundle = torchaudio.pipelines.__dict__[\"VOXPOPULI_ASR_BASE_10K_DE\"]; align_model = bundle.get_model(); labels = bundle.get_labels()'",
            "python3 -c 'import torchaudio; bundle = torchaudio.pipelines.__dict__[\"VOXPOPULI_ASR_BASE_10K_ES\"]; align_model = bundle.get_model(); labels = bundle.get_labels()'",
            "python3 -c 'import torchaudio; bundle = torchaudio.pipelines.__dict__[\"VOXPOPULI_ASR_BASE_10K_IT\"]; align_model = bundle.get_model(); labels = bundle.get_labels()'",
            
            # Add commands for Hugging Face models
            "python3 -c 'from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; import os; processor = Wav2Vec2Processor.from_pretrained(\"voidful/wav2vec2-xlsr-multilingual-56\", use_auth_token=os.environ[\"HF_AUTH_TOKEN\"], cache_dir=\"/build_cache\"); model = Wav2Vec2ForCTC.from_pretrained(\"voidful/wav2vec2-xlsr-multilingual-56\", use_auth_token=os.environ[\"HF_AUTH_TOKEN\"], cache_dir=\"/build_cache\")'",
            "python3 -c 'from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; processor = Wav2Vec2Processor.from_pretrained(\"jonatasgrosman/wav2vec2-large-xlsr-53-portuguese\", cache_dir=\"/build_cache\"); model = Wav2Vec2ForCTC.from_pretrained(\"jonatasgrosman/wav2vec2-large-xlsr-53-portuguese\", cache_dir=\"/build_cache\")'",
            "python3 -c 'from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; processor = Wav2Vec2Processor.from_pretrained(\"jonatasgrosman/wav2vec2-large-xlsr-53-greek\", cache_dir=\"/build_cache\"); model = Wav2Vec2ForCTC.from_pretrained(\"jonatasgrosman/wav2vec2-large-xlsr-53-greek\", cache_dir=\"/build_cache\")'",
            "python3 -c 'from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; processor = Wav2Vec2Processor.from_pretrained(\"infinitejoy/wav2vec2-large-xls-r-300m-bulgarian\", cache_dir=\"/build_cache\"); model = Wav2Vec2ForCTC.from_pretrained(\"infinitejoy/wav2vec2-large-xls-r-300m-bulgarian\", cache_dir=\"/build_cache\")'",
            "python3 -c 'from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; processor = Wav2Vec2Processor.from_pretrained(\"jonatasgrosman/wav2vec2-large-xlsr-53-hungarian\", cache_dir=\"/build_cache\"); model = Wav2Vec2ForCTC.from_pretrained(\"jonatasgrosman/wav2vec2-large-xlsr-53-hungarian\", cache_dir=\"/build_cache\")'",
        ]
    )
    .run_function(pre_download_model, gpu="t4")
    .run_function(pre_download_align_model, gpu="t4")
    .run_function(pre_download_diarization_model, gpu="t4")
)

web_app = FastAPI()
stub = modal.Stub(name="whisperX-app", image=whisperX_image)

class TranscriptionRequest(BaseModel):
    file_url: HttpUrl
    language: str
    output_formats: List[str]

@web_app.post("/transcribe_audio")
async def transcribe_audio_endpoint(
    transcription_request: TranscriptionRequest,
    request: Request = None
    ):

    pre_download_align_model(transcription_request.language)

    # download the file from the URL
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(transcription_request.file_url)
            resp.raise_for_status()
            file_contents = resp.content
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail="Error downloading file from URL")
    
    result = transcribe_and_align_audio(file_contents, transcription_request.language, transcription_request.output_formats)

    return {
        'status': 'success',
        'transcription_links': {
            format: f"https://nikola1975--whisperx-app-fastapi-app.modal.run/download/{result['full_paths'][format]}"
            for format in transcription_request.output_formats
        } | (
            {'aligned_json': f"https://nikola1975--whisperx-app-fastapi-app.modal.run/download/{result['aligned_transcription_file']}"}
            if 'align' in transcription_request.output_formats else {}
        )
    }

@stub.function(network_file_systems={"/root/output": volume})
@web_app.get("/download/{file_path:path}")
async def download(file_path: str):
    if not file_path.startswith("output/"):
        raise HTTPException(status_code=400, detail="Invalid file path")

    return FileResponse(file_path)

@stub.function(
        network_file_systems={"/root/output": volume},
        gpu="T4",
        timeout=1800, 
        secrets=[modal.Secret.from_name("keys")],
    )

@asgi_app()
def fastapi_app():
    return web_app

def transcribe_and_align_audio(file_contents: bytes, language: str, output_formats: list, custom_dictionary: str = ""):
    language = language.replace('"', '')
    output_formats = [format.replace('"', '') for format in output_formats]
    microtime = str(int(time.time() * 1000))

    #device = "cuda"
    device = "cpu"
    batch_size = 16
    #compute_type = "float16"
    compute_type = "float32"


    json_path = None
    aligned_json_path = None
    vtt_path = None

    if not os.path.exists("output"):
        os.makedirs("output")

    temp_audio_file_path = f"temp_audio_file_{microtime}.mp3"
    with open(temp_audio_file_path, "wb") as f:
        f.write(file_contents)

    # Prepare ASR options with custom dictionary coming from the frontend
    asr_options = {
        "initial_prompt": custom_dictionary,  # Use the custom dictionary provided from the frontend
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    # Transcription
    print(f"Transcribing audio file")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type,
                                download_root=CACHE_PATH, language=language,
                                asr_options=asr_options)
    print("Model Successfully Loaded!")
    result = model.transcribe(temp_audio_file_path, batch_size=batch_size)
    print(f"Result: {result}")
    all_transcriptions = result["segments"]
    print(f"All Transcriptions: {all_transcriptions}")
    transcription_text = cyrillic_to_latin(' '.join([segment["text"] for segment in all_transcriptions]))
    print(f"Transcription Raw Text: {transcription_text}")
    # TODO: Do the transcription in txt file and save to TXT
    
    # Save original JSON
    output_files = {}
    if 'json' in output_formats:
        json_path = os.path.join("output", f"transcription_{microtime}.json")
        mp3_path = os.path.join("output", f"{temp_audio_file_path}")
        with open(json_path, "w") as f:
            json.dump(all_transcriptions, f, indent=4)
        print(f"Transcription written to {json_path}")
        print(f"Transcription written to {mp3_path}")
        output_files['json'] = json_path

    # Alignment
    print(f"Starting alignment")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(all_transcriptions, model_a, metadata,
                           temp_audio_file_path, device,
                           return_char_alignments=False)

    # Save aligned JSON 
    if 'align' in output_formats:
        aligned_json_path = os.path.join("output", f"aligned_{microtime}.json")
        with open(aligned_json_path, "w") as f:
            json.dump(aligned_result["segments"], f, indent=4)
        print(f"Alignment written to {json_path}")
        output_files['align'] = aligned_json_path

    # Diarization
    if 'diarize' in output_formats:
        print("Starting diarization")
        hf_api_token = os.getenv("HF_API_TOKEN")
        if not hf_api_token:
            raise ValueError("Hugging Face API token not found in environment variables.")
        diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=hf_api_token)
        diarize_segments = diarize_model(temp_audio_file_path)
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        print(f"Diarization Result: {result}")

        # Prepare diarization results for JSON
        diarization_results_for_json = []
        for segment in result["segments"]:
            segment_copy = {key: value for key, value in segment.items() if key not in ["clean_char", "clean_cdx", "clean_wdx"]}
            diarization_results_for_json.append(segment_copy)
        diarization_json_path = os.path.join("output", f"diarize_{microtime}.json")
        with open(diarization_json_path, "w") as f:
            json.dump(diarization_results_for_json, f, indent=4)
        print(f"Diarization written to {json_path}")
        output_files['diarize'] = diarization_json_path

    # Create VTT and SRT using aligned JSON
    if 'vtt' in output_formats:
        print("Preparing subtitles")
        vtt_path = os.path.join("output", f"subtitles_{microtime}.vtt")
        try:
            write_vtt(aligned_result['segments'], vtt_path, language)
            output_files['vtt'] = vtt_path
            logger.info(f"VTT file created successfully at {vtt_path}")
        except Exception as e:
            logger.error(f"Failed to create VTT file at {vtt_path}: {e}")
            raise e

    # Parse VTT to Single Line Latin Char String:
    whisper_vtt_text = parse_webvtt(vtt_path)
    print(f"Whisper VTT Converted To Text: {whisper_vtt_text}")
    # TODO: Use Levenshtein Distance to calculate difference between TXT and VTT-txt
    distance_from_whisper = calc_levenshtein_distance(transcription_text, cyrillic_to_latin(whisper_vtt_text))
    # If Distance From Whisper Is Satisfactory, Return Right Away
    if distance_from_whisper > transcription_threshold:
        print(f"Distance between transcription and vtt text is larger than 90, can be returned. Distance: {distance_from_whisper}")
        full_paths = {key: value.replace('/root/', '') for key, value in output_files.items()}
        return {
            'status': 'success',
            'transcription_file': f'{temp_audio_file_path}_transcription.json',
            'aligned_transcription_file': aligned_json_path,
            'other_files': [f'{temp_audio_file_path}_{format}' for format in output_formats if format != 'json'],
            'full_paths': full_paths
        }
    # TODO: If different -> use Gladia and its VTT file
    else:
        transcript = transcribe_audio(test_audio_url)
        gladia_transcript_text = transcript['result']['transcription']['full_transcript'] #Raw Text
        distance_from_gladia = calc_levenshtein_distance(transcription_text, cyrillic_to_latin(gladia_transcript_text))
        #Case: Both whisper and gladia have a lev distance < then acceptable (RETURN THE CLOSER ONE WITH A FLAG)
        if distance_from_whisper < transcription_threshold and distance_from_gladia < transcription_threshold:
            full_paths = {key: value.replace('/root/', '') for key, value in output_files.items()}
            if distance_from_whisper > distance_from_gladia:
                print("Whisper closer than Gladia, returning Whisper transcription with warning flag")
            else:
                print("Gladia Closer Than Whisper, Returning Gladia transcription with warning flag")
            return {
                'status': 'success',
                'transcription_file': f'{temp_audio_file_path}_transcription.json',
                'aligned_transcription_file': aligned_json_path,
                'other_files': [f'{temp_audio_file_path}_{format}' for format in output_formats if format != 'json'],
                'full_paths': full_paths
            }
        if distance_from_gladia > distance_from_whisper:
            print(f"Whisper distance: {distance_from_whisper}, Gladia Distance: {distance_from_gladia}")
            print("Returning Gladia transcript")
            full_paths = {key: value.replace('/root/', '') for key, value in output_files.items()}
            return {
                'status': 'success',
                'transcription_file': f'{temp_audio_file_path}_transcription.json',
                'aligned_transcription_file': aligned_json_path,
                'other_files': [f'{temp_audio_file_path}_{format}' for format in output_formats if format != 'json'],
                'full_paths': full_paths
            }
        elif distance_from_whisper > distance_from_gladia:
            print(f"Whisper distance: {distance_from_whisper}, Gladia Distance: {distance_from_gladia}")
            print("Returning Whisper transcript")
            full_paths = {key: value.replace('/root/', '') for key, value in output_files.items()}
            return {
                'status': 'success',
                'transcription_file': f'{temp_audio_file_path}_transcription.json',
                'aligned_transcription_file': aligned_json_path,
                'other_files': [f'{temp_audio_file_path}_{format}' for format in output_formats if format != 'json'],
                'full_paths': full_paths
            }



    full_paths = {key: value.replace('/root/', '') for key, value in output_files.items()}

    return {
        'status': 'success',
        'transcription_file': f'{temp_audio_file_path}_transcription.json',
        'aligned_transcription_file': aligned_json_path,
        'other_files': [f'{temp_audio_file_path}_{format}' for format in output_formats if format != 'json'],
        'full_paths': full_paths
    }

def write_vtt(segments, path, language):
    subtitles_processor = SubtitlesProcessor(segments, language)
    processed_subtitles = subtitles_processor.process_segments()

    with open(path, 'w') as f:
        f.write("WEBVTT\n\n")
        for subtitle in processed_subtitles:
            start = format_timestamp(subtitle['start'], is_vtt=True)
            end = format_timestamp(subtitle['end'], is_vtt=True)
            f.write(f"{start} --> {end}\n{subtitle['text']}\n\n")

if __name__ == "__main__":
    import asyncio

    async def test_transcribe_audio():
        transcription_request = TranscriptionRequest(
            file_url=test_audio_url,
            language="en",
            output_formats=["json", "align", "vtt"]
        )

        transcription_request.file_url = str(transcription_request.file_url)

        response = await transcribe_audio_endpoint(transcription_request)
        print(response)

    asyncio.run(test_transcribe_audio())