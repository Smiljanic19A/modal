from difflib import SequenceMatcher

import modal
import json
import whisperx
import os
import time
import httpx
import logging
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from fastapi import FastAPI, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from modal import Image, Stub, asgi_app, NetworkFileSystem
from whisperx import load_align_model, align, SubtitlesProcessor
import whisperx
import Levenshtein
import requests
from whisperx.SubtitlesProcessor import SubtitlesProcessor
import re

from AwSubtitleProcessor import AwSubtitleProcessor

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

test_audio_url = "https://s3.eu-central-2.wasabisys.com/qira/658/2024/7/_ovjeku_se_plae__bacio_sam_deset_ton_1722256407942/_ovjeku_se_plae__bacio_sam_deset_ton_1722256407942.mp3"





# Functions to pre-download whisperX, alignment, and diarization models
def pre_download_model():
    import whisperx
    compute_type = "float16"
    #whisperx.load_model("large-v2", device="cuda", download_root=CACHE_PATH)
    whisperx.load_model("large-v2", device="cuda", download_root=CACHE_PATH)


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
        model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_PATH).to("cuda")
        #model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=CACHE_PATH).to("cuda")

    else:
        print(f"Model for {language_code} already downloaded.")

def pre_download_diarization_model():
    from pyannote.audio import Pipeline
    import torch
    YOUR_AUTH_TOKEN = "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"
    model_name = "pyannote/speaker-diarization-3.1"
    device = "cuda"
    #device = "cpu"


    diarization_model = Pipeline.from_pretrained(model_name, use_auth_token=YOUR_AUTH_TOKEN)
    device = torch.device('cuda')
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
        'Levenshtein'
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
    retranscription: Optional[bool]

@web_app.post("/transcribe_audio")
async def transcribe_audio_endpoint(
    transcription_request: TranscriptionRequest,
    request: Request = None
    ):

    print(f"Transcription Request: {transcription_request}")

    pre_download_align_model(transcription_request.language)

    # download the file from the URL
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(transcription_request.file_url)
            resp.raise_for_status()
            file_contents = resp.content
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail="Error downloading file from URL")
    
    result = transcribe_and_align_audio(file_contents, transcription_request.language, transcription_request.output_formats, transcription_request.file_url, transcription_request.retranscription)

    return {
        'status': 'success',
        'transcription_links': {
            format: f"https://modalqira--whisperx-app-fastapi-app.modal.run/download/{result['full_paths'][format]}"
            for format in transcription_request.output_formats
        } | (
            {'aligned_json': f"https://modalqira--whisperx-app-fastapi-app.modal.run/download/{result['aligned_transcription_file']}"}
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


def transcribe_and_align_audio(file_contents: bytes, language: str, output_formats: list, file_url: str, retranscription: bool, custom_dictionary: str = ""):
    #print(f"RETRANSCRIPTION: {retranscription}")
    retranscription = False
    language = language.replace('"', '')
    output_formats = [format.replace('"', '') for format in output_formats]
    microtime = str(int(time.time() * 1000))

    print(f"File url:{file_url}")

    device = "cuda"
    batch_size = 16
    compute_type = "float16"

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
    custom_json = convert_to_custom_json(all_transcriptions)
    print(f"created custom json {custom_json}")
    json_content = None
    # Save original JSON
    output_files = {}
    if 'json' in output_formats or "vtt" in output_formats:
        json_path = os.path.join("output", f"transcription_{microtime}.json")
        mp3_path = os.path.join("output", f"{temp_audio_file_path}")
        with open(json_path, "w") as f:
            json.dump(all_transcriptions, f, indent=4)
        print(f"Transcription written to {json_path}")
        print(f"Transcription written to {mp3_path}")
        output_files['json'] = json_path
        json_content = json.dumps(all_transcriptions, indent=4)

    # Alignment
    print(f"Starting alignment")
    if language != "sl":
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    else:
        model_a, metadata = whisperx.load_align_model(language_code="sr", device=device)

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
        hf_api_token = HF_AUTH_TOKEN
        if not hf_api_token:
            raise ValueError("Hugging Face API token not found in environment variables.")
        diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=hf_api_token)
        diarize_segments = diarize_model(temp_audio_file_path)
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

        # Prepare diarization results for JSON
        diarization_results_for_json = []
        for segment in result["segments"]:
            segment_copy = {key: value for key, value in segment.items() if
                            key not in ["clean_char", "clean_cdx", "clean_wdx"]}
            diarization_results_for_json.append(segment_copy)
        diarization_json_path = os.path.join("output", f"diarize_{microtime}.json")
        with open(diarization_json_path, "w") as f:
            json.dump(diarization_results_for_json, f, indent=4)
        print(f"Diarization written to {json_path}")
        output_files['diarize'] = diarization_json_path

    if 'vtt' in output_formats:
        if not retranscription:
            print("Preparing subtitles")

            aw_subtitle_processor = AwSubtitleProcessor(30, 40)
            words = aw_subtitle_processor.extract_subtitle_info(aligned_result)
            if words:
                print("Sucesfully fetched words!")

            subtitle_params = aw_subtitle_processor.prepare_params_for_write_vtt(words)
            #print(f"PARAMS: {json.dumps(subtitle_params, indent=4)}")



            vtt_path = os.path.join("output", f"subtitles_{microtime}.vtt")

            try:
                aw_subtitle_processor.write_vtt(subtitle_params, vtt_path, language)
                #write_vtt(subtitle_params, vtt_path, language)
                #logger.info(f"VTT file created successfully at {vtt_path}")
                whisper_vtt_text = parse_webvtt(vtt_path)
            except Exception as e:
                logger.error(f"Failed to create VTT file at {vtt_path}: {e}")
                raise e
        else:
            gladia_transcript = gladia_transcribe(file_url)
            ##print(f"Gladia Transcript: {gladia_transcript}")
            utterances = gladia_transcript['result']['transcription']['utterances']
            vtt_path = os.path.join("output", f"subtitles_gladia_{microtime}.vtt")

            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for utterance in utterances:
                    start_time = format_timestamp(utterance['start'], is_vtt=True)
                    end_time = format_timestamp(utterance['end'], is_vtt=True)
                    f.write(f"{start_time} --> {end_time}\n{utterance['text']}\n\n")

            print(f"Gladia VTT file written to {vtt_path}")
        output_files['vtt'] = vtt_path



    # Return the final result
    full_paths = {key: value.replace('/root/', '') for key, value in output_files.items()}

    return {
        'status': 'success',
        'transcription_file': f'{temp_audio_file_path}_transcription.json',
        'aligned_transcription_file': aligned_json_path,
        'other_files': [output_files.get(format) for format in output_formats if format != 'json'],
        'full_paths': full_paths  # Ensure vtt is included here
    }

def confirm_iteration(segments):
    print("Each Word In Words: \n")
    for segment in segments:
        print(segment['word'])

def normalize_string(s):
    return re.sub(r'\s+', '', s.lower())


def is_transcription_correct(transcription1, transcription2):

    # Normalize the transcriptions
    normalized1 = normalize_string(transcription1)
    normalized2 = normalize_string(transcription2)

    logging.info(f"String 1 : {normalized1}")
    logging.info(f"String 2 : {normalized2}")

    # Compare the normalized strings
    similarity = SequenceMatcher(None, normalized1, normalized2).ratio()

    # Return True if similarity is above 90%
    return similarity >= 0.9


def count_sentences(s):
    """
    Counts the number of sentences in a string by splitting it on sentence-ending punctuation.
    """
    # Split the string into sentences based on common sentence-ending punctuation (., !, ?)
    sentences = re.split(r'[.!?]\s*', s)
    # Filter out any empty strings that may result from the split
    return len([sentence for sentence in sentences if sentence.strip()])


def same_number_of_sentences(string1, string2):
    """
    Returns True if both strings have the same number of sentences, otherwise False.
    """
    # Count the number of sentences in each string
    num_sentences1 = count_sentences(string1)
    num_sentences2 = count_sentences(string2)
    print(f"Number of sentances 1: {num_sentences1}")
    print(f"Number of sentances 2: {num_sentences2}")
    print(f"string 1: {string1}")
    print(f"string 2: {string2}")

    # Compare the number of sentences
    return num_sentences1 == num_sentences2


import json
import re

def check_anomalies(segments):
    segments = json.loads(segments)
    # Define the average duration per word in seconds
    avg_word_length = 2  # average word duration in seconds

    for segment in segments:
        if not isinstance(segment, dict):
            print(f"Expected a dictionary but got {type(segment)}. Skipping this segment.")
            continue

        text = segment.get("text", "")
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)

        if not text:
            print("No text found in segment. Skipping this segment.")
            continue

        # Calculate the number of words in the text
        words = re.findall(r'\b\w+\b', text)
        num_words = len(words)

        # Calculate the actual duration of the segment
        actual_duration = end_time - start_time

        # Calculate the expected duration based on the number of words
        expected_duration = num_words * avg_word_length

        # Check if the actual duration is significantly longer than expected
        if actual_duration > expected_duration:
            print(
                f"Anomaly detected: Segment with start time {start_time} and end time {end_time} has {num_words} words, expected duration {expected_duration}, but actual duration is {actual_duration}.")
            return False

    # If all segments pass the check, return True
    return True

def gladia_transcribe(audio_url):
    # Define the URL and the headers
    post_url = "https://api.gladia.io/v2/transcription"
    headers = {
        "Content-Type": "application/json",
        "x-gladia-key": "f6698a07-6db5-4c37-84a4-9e4e85c71086"
    }

    # Define the payload
    payload = {
        "audio_url": audio_url
    }

    # Make the POST request
    try:
        response = requests.post(post_url, headers=headers, json=payload, timeout=60)
        #print("Status Code:", response.status_code)
        response_data = response.json()  # Parse the JSON response
        #print("Response:", response_data)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

    # Extract result_url and id if they exist
    result_url = response_data.get('result_url')
    id = response_data.get('id')

    # Check if the id is available before proceeding
    if not id:
        print("No valid ID returned from the initial request.")
        return None

    get_url = f"https://api.gladia.io/v2/transcription/{id}"
    headers = {
        "x-gladia-key": "f6698a07-6db5-4c37-84a4-9e4e85c71086"
    }

    # Polling loop to check the status
    while True:
        try:
            response = requests.get(get_url, headers=headers, timeout=60)
            response_data = response.json()
            #print("Polling Status Code:", response.status_code)
            #print("Polling Response:", response_data)

            if response_data.get('status') == 'done':
                break  # Exit loop if the status is 'done'

            print("Status is 'queued'. Retrying in 10 seconds...")
            time.sleep(10)  # Wait before the next poll
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while polling: {e}")
            time.sleep(10)  # Wait before retrying in case of an error

    # Final response output
    print("Gladia Response Data:", response_data)
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

def convert_to_custom_json(chunks):
    custom_json = []
    chunk_index = 1  # Start chunk_index at 1
    index_to_skip = None
    check_next = False
    should_increment = False
    length = len(chunks)
    first_sentance = ""
    second_sentance = ""

    try:
        for i, chunk in enumerate(chunks):
            remaining_sentences = length - (i + 1)  # Calculate remaining sentences
            if i == 0:
                first_sentance = chunk["text"]
                # logging.info(f"First Sentance Detected at index: {i}, :{first_sentance}")
            if i == 1:
                second_sentance = chunk["text"]

            if should_increment:
                chunk_index += 1
                should_increment = False

            speaker = chunk.get("speaker", "")
            text = chunk['text'].strip()

            if chunk["text"] == first_sentance:
                logging.info(f"First Sentance Detected at index: {i}, :{first_sentance}")
                if i != 0:
                    logging.info(f"Index to skip set: {index_to_skip}")
                    index_to_skip = chunk["chunk_index"]
                    #if chunks[i+1]['text'] == second_sentance and i+1 < length:
                    #    return custom_json


            # Check if there are more than 2 sentences left, then handle the chunk increment
            if remaining_sentences > 2:
                if i % 6 == 0 and text.endswith((".", "?", "!")) and i != 0:
                    should_increment = True
                elif i % 6 == 0 and not text.endswith((".", "?", "!")) and i != 0:
                    check_next = True
            else:
                # If there are 2 or fewer sentences left, do not increment the chunk index
                check_next = False
                should_increment = False
#
            if check_next:
                if text.endswith((".", "?", "!")):
                    should_increment = True
                    check_next = False

            if chunk["chunk_index"] == index_to_skip:
                logging.info(f"Skipping index: {index_to_skip}")
                continue

            custom_json.append({
                "sentence_index": i + 1,
                "chunk_index": chunk_index,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "speaker": speaker,
                "min_score": 0,
                "weighted_avg_score": 0
            })
    except Exception as e:
        logging.info(f"An error occurred while parsing chunks to custom json format: {e}")
        custom_json = chunks

    return custom_json


