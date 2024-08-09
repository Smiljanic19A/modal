import subprocess
import time
import json
import os

import Levenshtein
import pysrt
import whisperx
import webvtt
import requests

REMOTE_PATH = "C:/Users/aleks/Downloads/modal/audio"  # Ensure this is the correct path

def srt_to_text(file_path):
    subs = pysrt.open(file_path)
    text = ""
    for sub in subs:
        text += sub.text + "\n"
    return text.strip()
def vtt_to_text(file_path):
    text = ""
    for caption in webvtt.read(file_path):
        print(caption)
        text += caption.text + "\n"
    return text.strip()
def compare_paragraphs(string1, string2):
    distance = Levenshtein.distance(string1, string2)

    max_length = max(len(string1), len(string2))
    return (1 - distance / max_length) * 100
# Make and return a gladia transcription
def transcribe_audio(audio_url):
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
        print("Status Code:", response.status_code)
        response_data = response.json()  # Parse the JSON response
        print("Response:", response_data)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

    # Extract result_url and id if they exist
    result_url = response_data.get('result_url')
    id = response_data.get('id')

    if result_url:
        print("Result URL:", result_url)
    if id:
        print("Result ID:", id)

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
            print("Polling Status Code:", response.status_code)
            print("Polling Response:", response_data)

            if response_data.get('status') == 'done':
                break  # Exit loop if the status is 'done'

            print("Status is 'queued'. Retrying in 10 seconds...")
            time.sleep(10)  # Wait before the next poll
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while polling: {e}")
            time.sleep(10)  # Wait before retrying in case of an error

    # Final response output
    print("Final Response Data:", response_data)
    return response_data
#Remove All cyrilic charachters for distance calculation
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
#Parse the VTT file made in modal, and return it in a 1 line string
def parse_webvtt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        parsed_text = []

        for line in lines:
            # Check if the line contains a timestamp (format: HH:MM:SS.MMM --> HH:MM:SS.MMM)
            if '-->' in line:
                continue  # Skip the timestamp lines

            # Skip the "WEBVTT" header and any blank lines
            if line.strip() == "WEBVTT" or line.strip() == "":
                continue

            # Append the line to the parsed text
            parsed_text.append(line.strip())

        # Join the parsed text list into a single line string with spaces
        combined_text = ' '.join(parsed_text)

        # Convert Cyrillic to Latin
        return cyrillic_to_latin(combined_text)

    except FileNotFoundError:
        return "File does not exist."
#calculate levenshtein distance
def calc_levenshtein_distance(string1, string2):
    distance = Levenshtein.distance(string1, string2)

    max_length = max(len(string1), len(string2))
    return (1 - distance / max_length) * 100

def write_alignment_to_json(alignment_segments, path):
    with open(path, 'w') as f:
        json.dump(alignment_segments, f, indent=4)


def write_diarization_to_json(diarization_segments, path):
    with open(path, 'w') as f:
        json.dump(diarization_segments, f, indent=4)


def write_vtt(segments, path):
    with open(path, 'w', encoding="utf-8") as f:
        f.write("WEBVTT\n")
        for segment in segments:
            f.write(f"{segment['start']} --> {segment['end']}\n{segment['text']}\n\n")


def write_srt(segments, path):
    with open(path, 'w', encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            f.write(f"{i}\n{segment['start']} --> {segment['end']}\n{segment['text']}\n\n")


def whisper_function():
    # Construct the full path programmatically
    audio_file = os.path.join(REMOTE_PATH, "test.mp3")
    audio_file = os.path.abspath(audio_file)

    device = "cpu"
    batch_size = 4
    compute_type = "int8"

    print("Starting transcription")
    start_time_transcription = time.time()
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="sr")

    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Check if the file exists and print the absolute path
    print(f"Checking if file exists at: {audio_file}")
    if not os.path.isfile(audio_file):
        print(f"File does not exist: {audio_file}")

        # Print the contents of the directory to verify
        print(f"Contents of directory {REMOTE_PATH}:")
        print(os.listdir(REMOTE_PATH))

        return

    try:
        print(f"Loading audio file: {audio_file}")
        audio = whisperx.load_audio(audio_file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error loading audio: {e}")
        return

    result = model.transcribe(audio, batch_size=batch_size)
    end_time_transcription = time.time()
    #Make a pure string for comparison purposes
    audio_transcription_txt = cyrillic_to_latin(" ".join(segment["text"] for segment in result["segments"]))
    print(f"Transcribed: {audio_transcription_txt}")
    print(f"result: {result}")

    metadata_path = os.path.join(REMOTE_PATH, "transcription_result.json")
    with open(metadata_path, "w") as f:
        json.dump(result["segments"], f, indent=4)

    #print("Starting alignment")
    #start_time_alignment = time.time()
    #model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    #result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    #end_time_alignment = time.time()
    #pure_text = " ".join(segment["text"] for segment in result["segments"])
    #print(result["segments"])

    # Write VTT and SRT files
    print("Creating VTT and SRT subtitle files")
    write_vtt(result["segments"], os.path.join(REMOTE_PATH, "transcription_result.vtt"))
    write_srt(result["segments"], os.path.join(REMOTE_PATH, "transcription_result.srt"))

    print(f"VTT TO TEXT: {parse_webvtt(REMOTE_PATH+"/"+"transcription_result.vtt")}")
    distance = calc_levenshtein_distance(audio_transcription_txt, parse_webvtt(REMOTE_PATH+"/transcription_result.vtt"))
    print(f"distance {distance}")

    print("Starting diarization")
    start_time_diarization = time.time()
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    end_time_diarization = time.time()
    print(result["segments"])
    write_diarization_to_json(result["segments"], os.path.join(REMOTE_PATH, "diarization_data.json"))

    print(f"Time taken for Transcription: {end_time_transcription - start_time_transcription} seconds")
    #print(f"Time taken for Alignment: {end_time_alignment - start_time_alignment} seconds")
    print(f"Time taken for Diarization: {end_time_diarization - start_time_diarization} seconds")



whisper_function()
