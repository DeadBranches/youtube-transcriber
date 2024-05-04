import os
from glob import glob
import shutil
import subprocess
import argparse
import ffmpeg
import soundfile as sf
from typing import Callable, Dict, List, Tuple
from pydub import AudioSegment
from icecream import ic
import torch
import numpy as np

# For audio transcription
import whisper
from functools import partial
import io


VIDEO_DIRECTORY = "./video"
AUDIO_DIRECTORY = "./audio"
TRANSCRIPT_DIRECTORY = "./transcript"
ARCHIVE_DIRECTORY = "./archive"
WHISPER_CACHE = "F:/C/cache"  # If you have your models in a non-standard
WHISPER_MODEL = "F:/C/cache/whisper/large-v2.pt"
# location, you can set this to the path


### Helper Functions


def verify_folder_structure(video: str, audio: str, text=str) -> None:
    """
    Create the temp directories intended to hold the files during processing
    if they don't already exist
    """

    if not os.path.exists(audio):
        os.makedirs(audio)
    if not os.path.exists(video):
        os.makedirs(audio)
    if not os.path.exists(text):
        os.makedirs(text)
    return None


def fetch_resource(
    video_url: str,
):
    """Download audio from a video resource located online"""

    subprocess.run(
        [
            "C:/bin/yt-dlp.exe",
            "--restrict-filenames",
            "--windows-filenames",
            "--trim-filenames",
            "15",
            "--ignore-errors",
            "--extract-audio",
            "--audio-format",
            "wav",
            "-o",
            f"{AUDIO_DIRECTORY}/%(title)s.%(ext)s",
            video_url,
        ],
        check=True,
    )


def move_to_archive(file_name: str, archive_path: str = ARCHIVE_DIRECTORY):
    """Move a file to the archive folder"""

    shutil.copy2(file_name, archive_path)


def cleanup(video_file: str, audio_file: str):
    move_to_archive(video_file)
    move_to_archive(audio_file)


def process_videos() -> None:
    """
    Loop through all the videos in the video directory
    Extract the audio from each video and save it to the audio directory
    Move the video to the archive directory
    Return a string with the name of the video that was processed.
    If the audio already exists, return None.

    Note:
    - The audio is saved in .wav format.
    - The audio is 16kHz.
    - The audio is mono.
    - The audio is in the range of [-1, 1].

    Example:
    process_videos()

    """
    for video in glob(os.path.join(VIDEO_DIRECTORY, "*")):
        filename, file_extension = os.path.splitext(os.path.basename(video))

        if os.path.exists(f"{AUDIO_DIRECTORY}/{filename}.wav"):
            continue  # Don't extract audio more than once.

        subprocess.call(
            f"ffmpeg -i {video} -ac 1 -ar 16000 {AUDIO_DIRECTORY}/{filename}.wav",
            shell=True,
        )

        print(f"{video} >>> {AUDIO_DIRECTORY}/{filename}.wav")

    return None


def process_audio_folder() -> None:
    """
    Loop through all the audio files in the audio directory
    Transcribe the audio
    Create a text file with the transcription
    Move the audio file to the archive directory
    """

    if WHISPER_CACHE:
        os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    ic(torch.cuda.is_available())
    ic(type(model))

    for audio in glob(os.path.join(AUDIO_DIRECTORY, "*")):
        filename, file_extension = os.path.splitext(os.path.basename(audio))

        if os.path.exists(f"{TRANSCRIPT_DIRECTORY}/{filename}.txt"):
            continue  # Don't transcribe the audio if a transcript already exists.

        # Transcribe the audio file
        sound = AudioSegment.from_file(filename)
        sound.export(filename, format="wav", bitrate="16k")
        transcription = transcribe_audio(audio, model)

        with open(
            f"{TRANSCRIPT_DIRECTORY}/{filename}.txt", "w", encoding="utf8"
        ) as file:
            file.write(transcription)

        print(f"Transcribed audio to {filename}.txt")
        move_to_archive(audio)


def transcribe_audio(audio_file: str, model) -> str:
    """
    Transcribe the audio file using whisper.
    Return a string with the transcription.
    """
    # Whisper models can be stored in a non-standard directory by setting WHISPER_CACHE

    sound = AudioSegment.from_file(audio_file)
    sound = AudioSegment.from_raw
    sound.export(audio_file, format="wav", bitrate="16k")

    transcript = model.transcribe(audio_file)

    return transcript["text"]


AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi")


def get_file_list(folder: str) -> List:
    """Get files in folder and all subfolders"""
    try:
        return glob(os.path.join(folder, "*"))
    except Exception as e:
        ic(e)
        return []


def get_filepath(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_filename(path: str) -> str:
    return os.path.basename(path)


def get_extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_audio(path: str, audio_extension: Tuple):
    return path.endswith(audio_extension)


def is_video(path: str, extension_list: Tuple):
    return path.endswith(extension_list)


def is_media_file(path: str):
    return is_audio(path, AUDIO_EXTENSIONS) or is_video(path, VIDEO_EXTENSIONS)


def get_file_list(folder: str) -> List:
    """Given a folder, return a list of files in that folder and all subfolders."""
    try:
        return glob(os.path.join(folder, "*"))
    except Exception:
        print(
            "Media folder missing.\nPut your files in ./media or pass the folder\nas an argument to --folder."
        )


def build_list_of_files_to_process(
    file_list: List, audio_callback: Callable, video_callback: Callable
) -> List[Dict]:
    """
    End up with a list of dictionaries:
        {
            "file_path": "media/media_file.mp4",
            "filename": "media_file",
            "extension": ".mp4",
            "operations": ["extract_audio", "transcribe"],
            "transcription": "",
        }
    """
    list_of_files_to_process = []
    template_dict = {
        "file_path": "",
        "filename": "",
        "extension": "",
        "operations": [],
        "transcription": "",
    }
    for path in file_list:
        list_entry = template_dict.copy()
        list_entry["file_path"] = get_filepath(path)
        list_entry["filename"] = get_filename(path)
        list_entry["extension"] = get_extension(path)
        list_entry["operations"] = (
            ["extract_audio", "transcribe"] if video_callback(path) else ["transcribe"]
        )
        ic(list_entry)
        list_of_files_to_process.append(list_entry)
        ic(list_of_files_to_process)
    return list_of_files_to_process  # Add this line to return the list


def main():
    """By default when no arguments are passed to the script, the script
    will process videos, then process audio into transcriptions.
    If the -d flag is passed with a URL, the script will download the
    video and process the audio from the video.
    Arguments parsed using argparse:"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-f",
        "--folder",
        default="media",
        help="Folder containing media files to transcribe. Defaults to './media'",
        type=str,
    )
    arg_parser.add_argument(
        "-d",
        "--download",
        help="Download the video from the URL",
        type=str,
    )
    args = arg_parser.parse_args()

    MEDIA_FOLDER = args.folder if args.folder else "media"

    if args.download:
        fetch_resource(args.download)

    audio_callback = partial(is_audio, audio_extension=AUDIO_EXTENSIONS)
    video_callback = partial(is_video, extension_list=VIDEO_EXTENSIONS)

    files: List = get_file_list(MEDIA_FOLDER)
    media_files = list(filter(is_media_file, files))

    list_of_media_files = build_list_of_files_to_process(
        media_files, audio_callback, video_callback
    )

    if WHISPER_CACHE:
        os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    ic(torch.cuda.is_available())
    ic(type(model))

    # sound = AudioSegment.from_file(filename)
    # sound.export(filename, format="wav", bitrate="16k")

    # return transcript["text"]
    # with open(
    #     f"{TRANSCRIPT_DIRECTORY}/{filename}.txt", "w", encoding="utf8"
    # ) as file:
    #     file.write(transcription)

    # print(f"Transcribed audio to {filename}.txt")
    # move_to_archive(audio)

    # Iterate over each list item in list_of_media_files
    for item in list_of_media_files:
        in_filepath = item["file_path"]
        in_filename = item["filename"]
        for operation in item["operations"]:
            if operation == "extract_audio":
                try:
                    out, err = (
                        ffmpeg.input(in_filepath)
                        .output("-", format="wav", acodec="pcm_s16le", ac=1, ar="16000")
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    bytes_io = io.BytesIO(out)
                    audio, samplerate = sf.read(
                        bytes_io, dtype="float32", always_2d=True
                    )
                    # audio, samplerate = sf.read(io.BytesIO(out), format="wav")
                    # .output("-", format="wav")
                    # .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar="16k")
                    ic(type(audio))
                except ffmpeg.Error as e:
                    ic("stdout:", e.stdout.decode("utf8"))
                    ic("stderr:", e.stderr.decode("utf8"))
                    raise e
                # ic(out)
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                transcript = model.transcribe(audio)
                ic(transcript["text"])
                # out.export(f"F:\\{in_filename}.wav", format="wav")

        # print(f"{video} >>> {AUDIO_DIRECTORY}/{filename}.wav")
        #     elif operation == "transcribe":
        #         process_audio_folder()
        #     else:
        #         print(f"Unknown operation: {operation}")

    ic(list_of_media_files)


if __name__ == "__main__":
    main()
