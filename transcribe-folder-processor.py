import argparse
import gc
import os
import shutil
import subprocess
from glob import glob

import torch

# For audio transcription
import whisper
from pydub import AudioSegment

VIDEO_DIRECTORY = "./video"
AUDIO_DIRECTORY = "./audio"
TRANSCRIPT_DIRECTORY = "./transcript"
ARCHIVE_DIRECTORY = "./archive"
WHISPER_CACHE = "F:/C/cache"  # If you have your models in a non-standard
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
            "C:/bin/yt-dlp/yt-dlp.exe",
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


def load_whisper_model(model_name: str = "F:/C/cache/whisper/large-v2.pt"):
    if WHISPER_CACHE:
        os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[whisper] cuda is available: {torch.cuda.is_available()}")

    return whisper.load_model(model_name, device=device)


def process_audio_folder() -> None:
    """
    Loop through all the audio files in the audio directory
    Transcribe the audio
    Create a text file with the transcription
    Move the audio file to the archive directory
    """

    model = load_whisper_model()

    for audio in glob(os.path.join(AUDIO_DIRECTORY, "*")):
        filename, file_extension = os.path.splitext(os.path.basename(audio))

        if os.path.exists(f"{TRANSCRIPT_DIRECTORY}/{filename}.txt"):
            print(
                f"[audio transcriber] Transcript already found for {filename}. Skipping"
            )
            continue  # Don't transcribe the audio if a transcript already exists.

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

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file {audio_file} not found.")

    sound = AudioSegment.from_file(audio_file)
    sound.export(audio_file, format="wav", bitrate="16k")

    transcript = model.transcribe(audio_file)

    return transcript["text"]


def main():
    """By default when no arguments are passed to the script, the script
    will process videos, then process audio into transcriptions.
    If the -d flag is passed with a URL, the script will download the
    video and process the audio from the video.
    Arguments parsed using argparse:"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-d", "--download", help="Download the video from the URL", type=str
    )
    args = arg_parser.parse_args()

    verify_folder_structure(
        video=VIDEO_DIRECTORY, audio=AUDIO_DIRECTORY, text=TRANSCRIPT_DIRECTORY
    )

    if args.download:
        fetch_resource(args.download)
        process_audio_folder()
    else:
        process_videos()
        process_audio_folder()


if __name__ == "__main__":
    main()
