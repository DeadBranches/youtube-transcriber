import configparser
import gc
import json
import os

import torch
import yt_dlp
from fastapi import FastAPI, Request
from pydub import AudioSegment

# Optional: Specify the location of the directory `Whisper` containing models
SETTINGS_FILE = "settings.ini"
config = configparser.ConfigParser()
config.read(SETTINGS_FILE)
WHISPER_CACHE = config.get("whisper", "cache_directory")


def load_transformers_model(model_name: str = "openai/whisper-large-v3"):
    """Load a a whisper model for inference using the faster transformers library."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if WHISPER_CACHE:
        os.environ["TRANSFORMERS_CACHE"] = WHISPER_CACHE

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


# TODO: Remove hard coded location
def load_whisper_model(model_name: str = "F:/C/cache/whisper/large-v2.pt"):
    import whisper

    if WHISPER_CACHE:
        os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[whisper] cuda is available: {torch.cuda.is_available()}")

    return whisper.load_model(model_name, device=device)


app = FastAPI()


@app.get("/transcribe/{url:path}")
async def transcribe_video(url: str, request: Request):
    """Takes a fully qualified URL as path argument.
    Example: http://localhost/transcribe/http://www.youtube.com/watch?v=f3f3f3f3f3"""

    query = request.url.query
    if query:
        url += "?" + query
    # https://stackoverflow.com/a/74555405

    audio_file = list()

    def postprocessor_event(event_info):
        status = event_info["status"]
        process = event_info["postprocessor"]
        print(status)
        if status == "error":
            print(f"error: {event_info['error']}")
        if status == "finished" and process == "MoveFiles":
            audio_file.append(event_info["info_dict"]["filepath"])
            print(f"{status} {process}: {event_info['info_dict']['filepath']}")

    # Whisper prefers a mono wav at 16k
    ydl_opts = {
        "extract_flat": "discard_in_playlist",
        "final_ext": "wav",
        "format": "bestaudio/best",
        "fragment_retries": 10,
        "ignoreerrors": True,
        "outtmpl": {"default": "%(title)s.%(ext)s"},
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "nopostoverwrites": False,
                "preferredcodec": "wav",
                "preferredquality": "5",
            },
        ],
        "restrictfilenames": True,
        "retries": 10,
        "trim_file_name": 15,
        "windowsfilenames": True,
        "postprocessor_hooks": [postprocessor_event],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])
        # Print the last item in audio_file list

        model = load_transformers_model() 
        if not os.path.exists(audio_file[-1]):
            raise FileNotFoundError(f"Audio file {audio_file[-1]} not found.")
        sound = AudioSegment.from_file(audio_file[-1])
        sound.export(audio_file[-1], format="wav", bitrate="16k")
        
        transcript = model(audio_file[-1])
        #transcript = model.transcribe(audio_file[-1])
        # result = json.dumps(transcript)
        result = transcript["text"]

        # The model seems to stay in memory afterwards.
        #   del model
        #torch.cuda.empty_cache()
        #gc.collect()
        # https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch

        # pipe = load_transformers_model()
        # result = pipe(audio_file[-1])
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8669)
