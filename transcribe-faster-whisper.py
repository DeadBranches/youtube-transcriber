import os
from faster_whisper import WhisperModel
from icecream import ic
from time import perf_counter


AUDIO_FOLDER = "F:/S/side-projects/youtube-transcriber/media/"
AUDIO_FILES = ["0d20220109153825p+12493855696.m4a", "1d20210908141649p4163600486.m4a"]

arguments = {

}


def transcribe():
    model = WhisperModel("F:/faster", device="cuda", compute_type="float16")

    audio_file = os.path.join(AUDIO_FOLDER, AUDIO_FILES[1])
    try:
        segments, info = model.transcribe(audio_file,     "language": "en",
    "initial_prompt": "Hello? Hi, is Kat there? This is Kat. Um, hi Kat.",
    "best_of": 5,
    "beam_size": 5,
    "word_timestamps": True,)
    except FileNotFoundError:
        print("The file isn't available where you thought it would be.")
        ic(f"Looked for: {audio_file}")
        ic(AUDIO_FILES)

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    # record start time
    time_start = perf_counter()
    # run the task
    transcribe()
    # calculate the duration
    time_duration = perf_counter() - time_start
    ic(arguments)
    # report the duration
    print(f"Took {time_duration:.3f} seconds")

# def test_stereo_diarization(data_dir="F:/S/side-projects/youtube-transcriber/media/"):
#     model = WhisperModel("F:/faster")

#     audio_path = os.path.join(data_dir, "0d20220109153825p+12493855696.m4a")
#     left = decode_audio(audio_path, split_stereo=True)

#     segments, _ = model.transcribe(left)
#     for segment in segments:
#         print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#     # transcription = "".join(segment.text for segment in segments).strip()
#     # assert transcription == (
#     #     "He began a confused complaint against the wizard, "
#     #     "who had vanished behind the curtain on the left."
#     # )


# test_stereo_diarization()
