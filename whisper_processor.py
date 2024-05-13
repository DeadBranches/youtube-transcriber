import os
import subprocess
import sys


def process_audio(wav_file, model_name="base.en"):
    """
    Processes an audio file using a specified model and returns the processed string.

    :param wav_file: Path to the WAV file
    :param model_name: Name of the model to use
    :return: Processed string output from the audio processing
    :raises: Exception if an error occurs during processing
    """

    model = f"./whisper/models/{model_name}_acft_q8_0.bin"

    # Check if the file exists
    if not os.path.exists(model):
        raise FileNotFoundError(
            f"Model file not found: {model} \n\nDownload a model with this command:\n\n> bash ./models/download-ggml-model.sh {model_name}\n\n"
        )

    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"WAV file not found: {wav_file}")

    full_command = f"./whisper/main -m {model} -f {wav_file} -nt -t 1 -np -ng -nf -bo 1 -bs 1 -ac 512"

    # Execute the command
    process = subprocess.Popen(
        full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Get the output and error (if any)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error processing audio: {error.decode('utf-8')}")

    # Process and return the output string
    decoded_str = output.decode("utf-8").strip()
    processed_str = decoded_str.replace("[BLANK_AUDIO]", "").strip()

    return processed_str


def main():
    if len(sys.argv) >= 2:
        wav_file = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) == 3 else "base.en"
        try:
            result = process_audio(wav_file, model_name)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python whisper_processor.py <wav_file> [<model_name>]")


if __name__ == "__main__":
    main()
