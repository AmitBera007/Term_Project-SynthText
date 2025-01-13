import os.path
import pdfplumber
from gtts import gTTS
import gtts
from pydub import AudioSegment


def check_file_exists(path: str) -> bool:
    if not os.path.isfile(path):
        print("File not exist.")
        return False
    return True


def get_filename_and_extension(path: str) -> tuple:
    return os.path.splitext(path)


def check_file_extension(path: str) -> bool:
    extensions_lst = ['.pdf', '.txt']
    extension = get_filename_and_extension(path)[1]
    if extension not in extensions_lst:
        print("Check file extension.")
        return False
    return True


def check_language(lang: str) -> bool:
    lang_lst = gtts.lang.tts_langs().keys()
    if lang not in lang_lst:
        print("Incorrect language.")
        return False
    return True


def read_file(path: str) -> str:
    _, extension = get_filename_and_extension(path)

    if extension == '.pdf':
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            text = ''.join(pages)
            text = text.replace(chr(312), 'к')  # Fix for specific Russian letter issue

    elif extension == '.txt':
        with open(path, 'r') as file:
            text = file.read()

    else:
        text = ''
    text = text.replace('\n', '')
    print("Processing...")
    return text


def change_speed(input_file: str, output_file: str, speed: float) -> None:
    """Modify the speed of an MP3 file."""
    audio = AudioSegment.from_file(input_file)
    # Change speed (higher speed results in shorter duration)
    audio = audio.speedup(playback_speed=speed)
    audio.export(output_file, format="mp3")
    print(f"Adjusted speed and saved as {output_file}")


def convert_file_to_mp3(path: str, lang='en', speed=1.0) -> None:
    if not check_file_exists(path):
        return
    if not check_file_extension(path):
        return
    text = read_file(path)
    if not check_language(lang):
        return

    filename, _ = get_filename_and_extension(path)
    temp_file = filename + '.mp3'
    final_file = filename + '_speed.mp3'

    # Generate TTS MP3
    tts = gTTS(text, lang=lang)
    tts.save(temp_file)
    print("TTS file created successfully.")

    # Adjust speed if needed
    if speed != 1.0:
        change_speed(temp_file, final_file, speed)
        os.remove(temp_file)  # Clean up temporary file
    print("File was saved successfully.")


if __name__ == '__main__':
    print('Input file path:')
    path = input()
    print('Input language:')
    lang = input()
    print('Input speech speed (e.g., 1.0 for normal, 1.5 for faster, 0.75 for slower):')
    speed = float(input())
    convert_file_to_mp3(path, lang, speed)
