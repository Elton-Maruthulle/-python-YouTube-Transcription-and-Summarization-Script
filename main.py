import whisper
import nltk
from pytube import YouTube
from pydub import AudioSegment
from gtts import gTTS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os


nltk.download('punkt')

'''
Add your youtube link below!!!!!!!
'''
youtube_link = ""


yt = YouTube(youtube_link)
audio_stream = yt.streams.filter(only_audio=True).first()
temp_audio_path = "temp_audio.mp4"

try:

    audio_stream.download(filename=temp_audio_path)
    print("Video downloaded successfully.")

    
    sound = AudioSegment.from_file(temp_audio_path, format="mp4")
    sound.export("1.mp3", format="mp3")
    print("Audio converted successfully.")

except Exception as e:
    print(f"Error: {e}")


if os.path.exists(temp_audio_path):
    os.remove(temp_audio_path)


model = whisper.load_model("base")
result = model.transcribe("1.mp3")

with open("transcription.txt", "w") as f:
    f.write(result["text"])


with open("transcription.txt", "r") as f:
    transcription_text = f.read()

parser = PlaintextParser.from_string(transcription_text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 2)  # You can adjust the number of sentences in the summary

with open("summary.txt", "w") as f:
    for sentence in summary:
        f.write(str(sentence) + "\n")


summary_text = ""
with open("summary.txt", "r") as f:
    summary_text = f.read()

tts = gTTS(summary_text, lang='en')
tts.save("summary.mp3")
print("Summary audio created successfully.")
