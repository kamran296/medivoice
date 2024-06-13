import io
import os
import pyaudio
import wave
import requests
from flask import Flask, request, render_template

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/sahilkadge/whisper-small-en-sahil-kamran"
HEADERS = {"Authorization": "Bearer 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"}

@app.route('/')
def index():
    return render_template('final.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Check if audio file is present in the request
        if 'audio_file' not in request.files:
            return "No audio file found", 400

        audio_file = request.files['audio_file']

        # Check if the file is empty
        if audio_file.filename == '':
            return "Empty file name", 400

        # Save the audio file
        audio_file.save('recorded_audio.wav')

        # Stream audio data to the API for transcription
        response = stream_audio('recorded_audio.wav')
        print(response)
        transcription = response.get("transcription", response.get("text", "Transcription not available"))

        return transcription
    except Exception as e:
        return str(e), 500

def stream_audio(audio_file):
    try:
        with open(audio_file, "rb") as f:
            response = requests.post(API_URL, headers=HEADERS, data=f.read())
            return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)



