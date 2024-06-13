# # app.py (Flask backend)
# import numpy as np
# import torch
# import asyncio
# import base64
# import json
# import pyaudio
# from flask import Flask, jsonify, request, render_template
# import base64
# import json
# from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
# app = Flask(__name__)

# # Load the pre-trained model and processor
# model_path = "sahilkadge/whisper-small-en-sahil-kamran"  # Update the path to your model directory
# processor = WhisperProcessor.from_pretrained(model_path)
# model = WhisperForConditionalGeneration.from_pretrained(model_path)

# # Function to process audio data using the loaded model
# def process_audio(audio_data):
#     inputs = processor(audio_data, return_tensors="pt", padding=True, sampling_rate=16000)
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)
#     # print(transcription,123)
#     return transcription

# # Route for receiving audio data and sending transcription
# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     # try:
#     #     content = request.json
#     #     print("content",content)
#     #     audio_data_base64 = content['audio_data']
#     #     print("audio",audio_data_base64)
#     #     # Decode base64 encoded audio data
#     #     audio_data_bytes = base64.b64decode(audio_data_base64)
        
#     #     # Process audio data
#     #     transcription = process_audio(audio_data_bytes)
        
#     #     return jsonify({'transcription': transcription})
#     try:
#         content = request.json
#         audio_data_base64 = content['audio_data']
        
#         # Decode base64 encoded audio data
#         audio_data_bytes = base64.b64decode(audio_data_base64)
        
#         # Convert bytes to array of numbers
#         audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)
#         transcription = process_audio(audio_array)
#         print(transcription,321)
#         # Process audio data (replace this with your processing logic)
#         # For example, you can print the audio array
#         print("Audio Array:", audio_array)
        
#         # Return response
#         return jsonify({'message': 'Audio data received and processed successfully'})
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({'error': 'An error occurred during audio processing'}), 500

    


# if __name__ == '__main__':
#     app.run(debug=True)



import numpy as np
import torch  # Add this line to import torch
import base64
from flask import Flask, jsonify, request, render_template
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
import pyaudio

app = Flask(__name__)

# Load the pre-trained model and processor
model_path = "sahilkadge/whisper-small-en-sahil-kamran"  
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

# Define parameters for audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Function to process audio data using the loaded model
def process_audio(audio_data):
    try:
        # Extract audio features using the processor
        audio_features = feature_extractor(audio_data, sampling_rate=RATE, return_tensors="pt")

        # Pad the audio features to the correct length
        padding_length = 3000 - audio_features.input_values.shape[-1]
        if padding_length > 0:
            audio_features.input_values = torch.nn.functional.pad(audio_features.input_values, (0, padding_length))

        # Perform inference using the model
        output_ids = model.generate(input_features=audio_features.input_values)

        # Decode the predicted IDs
        transcription = processor.decode(output_ids[0], skip_special_tokens=True)
        return transcription
    except Exception as e:
        print("Error:", e)
        return "An error occurred during audio processing"

# Route for receiving audio data and sending transcription
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        content = request.json
        audio_data_base64 = content['audio_data']
        
        # Decode base64 encoded audio data
        audio_data_bytes = base64.b64decode(audio_data_base64)
        
        # Convert bytes to array of numbers (assuming int16 audio)
        audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)
        
        # Process audio data
        transcription = process_audio(audio_array)
        
        # Return response
        return jsonify({'transcription': transcription})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'An error occurred during audio processing'}), 500

if __name__ == '__main__':
    app.run(debug=True)
