# CODE FOR REAL TIME TRANSCRIPTION 
import pyaudio
import numpy as np
import io
from transformers import pipeline

# Initialize the transcription pipeline
transcriber = pipeline(model="sahilkadge/whisper-small-en-sahil-kamran")

# Define parameters for audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust as needed

# Create a PyAudio instance
p = pyaudio.PyAudio()

# Open the microphone for audio capture
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* Recording audio...")

# Continuously stream audio to the transcriber
while True:
    data = stream.read(CHUNK)
    audio_chunk = np.frombuffer(data, dtype=np.int16)  # Convert raw audio data to NumPy ndarray
    
    # Perform transcription on the audio chunk
    transcription = transcriber(audio_chunk)
    
    # Output the transcription result
    print("Transcription:", transcription)

# Stop and close the audio stream
stream.stop_stream()
stream.close()
p.terminate()




# import pyaudio
# import numpy as np
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# # Load the fine-tuned model and processor
# model_name = "sahilkadge/whisper-small-en-sahil-kamran"
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)

# # Define parameters for audio capture
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 1024

# # Create a PyAudio instance
# p = pyaudio.PyAudio()

# # Open the microphone for audio capture
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# print("* Recording audio...")

# # Continuously stream audio to the transcriber
# while True:
#     data = stream.read(CHUNK)
#     audio_chunk = np.frombuffer(data, dtype=np.int16)  # Convert raw audio data to NumPy ndarray
    
#     # Tokenize the audio chunk
#     inputs = processor(audio_chunk, return_tensors="pt", padding=True)

#     # Perform transcription with the model
#     with torch.no_grad():
#         output = model(**inputs)

#     # Decode the transcription output
#     transcription = processor.batch_decode(output["logits"], skip_special_tokens=True)
    
#     # Output the transcription result
#     print("Transcription:", transcription)

# # Stop and close the audio stream
# stream.stop_stream()
# stream.close()
# p.terminate()
