# Speech_to_speech
A app that can translate the voice to many different language voice.<br>
For example, what I did is translate your record(English) to Spanish, Chinese, Japanese.<br>
The cool thing is that you can even make your own voice project, so that you can hear that your voice be translated to different language but are all your own voice and style!<br>
To finish this app, we use two AI model and Gradio.<br>
### AssemblyAI
It can transcribe spoken words(no chinese service) into written text<br>
A simple code here. It trascribe the mp3 you suply into text.<br>
```python
import assemblyai as aai

aai.settings.api_key = "AssemblyAI_API_Key"

audio_url = "https://your_mp3_url"

transcriber = aai.Transcriber()

transcript = transcriber.transcribe(audio_url)

print(transcript.text)
```
This help me to solve STT.<br>
[AssemblyAI](https://www.assemblyai.com/dashboard/signup)
[How_to_use](https://www.assemblyai.com/docs/speech-to-text/speech-recognition)
### elevenlabs.io
A very useful AI.<br>
It can text-to-speech, speech-to-speech, sound effect, audio isolation.<br>
In this app, we use its text-to-sppech.<br>
Here is an example code, it can set the changing voice's similarity, style, and stability.<br>
```python
# Import necessary libraries
import requests  # Used for making HTTP requests
import json  # Used for working with JSON data

# Define constants for the script
CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
XI_API_KEY = "<xi-api-key>"  # Your API key for authentication
VOICE_ID = "<voice-id>"  # ID of the voice model to use
TEXT_TO_SPEAK = "<text>"  # Text you want to convert to speech
OUTPUT_PATH = "output.mp3"  # Path to save the output audio file

# Construct the URL for the Text-to-Speech API request
tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

# Set up headers for the API request, including the API key for authentication
headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY
}

# Set up the data payload for the API request, including the text and voice settings
data = {
    "text": TEXT_TO_SPEAK,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": True
    }
}

# Make the POST request to the TTS API with headers and data, enabling streaming response
response = requests.post(tts_url, headers=headers, json=data, stream=True)

# Check if the request was successful
if response.ok:
    # Open the output file in write-binary mode
    with open(OUTPUT_PATH, "wb") as f:
        # Read the response in chunks and write to the file
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
    # Inform the user of success
    print("Audio stream saved successfully.")
else:
    # Print the error message if the request was not successful
    print(response.text)
```
[elevenlabs.io](https://elevenlabs.io/app/speech-synthesis/text-to-speech)
[How_to_use](https://elevenlabs.io/docs/api-reference/getting-started)
# Reference
[Youtube AssemblyAI](https://www.youtube.com/watch?v=ZduW0N31JuE)
