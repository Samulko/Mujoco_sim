from openai import OpenAI
import os
import logging
import tempfile
import subprocess
import time
import backoff

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def transcribe_audio(filename):
    try:
        logging.info(f"Starting transcription of {filename}")
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        logging.info("Transcription completed successfully")
        return transcript.text
    except Exception as e:
        logging.error(f"Error in transcribe_audio: {str(e)}", exc_info=True)
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_response(prompt, conversation_history):
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}", exc_info=True)
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name
        
        # Try to play the audio using mpg123
        try:
            subprocess.run(["mpg123", "-q", temp_audio_path], check=True)
            logging.info("Text-to-speech playback completed")
        except FileNotFoundError:
            print("Error: mpg123 not found. Please install mpg123 to enable audio playback.")
            print("You can install it on Ubuntu/Debian with: sudo apt-get install mpg123")
            print("For other systems, please refer to your package manager or mpg123 website.")
            logging.error("mpg123 not found on the system")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error playing audio: {str(e)}", exc_info=True)
            print(f"An error occurred during audio playback: {str(e)}")
        
        # Remove the temporary file
        os.remove(temp_audio_path)
        
    except Exception as e:
        logging.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
        raise
