from openai import OpenAI
import os
import logging
import tempfile
import backoff
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

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
        
        # Play the audio using simpleaudio
        try:
            audio = AudioSegment.from_mp3(temp_audio_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(44100)  # Set sample rate to 44.1 kHz
            
            # Convert to raw PCM data
            raw_data = audio.raw_data
            num_channels = audio.channels
            bytes_per_sample = audio.sample_width
            sample_rate = audio.frame_rate

            # Play audio
            play_obj = sa.play_buffer(raw_data, num_channels, bytes_per_sample, sample_rate)
            play_obj.wait_done()
            
            logging.info("Text-to-speech playback completed")
        except Exception as e:
            logging.error(f"Error playing audio: {str(e)}", exc_info=True)
            print(f"An error occurred during audio playback: {str(e)}")
        
        # Remove the temporary file
        os.remove(temp_audio_path)
        
    except Exception as e:
        logging.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
        raise