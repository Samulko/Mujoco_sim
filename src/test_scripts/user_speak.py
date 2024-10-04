import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from openai import OpenAI
from datetime import datetime
import logging
from dotenv import load_dotenv
import tempfile
import threading
import time
import sys

# Load environment variables from .env file
load_dotenv()

# Set up logging
log_dir = os.path.expanduser("~/Documents/GitHub/script_test/logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "user_speak.log")
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    logging.error("OPENAI_API_KEY not found in environment variables.")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_audio(filename, sample_rate=16000):
    try:
        logging.info("Starting audio recording...")
        print("Press Enter to start recording, press Enter again to stop.")
        
        recording = []
        is_recording = False
        
        def input_thread():
            nonlocal is_recording
            while True:
                input()
                is_recording = not is_recording
                if is_recording:
                    print("Recording... (Press Enter to stop)")
                else:
                    print("Recording stopped.")
                    break
        
        def audio_callback(indata, frames, time, status):
            if is_recording:
                recording.append(indata.copy())
                volume_norm = np.linalg.norm(indata) * 10
                print(f"Recording volume: {'#' * int(volume_norm)}", end='\r')
        
        threading.Thread(target=input_thread, daemon=True).start()
        
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
            while is_recording or not recording:
                time.sleep(0.1)
        
        if not recording:
            print("No audio recorded.")
            return None
        
        print("\nRecording finished.")
        logging.info("Audio recording completed.")
        
        # Concatenate all recorded chunks
        recording = np.concatenate(recording, axis=0)
        
        # Normalize the recording to 16-bit range
        if np.max(np.abs(recording)) > 0:
            recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
        else:
            print("Warning: Recording contains only silence.")
            logging.warning("Recording contains only silence.")
            return None
        
        wav.write(filename, sample_rate, recording)
        logging.info(f"Audio saved to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error in record_audio: {str(e)}", exc_info=True)
        print(f"An error occurred while recording audio: {str(e)}")
        return None

def transcribe_audio(filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            logging.info(f"Starting transcription of {filename} (Attempt {attempt + 1})")
            with open(filename, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            logging.info("Transcription completed successfully")
            return transcript.text
        except Exception as e:
            logging.error(f"Error in transcribe_audio (Attempt {attempt + 1}): {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                print(f"Transcription failed. Retrying... (Attempt {attempt + 2})")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print("Max retries reached. Transcription failed.")
                raise

def main():
    temp_dir = None
    silent_recordings = 0
    max_silent_recordings = 3
    try:
        # Use a temporary directory for audio files
        temp_dir = tempfile.mkdtemp()
        audio_filename = os.path.join(temp_dir, "recorded_audio.wav")
        logging.info(f"Temporary audio file will be saved as: {audio_filename}")
        
        # Set the transcriptions directory path
        transcriptions_dir = os.path.expanduser("~/Documents/GitHub/script_test/data/transcriptions")
        os.makedirs(transcriptions_dir, exist_ok=True)
        
        # Generate a unique filename for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcription_filename = os.path.join(transcriptions_dir, f"transcription_{timestamp}.txt")

        print(f"Transcriptions will be saved to {transcription_filename}")
        logging.info(f"Transcriptions will be saved to {transcription_filename}")

        while True:
            try:
                # Record audio and save it to WAV file
                recorded_file = record_audio(audio_filename)
                if recorded_file is None:
                    silent_recordings += 1
                    print(f"No valid audio recorded. ({silent_recordings}/{max_silent_recordings})")
                    if silent_recordings >= max_silent_recordings:
                        print(f"Maximum number of silent recordings ({max_silent_recordings}) reached. Exiting...")
                        break
                    print("Try again or enter 'q' to quit.")
                    if input().lower() == 'q':
                        print("Quitting the program...")
                        return
                    continue
                silent_recordings = 0  # Reset silent recordings counter

                # Transcribe audio using Whisper API
                try:
                    transcription = transcribe_audio(recorded_file)
                    print("Transcription:")
                    print(transcription)

                    # Save the transcription to a file
                    with open(transcription_filename, "a") as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {transcription}\n\n")
                    logging.info(f"Transcription saved to {transcription_filename}")
                except Exception as e:
                    print(f"Transcription failed: {str(e)}")
                    logging.error(f"Transcription failed: {str(e)}")

                # Remove the temporary audio file
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logging.info(f"Temporary audio file {audio_filename} removed")

                print("Press Enter to record again, or 'q' to quit.")
                user_input = input().lower()
                if user_input == 'q':
                    print("Quitting the program...")
                    return

            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}", exc_info=True)
                print(f"An error occurred. Please check the log file at {log_file} for details.")
                print("Press Enter to try again, or 'q' to quit.")
                if input().lower() == 'q':
                    break

        print(f"Thank you for using the voice-to-text assistant! Transcriptions have been saved to {transcription_filename}")
        logging.info("Script execution completed successfully")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"An error occurred. Please check the log file at {log_file} for details.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        logging.info("Program interrupted by user")
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory {temp_dir} removed")

if __name__ == "__main__":
    main()
