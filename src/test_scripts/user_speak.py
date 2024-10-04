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
        
        # Check if the recording is not just silence
        rms = np.sqrt(np.mean(recording**2))
        if rms > 0.01:  # Lowered threshold, adjust as needed
            # Normalize the recording to 16-bit range
            recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
            wav.write(filename, sample_rate, recording)
            logging.info(f"Audio saved to {filename}")
            return filename
        else:
            print(f"Warning: Recording appears to be very quiet (RMS: {rms:.4f}). It may not be transcribed accurately.")
            logging.warning(f"Recording appears to be very quiet (RMS: {rms:.4f}).")
            wav.write(filename, sample_rate, recording)
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
                    print("No audio recorded. Try again or enter 'q' to quit.")
                    if input().lower() == 'q':
                        break
                    continue

                # Transcribe audio using Whisper API
                transcription = transcribe_audio(recorded_file)
                print("Transcription:")
                print(transcription)

                # Save the transcription to a file
                with open(transcription_filename, "a") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {transcription}\n\n")
                logging.info(f"Transcription saved to {transcription_filename}")

                # Remove the temporary audio file
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logging.info(f"Temporary audio file {audio_filename} removed")

                print("Press Enter to record again, or 'q' to quit.")
                if input().lower() == 'q':
                    break

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
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory {temp_dir} removed")

if __name__ == "__main__":
    main()