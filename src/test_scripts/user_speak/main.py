import os
import logging
from datetime import datetime
import tempfile
from collections import deque
import time
import sys
import signal
import shutil
from dotenv import load_dotenv

from .audio_utils import set_alsa_params, record_audio
from .openai_utils import transcribe_audio, generate_response, text_to_speech

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

TIMEOUT = 60  # 60 seconds timeout for recording

def timeout_handler(signum, frame):
    raise TimeoutError("Recording timed out")

def handle_conversation(audio_filename, transcription_filename, conversation_history):
    try:
        # Record audio and save it to WAV file
        recorded_file = record_audio(audio_filename)
        
        if recorded_file is None:
            print("No valid audio recorded. Please try again.")
            return False

        # Transcribe audio using Whisper API
        transcription = transcribe_audio(recorded_file)
        if not transcription:
            print("Transcription failed. Please try again.")
            return False
        
        print("Transcription:")
        print(transcription)

        # Save the transcription to a file
        with open(transcription_filename, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {transcription}\n\n")
        logging.info(f"Transcription saved to {transcription_filename}")

        # Generate response
        response = generate_response(transcription, conversation_history)
        if response:
            print("Response:")
            print(response)
            
            # Save the response to the transcription file
            with open(transcription_filename, "a") as f:
                f.write(f"Response: {response}\n\n")
            
            # Convert response to speech and play it
            text_to_speech(response)

            # Update conversation history
            conversation_history.append({"role": "user", "content": transcription})
            conversation_history.append({"role": "assistant", "content": response})
        else:
            print("Failed to generate a response. Please try again.")

        return True

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        logging.error(f"Error during processing: {str(e)}", exc_info=True)
        return False

def main():
    set_alsa_params()
    temp_dir = None
    conversation_history = deque(maxlen=5)  # Store last 5 exchanges
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
                # Set up the timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TIMEOUT)

                print("Press Enter to start recording, or 'q' to quit.")
                user_input = input().lower()
                if user_input == 'q':
                    print("Quitting the program...")
                    break

                success = handle_conversation(audio_filename, transcription_filename, conversation_history)
                
                # Cancel the alarm
                signal.alarm(0)

                if not success:
                    print("Try again or enter 'q' to quit.")
                    if input().lower() == 'q':
                        print("Quitting the program...")
                        break

                # Remove the temporary audio file
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logging.info(f"Temporary audio file {audio_filename} removed")
                
                # Add a small delay before the next recording attempt
                time.sleep(1)

            except TimeoutError:
                print("Recording timed out. Please try again.")
                logging.warning("Recording timed out")
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nRecording interrupted by user.")
                logging.info("Recording interrupted by user")
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logging.info(f"Temporary audio file {audio_filename} removed")
                print("Press Enter to record again, or 'q' to quit.")
                if input().lower() == 'q':
                    print("Quitting the program...")
                    break
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}", exc_info=True)
                print(f"An error occurred. Please check the log file at {log_file} for details.")
                print("Press Enter to try again, or 'q' to quit.")
                if input().lower() == 'q':
                    break
                time.sleep(1)

        print(f"Thank you for using the voice-to-text assistant! Transcriptions have been saved to {transcription_filename}")
        logging.info("Script execution completed successfully")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"An error occurred. Please check the log file at {log_file} for details.")
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory {temp_dir} removed")

if __name__ == "__main__":
    main()
