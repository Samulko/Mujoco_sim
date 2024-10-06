import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import logging
import threading
import time
import ctypes
import queue
import sys

def set_alsa_params():
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(None)
    except:
        pass  # If it fails, we'll just continue without it

def reset_audio_stream(sample_rate=16000):
    sd.stop()
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    sd.default.dtype = 'int16'

def record_audio(filename, sample_rate=16000, max_duration=60):
    try:
        logging.info("Starting audio recording...")
        print("Recording... (Press Enter to stop)")
        
        q = queue.Queue()
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
                logging.warning(f"Recording status: {status}")
            q.put(indata.copy())
        
        def input_thread(event):
            input()
            event.set()
        
        event = threading.Event()
        thread = threading.Thread(target=input_thread, args=(event,))
        thread.start()
        
        reset_audio_stream(sample_rate)
        
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
            start_time = time.time()
            while not event.is_set():
                if (time.time() - start_time) > max_duration:
                    print("Maximum recording duration reached.")
                    logging.warning("Maximum recording duration reached.")
                    event.set()
                    break
                time.sleep(0.1)
        
        # Gather all recorded data
        recording = []
        while not q.empty():
            recording.append(q.get())

        if not recording:
            print("No audio recorded.")
            logging.warning("No audio recorded.")
            return None

        recording = np.concatenate(recording, axis=0)

        print("\nRecording finished.")
        logging.info("Audio recording completed.")

        # Check for silence or low volume
        rms = np.sqrt(np.mean(recording**2))
        logging.debug(f"Recording RMS value: {rms:.4f}")

        if rms < 0.005:
            print("Recording contains only silence.")
            logging.warning("Recording contains only silence.")
            return None
        elif rms < 0.02:
            print(f"Warning: Recording appears to be very quiet (RMS: {rms:.4f}).")
            logging.warning(f"Recording appears to be very quiet (RMS: {rms:.4f}).")
        else:
            logging.info(f"Recording RMS value: {rms:.4f}")

        wav.write(filename, sample_rate, recording)
        logging.info(f"Audio saved to {filename}")
        
        return filename
    except Exception as e:
        logging.error(f"Error in record_audio: {str(e)}", exc_info=True)
        print(f"An error occurred while recording audio: {str(e)}")
        return None
    finally:
        # Ensure audio stream is stopped and resources are released
        sd.stop()
        time.sleep(0.5)