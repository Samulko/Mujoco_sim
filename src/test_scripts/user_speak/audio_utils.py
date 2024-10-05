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
        print("Press Enter to start recording, press Enter again to stop.")
        
        q = queue.Queue()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())
        
        def input_thread(event):
            input()
            event.set()
        
        event = threading.Event()
        thread = threading.Thread(target=input_thread, args=(event,))
        thread.start()
        
        reset_audio_stream(sample_rate)
        
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
            print("Recording... (Press Enter to stop)")
            thread.join()
        
        recording = []
        
        try:
            while True:
                recording.append(q.get_nowait())
        except queue.Empty:
            pass
        
        recording = np.concatenate(recording, axis=0)
        
        if len(recording) == 0:
            print("No audio recorded.")
            return None
        
        print("\nRecording finished.")
        logging.info("Audio recording completed.")
        
        # Normalize the recording to 16-bit range
        max_value = np.max(np.abs(recording))
        if max_value > 0:
            recording = np.int16(recording / max_value * 32767)
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
    finally:
        # Ensure audio stream is stopped and resources are released
        sd.stop()
        time.sleep(0.5)
