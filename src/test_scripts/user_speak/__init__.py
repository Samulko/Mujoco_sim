from .main import main

__all__ = ['main']
    set_alsa_params()
    temp_dir = None
    silent_recordings = 0
    max_silent_recordings = 3
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

                # Record audio and save it to WAV file
                recorded_file = record_audio(audio_filename)
                
                # Cancel the alarm
                signal.alarm(0)

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

                except Exception as e:
                    print(f"Transcription or response generation failed: {str(e)}")
                    logging.error(f"Transcription or response generation failed: {str(e)}")

                # Remove the temporary audio file
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logging.info(f"Temporary audio file {audio_filename} removed")

                print("Press Enter to record again, or 'q' to quit.")
                user_input = input().lower()
                if user_input == 'q':
                    print("Quitting the program...")
                    return

            except TimeoutError:
                print("Recording timed out. Please try again.")
                logging.warning("Recording timed out")
            except KeyboardInterrupt:
                print("\nRecording interrupted by user.")
                logging.info("Recording interrupted by user")
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
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory {temp_dir} removed")

if __name__ == "__main__":
    main()
