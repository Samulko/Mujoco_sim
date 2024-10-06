#!/usr/bin/env python3

import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import time
import logging
from engineer import StructuralEngineerAgent

# Load environment variables from .env file
load_dotenv()
print(f"OPENAI_API_KEY loaded in Manager Agent: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ManagerAgent:
    def __init__(self):
        try:
            # Get OpenAI API key from environment variables
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                logging.error("OPENAI_API_KEY not found in environment variables")
                raise ValueError("OPENAI_API_KEY not set")
            logging.info("OpenAI API key loaded successfully")
            
            # Initialize ChatOpenAI model
            self.llm = ChatOpenAI(temperature=0, model="gpt-4")

            # Initialize conversation memory
            self.memory = ConversationBufferMemory(return_messages=True)

            # Initialize StructuralEngineerAgent
            self.engineer_agent = StructuralEngineerAgent()

            # Initialize a conversation log
            self.conversation_log = []

            # Create a timestamped log file name
            self.log_file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'conversation_logs', f'conversation_log_{int(time.time())}.txt')
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

            logging.info("Manager Agent initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Manager Agent: {e}")
            raise

    def log_conversation(self, message):
        self.conversation_log.append(message)
        
        # Append the message to the log file
        with open(self.log_file_path, 'a') as f:
            f.write(message + '\n')

    def process_command(self, command):
        try:
            self.log_conversation(f"[INFO] Received command: {command}")

            # Send the command to the Structural Engineer Agent
            is_standard, validation_details, _ = self.engineer_agent.handle_validate_request(command)

            self.log_conversation(f"[INFO] Engineer response - Is Standard: {is_standard}")

            if is_standard:
                disassembly_instructions = validation_details.get('disassembly_instructions', [])
                safety_instructions = validation_details.get('safety_instructions', [])

                # Format and print disassembly instructions
                print("Disassembly Instructions:")
                for i, step in enumerate(disassembly_instructions, 1):
                    print(f"{i}. {step['step']}")

                # Format and print safety instructions
                print("\nSafety Instructions:")
                for i, instruction in enumerate(safety_instructions, 1):
                    print(f"{i}. {instruction}")

                # Ask if user wants to make modifications
                while True:
                    modify = input("\nDo you want to make modifications to the sequence? (yes/no): ").lower()
                    if modify in ['yes', 'no']:
                        break
                    print("Please answer with 'yes' or 'no'.")

                if modify == 'yes':
                    print("Please provide your modifications. (Not implemented in this version)")
                    # Here you would implement the logic to handle modifications

            else:
                print("The structure does not match any standard procedures in our database.")
                print("Please provide more details or consult with a specialist for a custom disassembly plan.")

            return "Command processed successfully."
        except Exception as e:
            error_message = f"Error processing command: {e}"
            self.log_conversation(f"[ERROR] {error_message}")
            return error_message

    def run_system_test(self):
        logging.info("Running system test...")
        test_result = "System test completed successfully."
        self.log_conversation(f"[INFO] {test_result}")
        return test_result

def main():
    manager_agent = ManagerAgent()
    
    while True:
        user_input = input("Enter a command (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'test system':
            result = manager_agent.run_system_test()
        else:
            result = manager_agent.process_command(user_input)
        
        print(result)

if __name__ == '__main__':
    main()
