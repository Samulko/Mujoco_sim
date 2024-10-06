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

            # Define the prompt template for the AI
            self.prompt = ChatPromptTemplate.from_template(
                """
                You are the Manager Agent in a multi-agent robotic system for industrial disassembly tasks. Your role is to coordinate all interactions and assign tasks to other agents, ensuring safe, efficient, and compliant operations. You are responsible for:

                1. Interpreting user commands accurately, even if they are ambiguous or complex. Ask for clarification if you don't understand, or the input is outside your scope.
                2. Maintaining context over multiple interactions using conversation history.
                3. Prioritizing and coordinating tasks among specialized agents.
                4. Synthesizing information from various agents to make informed decisions.
                5. Communicating results, progress, and any issues to the user clearly and proactively.
                6. Continuously monitoring task progress and adjusting plans as needed. If you are not sure, ask the user for help.

                The agents you coordinate are:
                - Structural Engineer Agent: Validates requests against current disassembly manuals using a RAG system.
                - Stability Agent: Analyzes the stability of structures during disassembly and suggests safety measures.
                - Planning Agent: Generates detailed action sequence plans for industrial arm robot control systems.
                - Safety Agent: Oversees all operations to ensure compliance with safety standards and regulations.

                Use your advanced natural language understanding to interpret the user's intent and maintain conversation context. When processing a request:

                1. Interpret the user's intent and clarify if necessary.
                2. Determine which agent(s) need to be involved.
                3. Prioritize the task within the current workflow.
                4. Coordinate the necessary information flow between agents.
                5. Synthesize the results from various agents.
                6. Develop a primary plan and contingency plans.
                7. Monitor progress and provide regular updates to the user.

                Always strive for clear communication, efficient task routing, and safe operation. If any stage cannot be completed safely or efficiently, communicate this to the user along with the reasons and possible alternatives.

                Current conversation:
                {history}
                Human: {human_input}
                AI: Let's process this request step by step:
                1. Interpret the user's intent.
                2. Determine which agent(s) need to be involved.
                3. Prioritize the task and coordinate information flow.
                4. Synthesize results and develop plans.
                5. Formulate a clear response or action plan for the user. Be brief in your response.
                Response:
                """
            )

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
            # Use the language model to interpret the command
            response = self.llm.invoke(input=f"Interpret this command for disassembling a simple portal frame: {command}")
            interpreted_command = response.content
            if isinstance(interpreted_command, list):
                interpreted_command = interpreted_command[0] if interpreted_command else ""
            interpreted_command = str(interpreted_command).strip()
            self.log_conversation(f"[INFO] Interpreted command: {interpreted_command}")

            # Here you would typically interact with other agents (Structural Engineer, Stability, Planning)
            # For this example, we'll just log the interpreted command
            self.log_conversation(f"[INFO] Processing command: {interpreted_command}")
            
            # Simulate response from other agents
            self.log_conversation("[INFO] Simulated validation: Request is standard")
            self.log_conversation("[INFO] Simulated planning: Plan created successfully")

            return f"Command processed: {interpreted_command}"
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
