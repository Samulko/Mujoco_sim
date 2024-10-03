#!/usr/bin/env python3

import os
from openai import OpenAI
from pydantic import Field
import json
import logging
import time
from typing import List
import instructor
from instructor import OpenAISchema
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RoboticAction(OpenAISchema):
    """
    Represents a single robotic action in the disassembly sequence.
    """
    human_working: bool = Field(..., description="Indicates if a human is working alongside the robot")
    selected_element: str = Field(..., description="The element being worked on")
    planning_sequence: List[str] = Field(..., description="List of actions for the robot to perform")

class ActionSequence(OpenAISchema):
    """
    Represents a sequence of robotic actions for the disassembly plan.
    """
    actions: List[RoboticAction] = Field(..., description="List of robotic actions to perform")

class PlanningAgent:
    def __init__(self):
        # Initialize OpenAI client with Instructor
        self.client = instructor.patch(OpenAI())

        # Dictionary of possible robot actions
        self.robot_actions = {
            "move_in_cartesian_path": "move_in_cartesian_path(move_distance_x, move_distance_y, move_distance_z)",
            "moveto": "moveto",
            "picking": "picking",
            "holding": "holding",
            "placing": "placing",
            "human_action": "human_action(action_description)"
        }

        # Initialize LangChain components
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=ActionSequence)

        logging.info("Planning Agent Initialized and ready to work.")

    def handle_plan_execution(self, plan):
        logging.info(f"Planning Agent: Received plan execution request: {plan}")

        # Translate plan into structured action sequences
        action_sequence = self.translate_plan(plan)

        if action_sequence is None:
            return False, "Failed to generate a valid action sequence."

        # Validate action sequence
        if self.validate_action_sequence(action_sequence):
            # Execute preliminary steps
            self.execute_preliminary_steps()

            # Check if additional safety measures are needed
            if "unsafe" in plan.lower() or "modifications" in plan.lower():
                action_sequence = self.add_safety_measures(action_sequence)

            # Write the JSON file
            json_file_path = self.write_json_file(action_sequence)

            # Execute actions
            success, execution_details = self.execute_actions(action_sequence)
            return success, f"{execution_details}. JSON file created at {json_file_path}"
        else:
            return False, "Invalid action sequence. Please check the plan and try again."

    def add_safety_measures(self, action_sequence):
        logging.info("Planning Agent: Adding additional safety measures to the action sequence")
        safety_measures = [
            "implement_temporary_supports",
            "distribute_load_evenly",
            "monitor_stability_continuously"
        ]
        action_sequence.actions[0].planning_sequence = safety_measures + action_sequence.actions[0].planning_sequence
        return action_sequence

    def translate_plan(self, plan):
        logging.info(f"Planning Agent: Translating plan: {plan}")
        action_schemas = ', '.join(f"{key}: {value}" for key, value in self.robot_actions.items())
        
        example_json = '''
        {
          "actions": [
            {
              "human_working": false,
              "selected_element": "element_2",
              "planning_sequence": ["moveto", "holding"]
            },
            {
              "human_working": true,
              "selected_element": "element_1",
              "planning_sequence": ["human_action(remove column 1)"]
            }
          ]
        }
        '''
        
        # Step 1: Extract Numbered Instructions
        step1_prompt = ChatPromptTemplate.from_template(
            "You are the Planning Agent in a multi-agent system that controls a robotic arm for disassembly tasks. "
            "Your first task is to extract the numbered Disassembly Instructions from the plan below.\n\n"
            "**Plan:**\n{plan}\n\n"
            "Please list the numbered Disassembly Instructions exactly as they appear."
        )
        
        step1_chain = LLMChain(llm=self.llm, prompt=step1_prompt, verbose=True)
        numbered_instructions = step1_chain.run(plan=plan)
        
        # Step 2: Interpret Instructions Step by Step
        step2_prompt = ChatPromptTemplate.from_template(
            "Now, interpret each of the numbered Disassembly Instructions step by step. For each instruction, do the following:\n"
            "1. Identify the preferences stated in the initial prompt for which actor should perform which action.\n"
            "2. Identify the actor (human or robot) performing the action.\n"
            "3. Determine the specific element being worked on.\n"
            "4. Decide which action schemas are needed to perform this instruction.\n"
            "5. Explain your reasoning.\n\n"
            "**Numbered Instructions:**\n{numbered_instructions}\n\n"
            "Remember to only use the following action schemas:\n{action_schemas}\n\n"
            "Proceed step by step."
        )
        
        step2_chain = LLMChain(llm=self.llm, prompt=step2_prompt, verbose=True)
        interpreted_steps = step2_chain.run(
            numbered_instructions=numbered_instructions,
            action_schemas=action_schemas
        )
        
        # Step 3: Generate the Action Sequence
        step3_prompt = ChatPromptTemplate.from_template(
            "Based on your interpretations, generate the action sequence in JSON format. Follow these guidelines:\n"
            "- Take into account the preferences for human-robot task division stated in the initial prompt\n"
            "- Use 'human_working' set to true if a human is performing the action, false if the robot is.\n"
            "- 'selected_element' should specify the element being worked on.\n"
            "- 'planning_sequence' should list the actions in execution order, using only the provided action schemas.\n"
            "- Ensure the sequence follows the exact order of the numbered instructions.\n"
            "- Maintain consistent roles for each actor throughout the process.\n"
            "- Include necessary preparatory movements before each main action.\n"
            "- For human actions, use 'human_action(action_description)'.\n"
            "- Use specific element names (e.g., 'element_1').\n"
            "- Robot pick-and-place sequences should follow: 'moveto' -> 'picking' -> 'holding' -> 'placing'.\n"
            "- Support actions should follow: 'moveto' -> 'holding', and continue 'holding' until released.\n"
            "- Use 'deposition_zone' as the destination for removed elements.\n\n"
            "**Your Interpretations:**\n{interpreted_steps}\n\n"
            "Provide the final action sequence in the following JSON format:\n{format_instructions}\n\n"
            "Here is an example:\n{example_json}"
        )
        
        step3_chain = LLMChain(llm=self.llm, prompt=step3_prompt, verbose=True)
        final_result = step3_chain.run(
            interpreted_steps=interpreted_steps,
            format_instructions=self.output_parser.get_format_instructions(),
            example_json=example_json
        )
        
        try:
            action_sequence = self.output_parser.parse(final_result)
            logging.info(f"Planning Agent: Translated plan into action sequence: {action_sequence}")
            return action_sequence
        except Exception as e:
            logging.error(f"Planning Agent: Failed to generate action sequence: {e}")
            return None

    def validate_action_sequence(self, action_sequence):
        if not isinstance(action_sequence, ActionSequence):
            logging.error("Planning Agent: Action sequence is not an ActionSequence object")
            return False

        for item in action_sequence.actions:
            if not isinstance(item, RoboticAction):
                logging.error(f"Planning Agent: Invalid item in action sequence: {item}")
                return False

            # Check if all actions in the sequence are valid
            invalid_actions = []
            for action in item.planning_sequence:
                action_name = action.split('(')[0]  # Extract action name
                if not any(action_name == valid for valid in self.robot_actions.keys()):
                    invalid_actions.append(action)

            if invalid_actions:
                logging.error(f"Planning Agent: Invalid actions in sequence: {', '.join(invalid_actions)}")
                return False

        logging.info("Planning Agent: Action sequence validated successfully")
        return True

    def execute_preliminary_steps(self):
        # Placeholder for executing preliminary steps
        logging.info("Planning Agent: Executing preliminary steps for safety.")

    def write_json_file(self, action_sequence):
        # Write the action sequence to a JSON file
        robot_sequence_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'robot_sequence')
        os.makedirs(robot_sequence_dir, exist_ok=True)
        timestamp = time.time()
        json_file_name = f"action_sequence_{timestamp}.json"
        json_file_path = os.path.join(robot_sequence_dir, json_file_name)
        
        with open(json_file_path, 'w') as json_file:
            json.dump(action_sequence.dict(), json_file, indent=4)
        logging.info(f"Planning Agent: Action sequence JSON file created at {json_file_path}")
        
        # Print the contents of the JSON file
        with open(json_file_path, 'r') as json_file:
            logging.info(f"Planning Agent: JSON file contents:\n{json_file.read()}")
        
        return json_file_path

    def execute_actions(self, action_sequence):
        # Simulate executing actions
        try:
            for action in action_sequence.actions:
                for step in action.planning_sequence:
                    logging.info(f"Planning Agent: Executed action: {step}")
            return True, "Action sequence executed successfully."
        except Exception as e:
            logging.error(f"Planning Agent: Error executing actions: {e}")
            return False, f"Error executing actions: {e}"

def main():
    planning_agent = PlanningAgent()
    plan = """
    Description of the Structure:
    - Simple portal frame structure with three elements: two vertical columns (column 1 and column 3) and one horizontal beam (beam 2).
    - The structure forms a basic 'П' shape, and all elements are assumed to be of standard construction material (like timber, steel, or concrete) with the same dimensions.

    Disassembly Instructions:
    1. actor_1 supports the beam (element 2) to secure the structure.
    2. actor_2 removes the vertical column (1) from below the beam (2).
    3. actor_2 removes the vertical column (3) from below the beam (2).
    4. Finally, actor_1 carefully removes the beam (2) that is being supported last and places it in the deposition zone.
    
    -The robot will be supporting element 2.
    """
    success, details = planning_agent.handle_plan_execution(plan)
    if success:
        logging.info(f"Plan executed successfully: {details}")
    else:
        logging.error(f"Plan execution failed: {details}")

if __name__ == "__main__":
    main()
