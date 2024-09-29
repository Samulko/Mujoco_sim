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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RoboticAction(OpenAISchema):
    human_working: bool = Field(..., description="Indicates if a human is working alongside the robot")
    selected_element: str = Field(..., description="The element being worked on")
    planning_sequence: List[str] = Field(..., description="List of actions for the robot to perform")

class ActionSequence(OpenAISchema):
    actions: List[RoboticAction] = Field(..., description="List of robotic actions to perform")

class PlanningAgent:
    def __init__(self):
        # Initialize OpenAI client with Instructor
        self.client = OpenAI()
        self.client = instructor.patch(self.client)

        # Dictionary of possible robot actions
        self.robot_actions = {
            "move_in_cartesian_path": "move_in_cartesian_path(group, move_distance_x, move_distance_y, move_distance_z)",
            "moveto": "moveto(pose_element, element_name)",
            "picking": "picking(pose_element, element_name)",
            "holding": "holding(pose_element, element_name)",
            "placing": "placing(pose_element, element_name)"
        }

        logging.info("Planning Agent Initialized and ready to work.")

    def handle_plan_execution(self, plan):
        logging.info(f"Planning Agent: Received plan execution request: {plan}")

        # Translate plan into structured action sequences
        action_sequence = self.translate_plan(plan)

        if action_sequence is None:
            return False, "Failed to generate a valid action sequence."

        # Validate action sequence
        if self.validate_action_sequence(action_sequence.actions):
            # Execute preliminary steps
            self.execute_preliminary_steps()

            # Check if additional safety measures are needed
            if "unsafe" in plan.lower() or "modifications" in plan.lower():
                action_sequence.actions = self.add_safety_measures(action_sequence.actions)

            # Write the JSON file
            json_file_path = self.write_json_file(action_sequence)

            # Execute actions
            success, execution_details = self.execute_actions(action_sequence.actions)
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
        action_sequence["planning_sequence"] = safety_measures + action_sequence["planning_sequence"]
        return action_sequence

    def translate_plan(self, plan):
        logging.info(f"Planning Agent: Translating plan: {plan}")
        action_schemas = ', '.join(self.robot_actions.values())
        prompt = f"""
        You are the Planning Agent in a multi-agent system that controls a robotic arm for disassembly tasks. Your role is to translate the disassembly sequence plan into a structured action sequence for the robotic arm to execute. 

        Given the disassembly plan:
        {plan}

        Your task is to:
        1. Analyze the given disassembly plan.
        2. Create a detailed action sequence using only the following action schemas:
           {action_schemas}
        3. Ensure the action sequence follows the correct order and includes all necessary steps for safe and efficient disassembly.
        4. Consider the collaboration between the robotic arm and human operator, if applicable.
        5. Identify the specific element being worked on in each step.

        Guidelines:
        - Break down complex movements into a series of simpler actions.
        - Include necessary preparatory movements before each main action.
        - Consider the need for holding or stabilizing elements during disassembly.
        - For human actions, include a "human_action" step in the planning sequence.
        - Use specific element names (e.g., "element_1" instead of "element 1") for consistency.
        - Re-evaluate the plan and make adjustments as needed.

        Ensure that:
        1. "human_working" is set to true if the current step requires human intervention; otherwise, it is false.
        2. "selected_element" specifies the element being worked on in the current step.
        3. The actions in the "planning_sequence" are organized in execution order.
        4. Include "human_action" steps where human intervention is required.
        5. Use underscores instead of spaces in element names and action parameters.

        Planning Agent, please provide the structured action sequence based on the given disassembly plan.
        """
        try:
            action_sequence = self.client.chat.completions.create(
                model="gpt-4-0125-preview",
                response_model=ActionSequence,
                messages=[
                    {"role": "system", "content": "You are a planning agent that translates disassembly plans into structured action sequences."},
                    {"role": "user", "content": prompt}
                ]
            )
            logging.info(f"Planning Agent: Translated plan into action sequence: {action_sequence}")
            return action_sequence
        except Exception as e:
            logging.error(f"Planning Agent: Failed to generate action sequence: {e}")
            return None

    def clean_action_sequence(self, action_sequence):
        if not isinstance(action_sequence, list):
            logging.error("Planning Agent: Action sequence is not a list")
            return None

        cleaned_sequence = []
        for item in action_sequence:
            if not isinstance(item, dict):
                logging.warning(f"Planning Agent: Skipping invalid item in action sequence: {item}")
                continue

            # Ensure all required keys are present
            required_keys = ["human_working", "selected_element", "planning_sequence"]
            if not all(key in item for key in required_keys):
                logging.warning(f"Planning Agent: Skipping item missing required keys: {item}")
                continue

            cleaned_item = {
                "human_working": bool(item["human_working"]),
                "selected_element": item["selected_element"].replace(" ", "_").lower(),
                "planning_sequence": []
            }

            # Clean up the planning_sequence
            for action in item["planning_sequence"]:
                if isinstance(action, str):
                    cleaned_action = action.replace(" ", "_").lower()
                    if any(cleaned_action.startswith(valid_action) for valid_action in self.robot_actions.keys()):
                        cleaned_item["planning_sequence"].append(cleaned_action)
                    else:
                        logging.warning(f"Planning Agent: Skipping invalid action: {action}")
                else:
                    logging.warning(f"Planning Agent: Skipping non-string action: {action}")

            if cleaned_item["planning_sequence"]:
                cleaned_sequence.append(cleaned_item)
            else:
                logging.warning(f"Planning Agent: Skipping item with empty planning sequence: {item}")

        if not cleaned_sequence:
            logging.error("Planning Agent: No valid items in action sequence after cleaning")
            return None

        return cleaned_sequence

    def validate_action_sequence(self, action_sequence):
        if not isinstance(action_sequence, list):
            logging.error("Planning Agent: Action sequence is not a list")
            return False

        for item in action_sequence:
            if not isinstance(item, RoboticAction):
                logging.error(f"Planning Agent: Invalid item in action sequence: {item}")
                return False

            # Check if planning_sequence is a list
            if not isinstance(item.planning_sequence, list):
                logging.error("Planning Agent: planning_sequence is not a list")
                return False

            # Check if all actions in the sequence are valid
            invalid_actions = [action for action in item.planning_sequence 
                               if not any(action.startswith(valid) for valid in self.robot_actions.keys())]
            if invalid_actions:
                logging.error(f"Planning Agent: Invalid actions in sequence: {', '.join(invalid_actions)}")
                return False

            # Check if selected_element is a non-empty string
            if not isinstance(item.selected_element, str) or not item.selected_element:
                logging.error("Planning Agent: selected_element is not a valid string")
                return False

            # Check if human_working is a boolean
            if not isinstance(item.human_working, bool):
                logging.error("Planning Agent: human_working is not a boolean")
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
        
        # Convert ActionSequence to a dictionary
        action_sequence_dict = {
            "actions": [action.dict() for action in action_sequence.actions]
        }
        
        with open(json_file_path, 'w') as json_file:
            json.dump(action_sequence_dict, json_file, indent=4)
        logging.info(f"Planning Agent: Action sequence JSON file created at {json_file_path}")
        
        # Print the contents of the JSON file
        with open(json_file_path, 'r') as json_file:
            logging.info(f"Planning Agent: JSON file contents:\n{json_file.read()}")
        
        return json_file_path

    def execute_actions(self, action_sequence):
        # Simulate executing actions
        try:
            for action in action_sequence:
                for step in action.planning_sequence:
                    logging.info(f"Planning Agent: Executed action: {step}")
            return True, "Action sequence executed successfully."
        except Exception as e:
            logging.error(f"Planning Agent: Error executing actions: {e}")
            return False, f"Error executing actions: {e}"

def main():
    planning_agent = PlanningAgent()
    plan = "Pick up the red cube and place it on the blue platform"
    success, details = planning_agent.handle_plan_execution(plan)
    if success:
        logging.info(f"Plan executed successfully: {details}")
    else:
        logging.error(f"Plan execution failed: {details}")

if __name__ == "__main__":
    main()
