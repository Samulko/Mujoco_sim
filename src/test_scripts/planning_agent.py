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
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import PydanticOutputParser

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
        
        # Step 1: Initial translation
        initial_translation_prompt = ChatPromptTemplate.from_template(
            "You are the Planning Agent in a multi-agent system that controls a robotic arm for disassembly tasks. "
            "Your role is to translate the disassembly sequence plan into a structured action sequence for the robotic arm to execute, "
            "collaborating with a human operator.\n\n"
            "Given the disassembly plan:\n{plan}\n\n"
            "Your task is to:\n"
            "1. Analyze the given disassembly plan, focusing primarily on the numbered Disassembly Instructions.\n"
            "2. Create a detailed action sequence using only the following action schemas:\n{action_schemas}\n"
            "3. Ensure the action sequence follows the EXACT order specified in the numbered Disassembly Instructions.\n"
            "4. Maintain consistent roles for each actor throughout the entire process as defined in the numbered instructions.\n"
            "5. Identify the specific element being worked on in each step.\n"
            "6. Ensure that if an actor is instructed to support an element, they continue to do so until explicitly instructed to release it.\n\n"
            "Guidelines:\n"
            "- Prioritize the numbered Disassembly Instructions over any additional comments or information provided.\n"
            "- Follow the disassembly instructions step by step, without changing the order or assigned roles.\n"
            "- Include necessary preparatory movements before each main action.\n"
            "- For human actions, use the format: human_action(action_description)\n"
            "- Use specific element names (e.g., 'element_1' instead of 'element 1') for consistency.\n"
            "- Use EXACTLY the action names provided (e.g., 'moveto' not 'move_to').\n"
            "- Set human_working to true for steps performed by humans, and false for steps performed by the robot.\n"
            "- When human_working is true, only include human_action in the planning_sequence.\n"
            "- When human_working is false, only include robot actions in the planning_sequence.\n"
            "- Ensure that each actor maintains their assigned role throughout the entire process as specified in the numbered instructions.\n"
            "- Pay special attention to any instructions about human involvement, such as 'I, the human, will remove column 1.'\n\n"
            "Ensure that:\n"
            "1. 'human_working' is set appropriately based on whether the action is performed by a human or the robot, as specified in the numbered instructions and any additional comments.\n"
            "2. 'selected_element' specifies the element being worked on in the current step.\n"
            "3. The actions in the 'planning_sequence' are organized in execution order.\n"
            "4. Robot pick-and-place sequences follow this pattern: moveto -> picking -> holding -> placing\n"
            "5. Support actions follow this pattern: moveto -> holding, and continue holding in subsequent steps\n"
            "6. Use 'deposition_zone' as the destination for removed elements.\n"
            "7. Each actor maintains their assigned role consistently throughout the entire sequence as per the numbered instructions.\n\n"
            "Translate the plan into a structured action sequence:\n\n{format_instructions}"
        )
        
        initial_chain = LLMChain(llm=self.llm, prompt=initial_translation_prompt, verbose=True)
        
        initial_result = initial_chain.run(
            plan=plan,
            action_schemas=action_schemas,
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        # Step 2: Analyze and correct
        analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze the following action sequence and ensure it adheres to the original plan:\n\n"
            "Original Plan:\n{plan}\n\n"
            "Generated Action Sequence:\n{action_sequence}\n\n"
            "Please identify any discrepancies or errors, especially regarding:\n"
            "1. The order of actions\n"
            "2. Actor roles and consistency\n"
            "3. Correct use of action schemas\n"
            "4. Adherence to the numbered Disassembly Instructions\n"
            "5. Continuous support of elements as specified\n"
            "6. Maintaining consistent roles for each actor throughout the entire process\n\n"
            "Provide a corrected action sequence if necessary."
        )
        
        analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt, verbose=True)
        
        analysis_result = analysis_chain.run(plan=plan, action_sequence=initial_result)
        
        # Step 3: Final validation
        validation_prompt = ChatPromptTemplate.from_template(
            "Given the original plan and the analyzed action sequence, provide a final, corrected action sequence "
            "that strictly adheres to the numbered Disassembly Instructions and maintains consistent actor roles:\n\n"
            "Original Plan:\n{plan}\n\n"
            "Analyzed Action Sequence:\n{analyzed_sequence}\n\n"
            "Provide the final, corrected action sequence using the following format:\n"
            "{format_instructions}"
        )
        
        validation_chain = LLMChain(llm=self.llm, prompt=validation_prompt, verbose=True)
        
        final_result = validation_chain.run(
            plan=plan,
            analyzed_sequence=analysis_result,
            action_schemas=action_schemas,
            format_instructions=self.output_parser.get_format_instructions()
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
    
    -I, the human, will remove column 1.
    """
    success, details = planning_agent.handle_plan_execution(plan)
    if success:
        logging.info(f"Plan executed successfully: {details}")
    else:
        logging.error(f"Plan execution failed: {details}")

if __name__ == "__main__":
    main()
