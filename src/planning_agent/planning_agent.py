#!/usr/bin/env python3

import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from pydantic import BaseModel, Field
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RoboticAction(BaseModel):
    human_working: bool = Field(..., description="Indicates if a human is working alongside the robot")
    selected_element: str = Field(..., description="The element being worked on")
    planning_sequence: list[str] = Field(..., description="List of actions for the robot to perform")

class PlanningAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["plan"],
            template="Translate the following plan into a sequence of robotic actions:\n{plan}\n\nOutput the result as a JSON object with the following structure:\n{{\"human_working\": boolean, \"selected_element\": string, \"planning_sequence\": [string]}}"
        )
        self.chain: Runnable = self.prompt | self.llm | StrOutputParser()

    def generate_action_sequence(self, plan):
        result = self.chain.invoke({"plan": plan})
        try:
            action_data = json.loads(result)
            action_sequence = RoboticAction(**action_data)
            return action_sequence
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON from LLM output")
            return None
        except ValueError as e:
            logging.error(f"Failed to create RoboticAction: {e}")
            return None

    def validate_action_sequence(self, action_sequence):
        # Implement validation logic here
        # For now, we'll just check if the sequence is not empty
        return len(action_sequence.planning_sequence) > 0

    def execute_plan(self, plan):
        action_sequence = self.generate_action_sequence(plan)
        if action_sequence and self.validate_action_sequence(action_sequence):
            logging.info(f"Executing plan: {action_sequence.model_dump()}")
            # Here you would implement the actual execution of the action sequence
            return True
        else:
            logging.error("Failed to generate or validate action sequence")
            return False

def main():
    planning_agent = PlanningAgent()
    # Example usage
    plan = "Pick up the red cube and place it on the blue platform"
    success = planning_agent.execute_plan(plan)
    if success:
        logging.info("Plan executed successfully")
    else:
        logging.error("Plan execution failed")

if __name__ == "__main__":
    main()
