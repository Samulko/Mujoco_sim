#!/usr/bin/env python3

import rospy
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
from instructor import OpenAISchema
from pydantic import Field

class RoboticAction(OpenAISchema):
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
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_action_sequence(self, plan):
        result = self.chain.run(plan)
        try:
            action_sequence = RoboticAction.from_response(result)
            return action_sequence
        except json.JSONDecodeError:
            rospy.logerr("Failed to parse JSON from LLM output")
            return None

    def validate_action_sequence(self, action_sequence):
        # Implement validation logic here
        # For now, we'll just check if the sequence is not empty
        return len(action_sequence.planning_sequence) > 0

    def execute_plan(self, plan):
        action_sequence = self.generate_action_sequence(plan)
        if action_sequence and self.validate_action_sequence(action_sequence):
            rospy.loginfo(f"Executing plan: {action_sequence.dict()}")
            # Here you would implement the actual execution of the action sequence
            return True
        else:
            rospy.logerr("Failed to generate or validate action sequence")
            return False

if __name__ == "__main__":
    rospy.init_node('planning_agent')
    planning_agent = PlanningAgent()
    # Example usage
    plan = "Pick up the red cube and place it on the blue platform"
    success = planning_agent.execute_plan(plan)
    if success:
        rospy.loginfo("Plan executed successfully")
    else:
        rospy.logerr("Plan execution failed")
