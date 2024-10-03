#!/usr/bin/env python3

# Import necessary libraries
import openai
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()
print(f"OPENAI_API_KEY loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

class StructuralEngineerAgent:
    def __init__(self):
        try:
            # Get OpenAI API key from environment variables
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            # Initialize OpenAI client
            self.client = OpenAI(api_key=self.openai_api_key)

            # Initialize RAG (Retrieval-Augmented Generation) system
            self.initialize_rag_system()

            # Define the prompt template for the AI
            self.prompt = ChatPromptTemplate.from_template("""
                You are the Structural Engineer Agent in a multi-agent robotic system for disassembly tasks. Your role is to validate requests against current disassembly manuals using an RAG system. You are responsible for:

                1. Analyze the request and compare it to the structures in your RAG document.
                2. If a match is found:
                   a. Confirm that the structure matches one in the RAG system.
                   b. Provide a brief description of the matching structure.
                   c. Pass this information along with the information on the proper disassembly sequence as it is described in the RAG to the manager agent for further processing.
                3. If no match is found:
                   a. Clearly communicate to the manager agent that the structure does not match any in the RAG system.
                4. In case of partial matches or ambiguities:
                   a. Explain the nature of the partial match or ambiguity.
                   b. If the match is close to what is described, but does not match the number of elements of structural types, respond negatively by saying structure is not standard.
                   c. Request more information if needed.

                Current request: {request}
                Relevant information from the manuals: {context}

                AI: Let's analyze this request systematically:

                1. Compare the request to the structures in the RAG system.
                2. Determine if there's a full match, partial match, or no match.
                3. Based on the result:
                   a. For a full match: Confirm the match and pass the information on the proper disassembly sequence as it is described in the RAG to the manager agent for further processing. In this case you must mention the proceedure is standard.
                   b. For no match: Clearly state that no matching structure was found in the RAG system. You must mention the proceedure is not standard.
                   c. For partial matches or ambiguities: Explain the situation. If the match is close but not a full match - for instance the structure description does not match the number of elements of structural types, respond negatively by saying structure is not standard.
                   d. Mention either that the proceedure is standard or not standard.

                Analysis and Validation Result:
            """)

            print("Structural Engineer Agent Initialized.")
        except Exception as e:
            print(f"Error initializing Structural Engineer Agent: {str(e)}")
            raise

    def initialize_rag_system(self):
        try:
            # Set the path to the disassembly manual
            manual_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'engineer_rag', 'disassembly_manual.json')
            print(f"Manual path: {manual_path}")
            print(f"File exists: {os.path.exists(manual_path)}")
            
            if not os.path.exists(manual_path):
                raise FileNotFoundError(f"Disassembly manual not found at {manual_path}")

            # Load the JSON data using JSONLoader
            loader = JSONLoader(
                file_path=manual_path,
                jq_schema='.procedures[]',
                text_content=False
            )

            # Load and process the documents
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")

            # Split the documents into smaller chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)

            # Create a vector store for efficient similarity search
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(texts, embeddings)
            print("RAG system initialized successfully")

        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            raise

    def generate_disassembly_plan(self, structure_info):
        # Create a prompt for generating a disassembly plan
        prompt = f"""
        Given the following structure information:
        {structure_info}

        Generate a high-level disassembly plan. The plan should be a list of steps, each describing a major disassembly action.
        Consider the structure's components and general disassembly instructions.

        Format your response as a list of steps, each on a new line, starting with a number.
        """

        # Use OpenAI to generate the disassembly plan
        response = self.client.chat.completions.create(
            temperature=0,
            model="gpt-4o",  # Keeping gpt-4o as the model
            messages=[
                {"role": "system", "content": "You are a structural engineer specializing in disassembly procedures."},
                {"role": "user", "content": prompt}
            ]
        )
        
        disassembly_plan = response.choices[0].message.content if response.choices else ""
        return disassembly_plan

    def handle_validate_request(self, request):
        try:
            print(f"StructuralEngineerAgent: Validating request: {request}")

            # Retrieve relevant context from the RAG system
            docs = self.vectorstore.similarity_search(request, k=2)
            context = "\n".join([doc.page_content for doc in docs])
            print(f"StructuralEngineerAgent: Retrieved context: {context}")

            # Use OpenAI to validate the request
            response = self.client.chat.completions.create(
                temperature=0,
                model="gpt-4o",  # Changed back to gpt-4o
                messages=[
                    {"role": "system", "content": self.prompt.format(request=request, context=context)},
                    {"role": "user", "content": request}
                ]
            )
            validation_details = response.choices[0].message.content
            print(f"StructuralEngineerAgent: Validation details: {validation_details}")

            # Determine if the request follows standard procedures
            is_standard = "standard" in validation_details.lower() and "not standard" not in validation_details.lower()

            # Generate disassembly plan
            disassembly_plan = self.generate_disassembly_plan(context)
            print(f"StructuralEngineerAgent: Is standard: {is_standard}")

            # Export response to file
            self.export_response(request, validation_details, disassembly_plan)

            return is_standard, validation_details, disassembly_plan
        except Exception as e:
            print(f"StructuralEngineerAgent: Error handling validate request: {str(e)}")
            return False, f"Error: {str(e)}", ""

    def export_response(self, request, validation_details, disassembly_plan):
        # Create the export directory if it doesn't exist
        export_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'engineer_response')
        os.makedirs(export_dir, exist_ok=True)

        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engineer_response_{timestamp}.log"
        filepath = os.path.join(export_dir, filename)

        # Prepare the data to be exported
        export_data = f"""
Request:
{request}

Validation Details:
{validation_details}

Disassembly Plan:
{disassembly_plan}
"""

        # Write the data to a plain text file
        with open(filepath, 'w') as f:
            f.write(export_data)

        print(f"Response exported to: {filepath}")

# Main execution block
if __name__ == '__main__':
    try:
        # Create an instance of the StructuralEngineerAgent
        structural_engineer_agent = StructuralEngineerAgent()
        
        # Example usage
        request = "Disassemble a laptop with 4 screws on the back panel"
        is_standard, validation_details, disassembly_plan = structural_engineer_agent.handle_validate_request(request)
        
        # Print the results
        print(f"Is Standard: {is_standard}")
        print(f"Validation Details: {validation_details}")
        print(f"Disassembly Plan: {disassembly_plan}")
        
    except Exception as e:
        print(f"Structural Engineer Agent failed: {str(e)}")