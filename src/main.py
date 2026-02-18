"""main.py

Main program to run agent workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.koios.AgentWorkflow.AgentWorkflow import AgentWorkflow
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt
from src.config import Config


class Main:
    """Main class to run the agent workflow."""

    @staticmethod
    def main(args: list[str]) -> None:
        """Main static method for passing CL arguments into workflow.

        Args:
            args (list[str]): Command line arguments.
        """
        import subprocess
        import sys

        config = Config()
        config.setup()

        actual_args = args[1:]

        if len(actual_args) >= 1:
            if actual_args[0] == "app":
                print("Launching Streamlit app...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
            else:
                question = actual_args[0]
                Main.run_agent(question)
        else:
            print("Please provide a question as an argument.")
            return

    @staticmethod
    def run_agent(question: str) -> None:
        """Run the agent workflow in CLI mode.

        Args:
            question (str): The research question.
        """
        print(f"Running agent with question: {question}")

        config = Config()
        config.setup()

        # Fetch models and pick the first one as default
        model_options = AgentPrompt.get_available_models()
        selected_model = model_options[0] if model_options else "llama3.2"
        temperature = 0.5

        print(f"Using model: {selected_model} (temp: {temperature})")
        print(f"Internet search enabled: {config.enable_internet_search}")

        workflow = AgentWorkflow(selected_model, temperature, enable_internet_search=config.enable_internet_search)
        output = workflow.local_agent.invoke({"question": question})

        generation = output.get("generation", "No generation produced.")
        print("\n--- Research Report ---\n")
        print(generation)
        print("\n-----------------------\n")

        Main.write_output_file(generation)

    @staticmethod
    def write_output_file(output: str) -> None:
        """Write the output to a file.

        Args:
            output (str): Output to be written to file.
        """
        print("Writing result to file...")
        writer = open("output.md", 'w')
        writer.write(output)
        writer.close()
        print("Result written to output.md file")
