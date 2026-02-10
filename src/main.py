"""main.py

Main program to run agent workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.koios.AgentWorkflow.AgentWorkflow import AgentWorkflow
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt
from src.config import Config

import streamlit
# Below can be used to display markdown in notebook.
# from IPython.display import display, Markdown


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
                subprocess.run([sys.executable, "-m", "streamlit", "run", "src/main.py"])
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
        
        # Fetch models and pick the first one as default
        model_options = AgentPrompt.get_available_models()
        selected_model = model_options[0] if model_options else "llama3.2"
        temperature = 0.5
        
        print(f"Using model: {selected_model} (temp: {temperature})")
        
        workflow = AgentWorkflow(selected_model, temperature)
        output = workflow.local_agent.invoke({"question": question})
        
        generation = output.get("generation", "No generation produced.")
        print("\n--- Research Report ---\n")
        print(generation)
        print("\n-----------------------\n")
        
        Main.write_output_file(generation)

    @staticmethod
    def run_streamlit() -> None:
        """Run the agent workflow and display the output in a Streamlit app."""
        streamlit.title("Koios Research Agent")

        # Fetch models from API via AgentPrompt
        model_options = AgentPrompt.get_available_models()

        selected_model = streamlit.sidebar.selectbox(
            "Choose the LLM Model",
            options=model_options,
            index=0
        )
        
        # Temperature Setting
        temperature = streamlit.sidebar.slider(
            "Set the creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
        workflow = AgentWorkflow(selected_model, temperature)
        query = streamlit.text_input("Enter your research question:", "")

        if streamlit.button("Run Query"):
            if query:
                with streamlit.spinner("Generating research report..."):
                    output = workflow.local_agent.invoke({"question": query})
                    streamlit.markdown("### Research Report")
                    streamlit.write(output["generation"])
            else:
                streamlit.warning("Please enter a research question.")

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


if __name__ == "__main__":
    Main.run_streamlit()
