"""main.py

Main program to run agent workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from koios.AgentWorkflow.AgentWorkflow import AgentWorkflow

import streamlit
# Below can be used to display markdown in notebook.
# from IPython.display import display, Markdown


class Main:
    """Main class to run the agent workflow."""

    @staticmethod
    def main(args: dict[str]) -> None:
        """Main static method for passing CL arguments into workflow.

        Args:
            args (dict[str]): Command line arguments.
        """
        question: str = ""
        if len(args) > 1:
            question = args[1]
            Main.run_agent(question)
        else:
            print("Please provide a question as an argument.")
            return

    @staticmethod
    def run_streamlit() -> None:
        """Run the agent workflow and display the output in a Streamlit app."""
        streamlit.title("Koios Research Agent")

        model_options = ["llama3.2"]
        selected_model = streamlit.sidebar.selectbox(
            "Choose the LLM Model",
            options=model_options,
            index=0
        )
        
        # Temperature Setting
        temperature = streamlit.sidebar.slider(
            "Set the Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
        workflow = AgentWorkflow(selected_model, temperature)
        query = streamlit.text_input("Enter your research question:", "")

        should_run_query = streamlit.button("Run Query")

        if len(query) > 0:
            output = workflow.local_agent.invoke({"question": query})
            if should_run_query:
                if query:
                    streamlit.write(output["generation"])

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
