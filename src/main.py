"""main.py

Main program to run agent workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.AgentWorkflow.AgentWorkflow import AgentWorkflow

import sys
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

    def run_agent(query: str) -> None:
        """Run the agent workflow and write the output to a file.

        Args:
            query (str): _description_
        """
        workflow = AgentWorkflow()
        output = workflow.local_agent.invoke({"question": query})
        print("=======")
        output: str = str(output["generation"])
        print("Writing result to file...")
        writer = open("output.md", 'w')
        writer.write(output)
        writer.close()
        print("Result written to output.md file")


if __name__ == "__main__":
    Main.main(sys.argv)
