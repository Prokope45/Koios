"""main.py

Main program to run agent workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
import os
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
    @streamlit.cache_resource
    def get_document_store():
        from src.koios.DocumentStore import DocumentStore
        return DocumentStore()

    @staticmethod
    def run_streamlit() -> None:
        """Run the agent workflow and display the output in a Streamlit app."""
        config = Config()
        config.setup()
        
        streamlit.set_page_config(page_title="Koios Research Agent", layout="wide")
        streamlit.title("Koios Research Agent")

        # Initialize session state for chat history
        if "messages" not in streamlit.session_state:
            streamlit.session_state.messages = []

        # Sidebar configuration
        streamlit.sidebar.title("Settings")
        
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

        # Internet Search Toggle
        enable_internet_search = streamlit.sidebar.toggle(
            "Enable Internet Search",
            value=config.enable_internet_search
        )

        streamlit.sidebar.divider()
        streamlit.sidebar.subheader("Document Store")
        
        doc_store = Main.get_document_store()

        uploaded_files = streamlit.sidebar.file_uploader(
            "Upload PDF documentation",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save file temporarily to process it
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Only process if not already in store (simple check by filename)
                existing_docs = doc_store.get_all_documents()
                if uploaded_file.name not in existing_docs:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with streamlit.sidebar.spinner(f"Indexing {uploaded_file.name}..."):
                        doc_store.add_pdf(file_path)
                    streamlit.sidebar.success(f"Indexed {uploaded_file.name}")

        # Show uploaded documents
        docs_in_store = doc_store.get_all_documents()
        if docs_in_store:
            streamlit.sidebar.write("Uploaded Documents:")
            for doc_name in docs_in_store:
                streamlit.sidebar.text(f"📄 {doc_name}")
        else:
            streamlit.sidebar.info("No documents uploaded yet.")

        if streamlit.sidebar.button("Clear Chat History"):
            streamlit.session_state.messages = []
            streamlit.rerun()
    
        workflow = AgentWorkflow(selected_model, temperature, enable_internet_search=enable_internet_search)

        # Display chat messages from history on app rerun
        for message in streamlit.session_state.messages:
            with streamlit.chat_message(message["role"]):
                streamlit.markdown(message["content"])

        # React to user input
        if prompt := streamlit.chat_input("Enter your research question:"):
            # Display user message in chat message container
            with streamlit.chat_message("user"):
                streamlit.markdown(prompt)
            
            # Add user message to chat history
            streamlit.session_state.messages.append({"role": "user", "content": prompt})

            # Prepare history for the agent (excluding the current prompt)
            history = streamlit.session_state.messages[:-1]

            # Display assistant response in chat message container
            with streamlit.chat_message("assistant"):
                with streamlit.spinner("Thinking..."):
                    # Invoke agent with question and history
                    output = workflow.local_agent.invoke({
                        "question": prompt,
                        "history": history,
                        "context": "",
                        "generation": "",
                        "search_query": ""
                    })
                    response = output["generation"]
                    streamlit.markdown(response)
            
            # Add assistant response to chat history
            streamlit.session_state.messages.append({"role": "assistant", "content": response})

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
