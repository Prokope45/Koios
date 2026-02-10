"""WorkflowActions.py

Workflow actions class containing actions for agent to take.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt


class WorkflowActions:
    """Provide workflow actions for agent to take."""

    def __init__(self, agent_prompt: AgentPrompt):
        """Construct WorkflowActions object.

        Args:
            agent_prompt (AgentPrompt): AgentPrompt object to use for getting
                chains.
        """
        self.__agent_prompt = agent_prompt

    def generate(self, state: dict) -> dict:
        """Generate answer based on existing knowledge.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): New key added to state, generation, containing
                LLM generation.
        """
        print("Step: Generating Final Response")
        question = state["question"]
        history = state.get("history", [])
        
        # Ensure context is not None or empty if we skipped web search
        context = state.get("context")
        if not context:
            context = "No web search context provided. Answer based on your internal knowledge."

        # We can pass history to the chain if we update the template, 
        # but for now let's just include it in the context or question if needed.
        # A better way is to update the prompt template to handle history.
        
        generation = self.__agent_prompt.get_generate_chain.invoke(
            {"context": context, "question": question, "history": history}
        )
        return {"generation": generation}

    def transform_query(self, state: dict) -> dict:
        """Transform user question to web search.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Appended search query.
        """
        print("Step: Optimizing Query for Web Search")
        question = state['question']
        gen_query = self.__agent_prompt.get_query_chain.invoke(
            {"question": question}
        )
        search_query = gen_query["query"]
        return {"search_query": search_query}

    def web_search(self, state: dict) -> dict:
        """Web search based on the question.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Appended web results to context.
        """
        search_query = state['search_query']
        print(f'Step: Searching the Web for: "{search_query}"')
        search_result = self.__agent_prompt.web_search_with_fallback(
            search_query
        )
        return {"context": search_result}

    def route_question(self, state: dict) -> dict:
        """Route question to web search or generation.

        Args:
            state (dict): The current graph state.

        Returns:
            str: Action to call.
        """
        print("Step: Routing Query")
        question = state['question']
        output = self.__agent_prompt.get_router_chain.invoke(
            {"question": question}
        )
        
        choice = output.get('choice', 'generate')
        print(f"Step: Router Decision: {choice}")
        
        if choice == "web_search":
            print("Step: Routing Query to Web Search")
            return "web_search"
        else:
            print("Step: Routing Query to Generation")
            return "generate"
