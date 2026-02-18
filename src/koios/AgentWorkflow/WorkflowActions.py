"""WorkflowActions.py

Workflow actions class containing actions for agent to take.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt
from src.koios.DocumentStore import DocumentStore


class WorkflowActions:
    """Provide workflow actions for agent to take."""

    def __init__(self, agent_prompt: AgentPrompt, enable_internet_search: bool = False):
        """Construct WorkflowActions object.

        Args:
            agent_prompt (AgentPrompt): AgentPrompt object to use for getting
                chains.
            enable_internet_search (bool): Whether to allow web search.
        """
        self.__agent_prompt = agent_prompt
        self.__enable_internet_search = enable_internet_search
        self.__doc_store = DocumentStore()

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
            context = "No additional context provided. Answer based on your internal knowledge."

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
        # Prefer the optimized search_query produced by transform_query; fall
        # back to the raw question if transform_query was somehow skipped.
        search_query = state.get('search_query') or state['question']
        print(f'Step: Searching the Web for: "{search_query}"')
        search_result = self.__agent_prompt.web_search_with_fallback(
            search_query
        )
        return {"context": search_result}

    def doc_search(self, state: dict) -> dict:
        """Search document store based on the question.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Appended document results to context.
        """
        question = state['question']
        print(f'Step: Searching Document Store for: "{question}"')
        docs = self.__doc_store.search(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context}

    def decide_after_doc_search(self, state: dict) -> str:
        """Determine whether to proceed to web search or generation.

        Args:
            state (dict): The current graph state.

        Returns:
            str: Next node to call.
        """
        context = state.get("context", "")
        if not context or len(context.strip()) == 0:
            if self.__enable_internet_search:
                print("Step: No relevant documents found. Routing to Web Search.")
                return "web_search"
            else:
                print("Step: No relevant documents found and Internet Search disabled. Routing to Generation.")
                return "generate"
        else:
            print("Step: Relevant documents found. Routing to Generation.")
            return "generate"

    def route_question(self, state: dict) -> str:
        """Route question to document search or generation.

        Uses the router template to determine whether the question may be
        answered from the local document store (doc_search) or directly from
        the model's internal knowledge (generate).

        Args:
            state (dict): The current graph state.

        Returns:
            str: Next node to call — either 'doc_search' or 'generate'.
        """
        print("Step: Routing Query")
        question = state['question']
        output = self.__agent_prompt.get_router_chain.invoke(
            {"question": question}
        )

        # Default to doc_search so we always try the document store when uncertain
        choice = output.get('choice', 'doc_search')
        if choice not in ('doc_search', 'web_search', 'generate'):
            choice = 'doc_search'

        print(f"Step: Router Decision: {choice}")
        if choice == "doc_search":
            print("Step: Routing Query to Document Search")
        elif choice == "web_search":
            print("Step: Routing Query to Web Search (via Query Transform)")
        else:
            print("Step: Routing Query to Generation")
        return choice
