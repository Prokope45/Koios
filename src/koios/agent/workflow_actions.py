"""WorkflowActions.py

Workflow actions class containing actions for agent to take.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI

from src.koios.agent.prompt import Prompt
from src.koios.data_store.DocumentStore import DocumentStore
from src.koios.toon_serializer.ToonSerializer import ToonSerializer
from src.config import logger

# System prompt that instructs the LLM to reformulate the user's question
# into a standalone query that can be understood without the chat history.
_CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

_CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


class WorkflowActions:
    """Provide workflow actions for agent to take."""

    def __init__(self, agent_prompt: Prompt, enable_internet_search: bool = False):
        """Construct WorkflowActions object.

        Args:
            agent_prompt (Prompt): Prompt object to use for getting
                chains.
            enable_internet_search (bool): Whether to allow web search.
        """
        self.__agent_prompt = agent_prompt
        self.__enable_internet_search = enable_internet_search
        self.__doc_store = DocumentStore()

        # Build the history-aware retriever once at construction time.
        # It uses a small, fast LLM to reformulate the user's question into a
        # standalone query before hitting the vector store.
        base_url = os.getenv("OPENAI_URL", "http://127.0.0.1:1234")
        _llm = ChatOpenAI(
            base_url=f"{base_url}/v1",
            api_key="lm-studio",
            model=agent_prompt.model,
            temperature=0,
        )
        self.__history_aware_retriever = create_history_aware_retriever(
            llm=_llm,
            retriever=self.__doc_store.get_retriever(),
            prompt=_CONTEXTUALIZE_Q_PROMPT,
        )

    def generate(self, state: dict) -> dict:
        """Generate answer based on existing knowledge.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): New key added to state, generation, containing
                LLM generation.
        """
        logger.info("Step: Generating Final Response")
        question = state["question"]
        history = state.get("history", [])
        
        # Ensure context is not None or empty if we skipped web search
        context = state.get("context")
        if not context:
            context = "No additional context provided. Answer based on your internal knowledge."
        results = {"context": context, "question": question, "history": history}
        generation = self.__agent_prompt.get_generate_chain.invoke(results)
        return {"generation": generation}

    def web_search(self, state: dict) -> dict:
        """Optimize the user query and perform a web search.

        The raw question is first transformed into an optimized search query
        via the query chain, then the search is executed against the web.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Appended web results to context.
        """
        question = state['question']
        logger.info("Step: Optimizing Query for Web Search")
        gen_query = self.__agent_prompt.get_query_chain.invoke(
            {"question": question}
        )
        search_query = gen_query["query"]
        logger.info(f'Step: Searching the Web for: "{search_query}"')
        search_result = self.__agent_prompt.web_search_with_fallback(
            search_query
        )
        # Encode the list of {"title", "href", "body"} dicts as TOON.
        return {"context": ToonSerializer.dumps({"results": search_result})}

    @staticmethod
    def _to_langchain_messages(history: list) -> list[BaseMessage]:
        """Convert Streamlit-style history dicts to LangChain message objects.

        Streamlit stores messages as `{"role": "user"|"assistant", "content": "..."}`.
        LangChain's history-aware retriever expects `HumanMessage` /
        `AIMessage` instances.

        Args:
            history (list): List of `{"role", "content"}` dicts.

        Returns:
            list[BaseMessage]: Equivalent LangChain message objects.
        """
        messages: list[BaseMessage] = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def doc_search(self, state: dict) -> dict:
        """Search document store based on the question using history-aware retrieval.

        The history-aware retriever first reformulates the user's question into
        a standalone query (using the chat history for context) and then
        performs similarity search against the vector store.  Retrieved
        documents are encoded as TOON before being stored in the graph state
        so that the downstream generate prompt receives a token-efficient
        representation of the document context.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Appended document results to context.
        """
        question = state["question"]
        raw_history = state.get("history", [])
        chat_history = self._to_langchain_messages(raw_history)

        logger.info(f'Step: Searching Document Store for: "{question}"')
        if chat_history:
            logger.info("Step: Reformulating query with chat history context")

        docs = self.__history_aware_retriever.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        # Build a list of document dicts and encode as TOON.
        # Each document is represented with its source metadata (if available)
        # and its text content.
        doc_records = [
            {
                "source": doc.metadata.get("source", "unknown") if doc.metadata else "unknown",
                "content": doc.page_content,
            }
            for doc in docs
        ]
        return {"context": ToonSerializer.dumps({"documents": doc_records})}

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
                logger.info("Step: No relevant documents found. Routing to Web Search.")
                return "web_search"
            else:
                logger.info("Step: No relevant documents found and Internet Search disabled. Routing to Generation.")
                return "generate"
        else:
            logger.info("Step: Relevant documents found. Routing to Generation.")
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
        logger.info("Step: Routing Query")
        question = state['question']
        output = self.__agent_prompt.get_router_chain.invoke(
            {"question": question}
        )
        logger.info(f"Chain output: {output!r}")

        # Default to doc_search so we always try the document store when uncertain
        choice = output.get('choice', 'doc_search')
        if choice not in ('doc_search', 'web_search', 'generate'):
            logger.warning(
                "Router returned unrecognized choice %r; defaulting to doc_search.",
                choice,
            )
            choice = 'doc_search'

        # Force doc_search if model attempts disabled web_search
        if choice == 'web_search' and not self.__enable_internet_search:
            logger.info(
                "Step: Router chose web_search but internet search is disabled; "
                "falling back to doc_search."
            )
            choice = 'doc_search'

        logger.info(f"Step: Router Decision: {choice}")
        if choice == "doc_search":
            logger.info("Step: Routing Query to Document Search")
        elif choice == "web_search":
            logger.info("Step: Routing Query to Web Search")
        else:
            logger.info("Step: Routing Query to Generation")
        return choice
