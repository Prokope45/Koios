"""AgentPrompt.py

Agent prompt class for creating prompt chains to be used in workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
import re, os
import time
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from ddgs import DDGS

from src.koios.enums.Template import Template
from src.koios.ReadTemplate.ReadTemplate import ReadTemplate


class AgentPrompt:
    """Create prompt chains for invoking agent workflow actions."""

    # Class-level variable to track last DuckDuckGo search time
    _last_ddg_search_time = 0

    def __init__(self, model: str, temperature: float) -> None:
        """Construct AgentPrompt object.

        Args:
            model (str): Selected model to load.
            temperature (float): Model temperature to use when generating.
        """
        self.__model = model
        self.__temperature = temperature
        self.__read_prompt = ReadTemplate()
        self.__base_url = os.getenv("OPENAI_URL")

    @staticmethod
    def get_available_models() -> list[str]:
        """Fetch available models from the OpenAI-compatible API.

        Returns:
            list[str]: List of model IDs that are currently loaded on the server.
        """
        try:
            # Use the configured OPENAI_URL; fall back to default if not set.
            base_url = os.getenv('OPENAI_URL')
            if not base_url:
                base_url = 'http://127.0.0.1:1234'   # default local server
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                # OpenAI-compatible API returns a list of model objects with an 'id' field
                return [model["id"] for model in response.json().get("data", [])]
        except Exception:
            pass
        return ["llama3.2"]

    def web_search_with_fallback(self, query: str) -> str:
        """Perform web search with fallback to Wikipedia on rate limit.

        Args:
            query (str): The search query.

        Returns:
            str: Search results or Wikipedia summary.
        """
        try:
            # Enforce rate limit: DuckDuckGo allows 1 request per second
            current_time = time.time()
            time_since_last_search = current_time - AgentPrompt._last_ddg_search_time
            
            if time_since_last_search < 1.0:
                # Wait for the remaining time to respect the 1-second rate limit
                sleep_time = 1.0 - time_since_last_search
                print(f"Rate limiting: waiting {sleep_time:.2f}s before DuckDuckGo search...")
                time.sleep(sleep_time)

            # Update the last search time
            AgentPrompt._last_ddg_search_time = time.time()

            with DDGS() as ddgs:
                result = ddgs.text(query, safesearch="moderate", max_results=10, page=1)
                print(result)
                return result
        except Exception as e:
            print(f"DuckDuckGo search failed or rate limited: {e}")
            print("Falling back to Wikipedia...")
            try:
                wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
                return wiki.invoke(query)
            except Exception as wiki_e:
                return f"Search failed: {e}. Fallback failed: {wiki_e}"

    @property
    def get_generate_chain(self) -> str:
        """Generation stage of agent.

        Creates template where questions and context are dynamically
        inserted. The template instructs the agent to synthesize the search
        results into a research report. "I don't know.." is allowed if no
        additional information is provided in the context.

        Returns:
            str: String chain of prompt template processed by LLM.
        """
        template_content = self.__read_prompt.get_contents(Template.GENERATE)
        
        # Add history to the template if it's not there
        if "{history}" not in template_content:
            template_content = "Conversation History: {history}\n\n" + template_content

        generate_prompt = PromptTemplate(
            template=template_content,
            input_variables=["question", "context", "history"],
        )

        # Chain (pipes between each operation)
        # StrOutputParser ensures result is in plain-text
        llm = ChatOpenAI(
            base_url=f"{self.__base_url}/v1",
            api_key="lm-studio",
            model=self.__model,
            temperature=self.__temperature
        )
        generate_chain = generate_prompt | llm | StrOutputParser() | self.__remove_special_tokens

        return generate_chain

    @property
    def get_router_chain(self) -> str:
        """Calls `__prompt_using_json` using default template (router).

        Returns:
            str: String chain of prompt template processed by LLM.
        """
        return self.__prompt_using_json()

    @property
    def get_query_chain(self):
        """Calls `__prompt_using_json` using query template.

        Returns:
            str: String chain of prompt template processed by LLM.
        """
        return self.__prompt_using_json(Template.QUERY)

    def __remove_special_tokens(self, text: str) -> str:
        """Remove Llama 3 special tokens from LLM output.

        Args:
            text (str): Raw LLM output.

        Returns:
            str: Cleaned output.
        """
        special_tokens = [
            "<|begin_of_text|>",
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token"
        ]
        for token in special_tokens:
            text = text.replace(token, "")
        return text.strip()

    def __extract_json(self, text: str) -> str:
        """Extract JSON block from text.

        Args:
            text (str): Text containing JSON.

        Returns:
            str: Extracted JSON string.
        """
        if "{" in text and "}" in text:
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return match.group(1)
        return text

    def __prompt_using_json(self, template: Template = Template.ROUTER) -> str:
        """Generation stage of agent.

        Helper method for creating chain string depending on whether template
        is router or query. If router, the agent will determine if the question
        can be answered using it's knowledge. If query, the agent will reword
        the query in preparation for a web search to add context to its answer.

        Args:
            template (Template, optional): Template enum to get file path.
                Defaults to Template.ROUTER.

        Returns:
            str: String chain of prompt template processed by LLM.
        """
        router_prompt = PromptTemplate(
            template=self.__read_prompt.get_contents(template),
            input_variables=["question"],
            )

        # Use temperature 0 for deterministic routing and query transformation
        llm = ChatOpenAI(
            base_url=f"{self.__base_url}/v1",
            api_key="lm-studio",
            model=self.__model,
            temperature=0
        )

        # We use StrOutputParser first, then remove tokens, then extract JSON, then JsonOutputParser
        chain = (
            router_prompt | llm | StrOutputParser() | self.__remove_special_tokens | self.__extract_json | JsonOutputParser()
        )
        return chain
