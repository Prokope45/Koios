"""AgentPrompt.py

Agent prompt class for creating prompt chains to be used in workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
import os
import time
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from ddgs import DDGS

from src.koios.enums.Template import Template
from src.koios.read_template.ReadTemplate import ReadTemplate
from src.config import logger


class Prompt:
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

    @property
    def model(self) -> str:
        """Getter for the model name.

        Returns:
            str: The model identifier string.
        """
        return self.__model

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

        Results from DuckDuckGo are encoded as TOON before being returned so
        that the downstream generate prompt receives a token-efficient
        representation of the search context.

        Args:
            query (str): The search query.

        Returns:
            str: TOON-encoded search results, or a Wikipedia summary string
                on fallback.
        """
        try:
            # Enforce rate limit: DuckDuckGo allows 1 request per second
            current_time = time.time()
            time_since_last_search = current_time - Prompt._last_ddg_search_time

            if time_since_last_search < 1.0:
                # Wait for the remaining time to respect the 1-second rate limit
                sleep_time = 1.0 - time_since_last_search
                logger.info(f"Rate limiting: waiting {sleep_time:.2f}s before DuckDuckGo search...")
                time.sleep(sleep_time)

            # Update the last search time
            Prompt._last_ddg_search_time = time.time()

            with DDGS() as ddgs:
                results = ddgs.text(query, safesearch="moderate", max_results=3, page=1)
                logger.debug("DuckDuckGo results: %s", results)
            return results

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed or rate limited: {e}")
            logger.info("Falling back to Wikipedia...")
            try:
                wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
                return wiki.invoke(query)
            except Exception as wiki_e:
                return f"Search failed: {e}. Fallback failed: {wiki_e}"

    @property
    def get_generate_chain(self) -> str:
        """Generation stage of agent.

        Creates a model-aware prompt by applying the HuggingFace chat template
        for the loaded model. The template instructs the agent to synthesize
        the search results into a research report. "I don't know.." is allowed
        if no additional information is provided in the context.

        Returns:
            str: String chain of prompt template processed by LLM.
        """
        # Apply the model's chat template. LangChain {placeholders} are left
        # intact by Jinja2 and substituted by PromptTemplate at invoke time.
        formatted_template = self.__read_prompt.get_chat_prompt(
            self.__model,
            Template.GENERATE,
        )

        generate_prompt = PromptTemplate(
            template=formatted_template,
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
        generate_chain = generate_prompt | llm | StrOutputParser()

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

    def __prompt_using_json(self, template: Template = Template.ROUTER) -> str:
        """Generation stage of agent.

        Helper method for creating chain string depending on whether template
        is router or query. If router, the agent will determine if the question
        can be answered using it's knowledge. If query, the agent will reword
        the query in preparation for a web search to add context to its answer.

        The prompt is formatted using the HuggingFace chat template for the
        loaded model so that the correct special tokens are injected
        automatically — no post-processing cleanup is required.

        `JsonOutputParser.get_format_instructions()` is injected into the
        prompt as `{format_instructions}` so the model receives an explicit
        schema contract.  Templates that do not contain the placeholder (e.g.
        the query template) simply omit the variable from `input_variables`
        and the partial is not applied.

        Args:
            template (Template, optional): Template enum to get file path.
                Defaults to Template.ROUTER.

        Returns:
            str: String chain of prompt template processed by LLM.
        """
        # Apply the model's chat template. LangChain {placeholders} are left
        # intact by Jinja2 and substituted by PromptTemplate at invoke time.
        formatted_template = self.__read_prompt.get_chat_prompt(
            self.__model,
            template,
        )

        parser = JsonOutputParser()

        # Inject format instructions only when the template contains the
        # {format_instructions} placeholder so the model knows the exact
        # JSON schema it must produce.
        if "{format_instructions}" in formatted_template:
            prompt = PromptTemplate(
                template=formatted_template,
                input_variables=["question"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
        else:
            prompt = PromptTemplate(
                template=formatted_template,
                input_variables=["question"],
            )

        # Use temperature 0 for deterministic routing and query transformation
        llm = ChatOpenAI(
            base_url=f"{self.__base_url}/v1",
            api_key="lm-studio",
            model=self.__model,
            temperature=0
        )

        chain = prompt | llm | StrOutputParser() | parser
        return chain
