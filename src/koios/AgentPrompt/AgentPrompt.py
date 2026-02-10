"""AgentPrompt.py

Agent prompt class for creating prompt chains to be used in workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
import re, os
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from src.koios.enums.Template import Template
from src.koios.ReadTemplate.ReadTemplate import ReadTemplate


class AgentPrompt:
    """Create prompt chains for invoking agent workflow actions."""

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
        self.__load_model(model)

    def __load_model(self, model_key: str) -> None:
        """Explicitly load a model in LM Studio.

        Args:
            model_key (str): The key of the model to load.
        """
        try:
            # LM Studio API for loading a model
            # Note: The exact endpoint might vary, but /api/v1/model/load is common
            requests.post(
                f"{self.__base_url}/api/v1/model/load",
                json={"modelKey": model_key},
                timeout=5
            )
        except Exception:
            pass

    @staticmethod
    def get_available_models() -> list[str]:
        """Fetch available models from LM Studio API.

        Returns:
            list[str]: List of model keys.
        """
        try:
            response = requests.get(f"{os.getenv('OPENAI_URL', '')}/api/v1/models", timeout=2)
            if response.status_code == 200:
                return [model["key"] for model in response.json().get("models", [])]
        except Exception:
            pass
        return ["llama3.2"]

    @property
    def get_web_search_tool(self) -> DuckDuckGoSearchRun:
        """Web search stage of agent.

        Returns:
            DuckDuckGoSearchRun: Runnable tool for web search.
        """
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
        web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        return web_search_tool

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
        generate_prompt = PromptTemplate(
            template=self.__read_prompt.get_contents(Template.GENERATE),
            input_variables=["question", "context"],
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

        llm = ChatOpenAI(
            base_url=f"{self.__base_url}/v1",
            api_key="lm-studio",
            model=self.__model,
            temperature=self.__temperature
        )

        # We use StrOutputParser first, then remove tokens, then extract JSON, then JsonOutputParser
        chain = (
            router_prompt | llm | StrOutputParser() | self.__remove_special_tokens | self.__extract_json | JsonOutputParser()
        )
        return chain
