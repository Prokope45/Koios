"""AgentPrompt.py

Agent prompt class for creating prompt chains to be used in workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from koios.enums.Template import Template
from koios.ReadTemplate.ReadTemplate import ReadTemplate


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
        llama3 = ChatOllama(
            model=self.__model,
            temperature=self.__temperature
        )
        generate_chain = generate_prompt | llama3 | StrOutputParser()

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

        llama3_json = ChatOllama(
            model=self.__model,
            temperature=self.__temperature,
            format='json'
        )
        chain = (
            router_prompt | llama3_json | JsonOutputParser()
        )
        return chain
