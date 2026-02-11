"""AgentWorkflow.py

Agent workflow class establishing decision pathway for agent. First determines
if response generation should be done initially or if web search is needed to
add more context. Then goes through the graph to the appropriate node, either
skipping directly to generation or making web search then generating a
response.

Path:
Generate OR transform query into web-readable query -> search web -> generate

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from langgraph.graph import END, StateGraph

from src.koios.GraphState.GraphState import GraphState
from src.koios.AgentWorkflow.WorkflowActions import WorkflowActions
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt


class AgentWorkflow:
    """AgentWorkflow class that contains the workflow for the agent."""

    def __init__(self, model: str, temperature: float, enable_internet_search: bool = False) -> None:
        """Construct AgentWorkflow object and initializes workflow.

        Args:
            model (str): Selected model to load.
            temperature (float): Model temperature to use when generating.
            enable_internet_search (bool): Whether to allow web search.
        """
        agent_prompt: AgentPrompt = AgentPrompt(model, temperature)
        actions = WorkflowActions(agent_prompt, enable_internet_search)

        workflow = StateGraph(GraphState)
        workflow.add_node("web_search", actions.web_search)
        workflow.add_node("doc_search", actions.doc_search)
        workflow.add_node("transform_query", actions.transform_query)
        workflow.add_node("generate", actions.generate)

        workflow.set_conditional_entry_point(
            actions.route_question,
            {
                "doc_search": "doc_search",
                "generate": "generate",
            },
        )
        
        workflow.add_conditional_edges(
            "doc_search",
            actions.decide_after_doc_search,
            {
                "web_search": "transform_query",
                "generate": "generate",
            }
        )
        
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        self.__local_agent = workflow.compile()

    @property
    def local_agent(self) -> StateGraph:
        """Getter property for local agent.

        Returns:
            StateGraph: Agent state graph object.
        """
        return self.__local_agent
