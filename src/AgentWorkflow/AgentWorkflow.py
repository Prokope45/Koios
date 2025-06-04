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

from src.GraphState.GraphState import GraphState
from src.AgentWorkflow.WorkflowActions import WorkflowActions


class AgentWorkflow:
    """AgentWorkflow class that contains the workflow for the agent."""

    def __init__(self) -> None:
        """Construct AgentWorkflow object and initializes workflow."""
        workflow = StateGraph(GraphState)

        actions = WorkflowActions()
        workflow.add_node("web_search", actions.web_search)
        workflow.add_node("transform_query", actions.transform_query)
        workflow.add_node("generate", actions.generate)

        workflow.set_conditional_entry_point(
            actions.route_question,
            {
                "web_search": "transform_query",
                "generate": "generate",
            },
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
