from typing import Any

from langgraph.graph import StateGraph

from .langGraph_nodes import (
    PDFtoImages_node,
    classify_pdf_files,
    cross_validate_node,
    final_decision_node,
)
from .langGraph_states import ClaimState

# MAX_TOKENS = 4096

graph = StateGraph(ClaimState)
graph.add_node("PDFtoImages", PDFtoImages_node)  # Parallel entry
graph.add_node("classifyPDFfiles", classify_pdf_files)
graph.add_node("cross_validate", cross_validate_node)
graph.add_node("decide", final_decision_node)
# ------- node relationships --------
graph.set_entry_point("PDFtoImages")
graph.add_edge("PDFtoImages", "classifyPDFfiles")
graph.add_edge("classifyPDFfiles", "cross_validate")
graph.add_edge("cross_validate", "decide")
graph.set_finish_point("decide")

compiled_graph = graph.compile()


async def invoke(state: ClaimState) -> dict[str, str | Any]:
    """Invoke the comiled graph

    :param state: state dataclass object created inside fastapi endpoint function
    :return: ClaimState object after processing
    """
    result_state = await compiled_graph.ainvoke(state)
    # need to debug to see if result is a ClaimState object
    return result_state
