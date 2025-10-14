"""LangGraph tools for the fitness agent."""

import json
from datetime import datetime
from functools import wraps
from typing import Callable, List

from langchain_core.tools import tool

from src.models import FinalAnswer
from src.services import PineconeService


def create_logging_decorator(trajectory_messages: List[dict]):
    """
    Create a decorator that logs tool calls to the trajectory.

    Args:
        trajectory_messages: List to append tool logs to

    Returns:
        Decorator function
    """

    def log_tool(tool_name: str):
        def decorator(fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                GREEN = "\033[92m"
                RESET = "\033[0m"

                call = {
                    "tool": tool_name,
                    "ts": datetime.utcnow().isoformat(),
                    "args": args,
                    "kwargs": kwargs,
                }

                color_prefix = GREEN if "final_answer" in tool_name else ""
                color_reset = RESET if "final_answer" in tool_name else ""

                print(f"{color_prefix}[TOOL START] {tool_name} args={kwargs}{color_reset}")
                trajectory_messages.append(
                    {"role": "tool_log", "content": json.dumps({"start": call})}
                )

                try:
                    out = fn(*args, **kwargs)
                    print(
                        f"{color_prefix}[TOOL END] {tool_name} result_preview={str(out)[:400]}{color_reset}"
                    )
                    trajectory_messages.append(
                        {
                            "role": "tool_log",
                            "content": json.dumps({"end": {**call, "result": out}}),
                        }
                    )
                    return out
                except Exception as e:
                    print(f"\033[91m[TOOL ERROR] {tool_name}: {e}\033[0m")
                    trajectory_messages.append(
                        {
                            "role": "tool_log",
                            "content": json.dumps(
                                {"error": {**call, "error": str(e)}}
                            ),
                        }
                    )
                    raise

            return wrapper

        return decorator

    return log_tool


def get_payload(obj):
    """
    Return a dict payload from FinalAnswer/str/dict.

    Args:
        obj: Object to extract payload from

    Returns:
        Dictionary payload
    """
    # unwrap FinalAnswer wrapper
    if hasattr(obj, "answer"):
        obj = obj.answer

    # parse JSON string
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            # return sentinel dict so schema clearly fails
            return {"_error": "invalid_json_string", "_raw": obj}

    # pydantic compatibility
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "dict"):
        obj = obj.dict()

    return (
        obj
        if isinstance(obj, dict)
        else {"_error": f"unexpected_type:{type(obj).__name__}", "_raw": str(obj)}
    )


def create_agent_tools(
    pinecone_service: PineconeService,
    trajectory_messages: List[dict],
    final_answer_ref: List,  # Using list as mutable container for nonlocal reference
    top_k: int = 5,
):
    """
    Create the tools for the LangGraph agent.

    Args:
        pinecone_service: Pinecone service instance
        trajectory_messages: List to log tool calls to
        final_answer_ref: Mutable reference to store final answer
        top_k: Number of results to return from searches

    Returns:
        List of LangGraph tools
    """
    log_tool = create_logging_decorator(trajectory_messages)

    @tool
    @log_tool("recipe_semantic_search")
    def recipe_semantic_search(meal_query: str, k: int = top_k) -> str:
        """Search the recipe index for the most similar recipes to the query."""
        results = pinecone_service.search_recipes(meal_query, top_k=k)
        print(f"Recipe search results: {results}")
        return str(results)

    @tool
    @log_tool("return_final_answer_tool")
    def return_final_answer_tool(answer: str) -> dict:
        """Return the final answer (daily meal plan) in the correct format."""
        payload = get_payload(answer)  # normalize here
        final_answer = FinalAnswer(answer=payload)
        final_answer_ref[0] = final_answer  # Store in reference
        return final_answer.model_dump()

    return [recipe_semantic_search, return_final_answer_tool]

