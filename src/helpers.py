# =============================================================================
# Helper Function: Beautify the LLM Text Response
# =============================================================================
def beautify_response(response: str) -> str:
    """
    Formats the LLM response with markdown headers and separators for clarity.
    
    Args:
        response (str): The raw text response from the LLM.
        
    Returns:
        str: A formatted version of the response.
    """
    # Strip any extra whitespace and add markdown formatting.
    beautified = (
        "## LLM Response\n\n"
        + response.strip() +
        "\n\n---\n"
    )
    return beautified

# from googletrans import Translator
import asyncio
# =============================================================================
# Helper: Asynchronous Translation Function
# =============================================================================
# async def async_translate(text: str, dest='en') -> str:
#     """
#     Asynchronously translates text to English using googletrans.

#     Args:
#         text (str): Text in any language.
#     Returns:
#         str: The translated English text.
#     """
#     translator = Translator()
#     # Await the translate coroutine
#     translation = await translator.translate(text, dest=dest)
#     return translation.text

# def translate(text: str, dest='en') -> str:
#     """
#     Synchronously translates text to English by running the async translator.
    
#     Args:
#         text (str): Text in any language.
#     Returns:
#         str: The translated English text.
#     """
#     return asyncio.run(async_translate(text, dest=dest))


import pandas as pd


# Get the exercise list once at module load time
def get_exercise_list_for_prompt():
    try:
        exercise_df = pd.read_csv("data/joint_exercises_metadata.csv")
        exercise_list = {}
        
        for _, row in exercise_df.iterrows():
            body_part = row['body_part']
            if body_part not in exercise_list:
                exercise_list[body_part] = []
            
            exercise_list[body_part].append(row['exercise_name'])
        
        # Format as text
        result = "# Exercise Database\n\n"
        for body_part, exercises in exercise_list.items():
            result += f"## {body_part}\n"
            for exercise in exercises:
                result += f"- {exercise}\n"
            result += "\n"
            
        return result
    except Exception as e:
        return f"Error loading exercise list: {str(e)}"
    
from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

def extract_primary_worker_reply(run_state):
    """
    Works with:
      - dict with agent blocks (e.g., {"supervisor": {...}, "workerA": {...}})
      - dict with a single "messages" list
      - a raw list of messages
    Returns: (agent_name, content, all_last_replies_by_agent_dict)
    """
    # --- helpers to read both dicts and LC message objects ---
    def class_name(x):
        return getattr(x, "__class__", type(x)).__name__

    def get(field, m, default=None):
        if isinstance(m, dict):
            return m.get(field, default)
        return getattr(m, field, default)

    def get_response_metadata(m):
        rm = get("response_metadata", m, {}) or {}
        # sometimes stored under additional_kwargs, mirror __is_handoff_back
        if not rm and isinstance(m, dict):
            rm = m.get("additional_kwargs", {}) or {}
        else:
            add = get("additional_kwargs", m, {}) or {}
            rm = {**add, **rm}
        return rm

    def is_tool_message(m):
        # ToolMessage usually has tool_call_id or type == 'tool'
        if get("tool_call_id", m) is not None:
            return True
        t = get("type", m)
        if t == "tool":  # langchain ToolMessage.type == 'tool'
            return True
        return class_name(m) == "ToolMessage"

    def is_handoff(m):
        rm = get_response_metadata(m)
        return bool(rm.get("__is_handoff_back"))

    def name_of(m):
        return get("name", m)

    def content_of(m):
        c = get("content", m)
        # Content might be non-string (list/parts); only accept non-empty strings
        return c if isinstance(c, str) and c.strip() else ""

    def is_ai_message(m):
        if is_tool_message(m):
            return False
        t = get("type", m)
        if t == "ai":
            return True
        return class_name(m) == "AIMessage"

    # --- normalize to a flat list of messages ---
    def flatten_messages(state):
        # raw list
        if isinstance(state, list):
            return state
        # dict with "messages"
        if isinstance(state, dict) and isinstance(state.get("messages"), list):
            return state["messages"]
        # dict of agent blocks
        msgs = []
        if isinstance(state, dict):
            for _, block in state.items():
                if isinstance(block, dict) and isinstance(block.get("messages"), list):
                    msgs.extend(block["messages"])
        return msgs

    msgs = flatten_messages(run_state)

    # --- collect last contentful AI messages by agent (excluding supervisor, tool msgs, and handoffs) ---
    all_last = {}
    candidates = []
    for m in msgs:
        if not is_ai_message(m):
            continue
        if is_handoff(m):
            continue
        nm = name_of(m)
        if nm == "supervisor":
            continue
        c = content_of(m)
        if not c:
            continue
        all_last[nm] = c
        candidates.append(m)

    if not candidates:
        return (None, None, {})

    winner = candidates[-1]  # most recent in the list
    return (name_of(winner), content_of(winner), all_last)