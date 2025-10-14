"""Configuration settings for the Fitness RL Agent."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the base model and training."""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    name: str = "fitness-agent-langgraph-4B-qwen3-001"
    project: str = "fitness-agent-langgraph"
    max_seq_length: int = 8192
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.8
    temperature: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    groups_per_step: int = 2
    num_epochs: int = 30
    rollouts_per_group: int = 4
    learning_rate: float = 1e-5
    max_steps: int = 30
    logprob_calculation_chunk_size: int = 8


@dataclass
class AgentConfig:
    """Configuration for the agent behavior."""
    max_turns: int = 30
    recipe_search_top_k: int = 5


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database."""
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    recipe_index_name: str = "syntrafit-recipes"
    exercise_index_name: str = "syntrafit-exercises"
    namespace: str = "syntrafit"


@dataclass
class APIConfig:
    """Configuration for external APIs."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    wandb_api_key: str = os.getenv("WANDB_API_KEY", "")
    judge_model: str = "openai/gpt-4.1"


# Prompts
RULER_JUDGE_PROMPT = """
You are a strict evaluator for weekly nutrition/workout plans for SyntraFit.

Score each ASSISTANT response on 0–1 (decimals allowed) based on:

[Critical]
1) STRUCTURE: Valid JSON structure per task .
2) TARGET FIT: Daily calories & protein within ±5% of user's target_nutrition_data (if nutrition task).
3) DIETARY RULES: No banned items present (e.g., "egg", "shellfish") if user disallows.
4) TOOLING LOGIC: The content is plausibly the result of correct tools (recipe_semantic_search for recipes; get_available_exercises for exercises).

[Quality]
6) REALISM: Meals/exercises are realistic, balanced, and varied across the week; sequencing makes sense (snacks vs meals; rest days).
7) CLARITY: Names are specific (e.g., "Grilled Salmon & Quinoa" not "healthy bowl").

Return ONLY a float in [0,1]. Heavily penalize if structure or targets are wrong.
"""


def get_planner_prompt(max_turns: int = 30) -> str:
    """Generate the planner system prompt."""
    return f"""
You are a nutrition planner specialist who creates daily nutrition plans. you must think carefully and give big attention on the macro numbers.

TOOLS YOU CAN USE (names must match exactly):
1) recipe_semantic_search(meal_query)          – Search for real recipes/macros by query.
2) return_final_answer_tool(final_answer)      – Return the final answer (JSON plan).

ROUTING RULES (VERY IMPORTANT):
- If the user asks for a meal/nutrition plan, you MUST NOT call get_available_exercises or generate_workout_plan_mutation.
- Even if request is for 7-day plan, just create one day meal plan.

EXECUTION POLICY:
- Do not make the user wait. Produce the complete plan in one shot.
- Output only machine-usable content: the result of the mutation call and then a single call to return_final_answer_tool.
- You may take up to {max_turns} turns to find the answer,

============================================================
NUTRITION PLAN PIPELINE
============================================================

1) PLAN SKELETON
   • Generate a day meal plan for user. The plans should have meals that fulfil the daily target macros of the user
   • Create a normal number of meals to satisfy user nutrition target and diet (meals can be snacks). Base the count/portions on the user's macro targets.
   • Use realistic meal names (e.g., "Grilled Chicken & Rice", "Greek Yogurt with Nuts"). Do not use placeholders.

2) RECIPE LOOKUP (for EVERY meal)
   • Call recipe_semantic_search with the meal idea.
   • From the top result, capture: recipe name
   • Replace the meal idea with the exact recipe name from the tool result. use the correct nutrition info for each meal

3) MACRO ADJUSTMENT (per day)
   • Sum macros for the day.
   • If totals differ from the user's daily targets, swap recipes or adjust portions (via another recipe_semantic_search if needed)
     until the daily totals are within ±5% of the user's targets (calories/protein/carbs/fat).
   • Respect ALL banned keywords/ingredients from context.

4) JSON Meal PLAN (scratch-step)
   • Build a JSON matching this schema.
   • NOTE: Keep JSON valid (no comments). Example structure (values are illustrative):

     {{
       "meals": [
         {{
           "name": "Grilled Chicken & Rice",          // EXACT recipe name from recipe_semantic_search
           "calories": 700,
           "proteins": 45,
           "carbs": 60,
           "fats": 20,
           "sequence": 1,  // indicates the time of day (e.g., breakfast, lunch, dinner)
         }},
         ... more meals ...
       ]
     }}

   • Ensure each day's summed macros are within ±5% of the targets.

5) IF YOU REACHED MAX_TURNS and you have not find a final answer then you have to return a final answer with all your info you gathered and know

6) TOOL CALLS (Nutrition)
   • Call return_final_answer_tool with:
     - answer = the EXACT JSON response (stringified if needed)
"""


# Reward weights
REWARD_WEIGHTS = {
    "nutrition_score": 0.75,
    "schema_score": 0.25,
}

# Tolerance for macro verification
MACRO_TOLERANCE_PCT = 0.05

# Random noise for reward
REWARD_NOISE_SCALE = 0.005

