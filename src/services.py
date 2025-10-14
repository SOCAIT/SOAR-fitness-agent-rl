"""External services integration (Pinecone, LLM judge, etc.)."""

from typing import List

from litellm import acompletion
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt

from src.config import APIConfig, PineconeConfig
from src.models import NutritionJudgeResponse, Scenario


class PineconeService:
    """Service for interacting with Pinecone vector database."""

    def __init__(self, config: PineconeConfig):
        """Initialize Pinecone service."""
        self.config = config
        self.pc = Pinecone(api_key=config.api_key)
        self.recipe_index = self.pc.Index(config.recipe_index_name)
        self.exercise_index = self.pc.Index(config.exercise_index_name)

    def search_recipes(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for recipes using semantic search.

        Args:
            query: The search query (e.g., "Chicken and rice healthy")
            top_k: Number of results to return

        Returns:
            List of recipe names
        """
        results = self.recipe_index.search(
            namespace=self.config.namespace,
            query={"top_k": top_k, "inputs": {"text": query}},
        )

        return self._extract_meal_names(results)

    def search_exercises(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for exercises using semantic search.

        Args:
            query: The search query (e.g., "chest exercises")
            top_k: Number of results to return

        Returns:
            List of exercise names
        """
        results = self.exercise_index.search(
            namespace=self.config.namespace,
            query={"top_k": top_k, "inputs": {"text": query}},
        )

        return self._extract_exercise_names(results)

    @staticmethod
    def _extract_meal_names(data: dict) -> List[str]:
        """Extract meal names from Pinecone search results."""
        data = data.get("result", data)
        return [
            hit["fields"]["name"]
            for hit in data.get("hits", [])
            if "fields" in hit and "name" in hit["fields"]
        ]

    @staticmethod
    def _extract_exercise_names(data: dict) -> List[str]:
        """Extract exercise names from Pinecone search results."""
        data = data.get("result", data)
        return [
            hit["fields"]["name"]
            for hit in data.get("hits", [])
            if "fields" in hit and "name" in hit["fields"]
        ]


class JudgeService:
    """Service for judging agent responses using LLM."""

    def __init__(self, config: APIConfig):
        """Initialize judge service."""
        self.config = config

    @retry(stop=stop_after_attempt(3))
    async def judge_nutrition_plan(
        self, scenario: Scenario, answer: str, system_prompt: str
    ) -> NutritionJudgeResponse:
        """
        Judge a nutrition plan using an LLM.

        Args:
            scenario: The scenario/question
            answer: The agent's answer
            system_prompt: The system prompt for judging

        Returns:
            NutritionJudgeResponse with score and reasoning
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {scenario.question}\n"
                    f"Assistant Answer: {answer}\n"
                    f"AI answer: {answer}"
                ),
            },
        ]

        response = await acompletion(
            model=self.config.judge_model,
            messages=messages,
            response_format=NutritionJudgeResponse,
        )

        # Access the response content
        content = response.choices[0].message.content
        return NutritionJudgeResponse(
            reasoning=content.reasoning, score=content.score
        )

