"""Main training script for the Fitness RL Agent."""

import asyncio
import os
import random

import art
import weave
from art.local import LocalBackend
from dotenv import load_dotenv

from src.config import (
    AgentConfig,
    APIConfig,
    ModelConfig,
    PineconeConfig,
    TrainingConfig,
)
from src.data_loader import load_fitness_scenarios
from src.services import PineconeService
from src.training.trainer import train_agent


def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment."
        )

    # Optional but recommended
    if not os.getenv("WANDB_API_KEY"):
        print(
            "Warning: WANDB_API_KEY is not set. "
            "Skipping logging metrics to Weights & Biases."
        )


async def main():
    """Main training function."""
    # Load environment variables
    load_dotenv()

    # Validate environment
    validate_environment()

    # Set random seed for reproducibility
    random.seed(42)

    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    agent_config = AgentConfig()
    pinecone_config = PineconeConfig()
    api_config = APIConfig()

    print("=" * 80)
    print("Fitness RL Agent Training")
    print("=" * 80)
    print(f"Base Model: {model_config.base_model}")
    print(f"Model Name: {model_config.name}")
    print(f"Project: {model_config.project}")
    print(f"Max Epochs: {training_config.num_epochs}")
    print(f"Max Steps: {training_config.max_steps}")
    print(f"Groups per Step: {training_config.groups_per_step}")
    print(f"Rollouts per Group: {training_config.rollouts_per_group}")
    print(f"Learning Rate: {training_config.learning_rate}")
    print("=" * 80)

    # Load training data
    print("\nLoading training scenarios...")
    scenarios = load_fitness_scenarios("fitness_scenarios.jsonl")
    print(f"Loaded {len(scenarios)} scenarios")

    # Initialize Pinecone service
    print("\nInitializing Pinecone service...")
    pinecone_service = PineconeService(pinecone_config)
    print("Pinecone service initialized")

    # Create ART model
    print("\nInitializing ART model...")
    model = art.TrainableModel(
        name=model_config.name,
        project=model_config.project,
        base_model=model_config.base_model,
    )

    # Set internal config for the model
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=model_config.max_seq_length,
        ),
        engine_args=art.dev.EngineArgs(
            enforce_eager=model_config.enforce_eager,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
        ),
    )

    # Initialize the backend
    print("\nInitializing ART backend...")
    backend = LocalBackend(
        in_process=False,  # Set to True for debugging, False for production
        path="./.art",
    )

    # Register the model with the backend
    print("Registering model with backend...")
    await model.register(backend)
    print("Model registered successfully")

    # Initialize Weave for logging (if WANDB_API_KEY is set)
    if api_config.wandb_api_key:
        print("\nInitializing Weave logging...")
        weave.init(model.project, settings={"print_call_link": False})
        print("Weave logging initialized")

    # Start training
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    await train_agent(
        model=model,
        scenarios=scenarios,
        pinecone_service=pinecone_service,
        training_config=training_config,
        agent_config=agent_config,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

