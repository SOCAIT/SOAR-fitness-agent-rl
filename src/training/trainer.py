"""Training loop for the fitness agent."""

import art
from art.langgraph import wrap_rollout
from art.utils import iterate_dataset

from src.agent.rollout import rollout
from src.config import AgentConfig, TrainingConfig
from src.models import FitnessScenario, Scenario
from src.services import PineconeService


async def train_agent(
    model: art.TrainableModel,
    scenarios: list[Scenario],
    pinecone_service: PineconeService,
    training_config: TrainingConfig,
    agent_config: AgentConfig,
):
    """
    Train the fitness agent.

    Args:
        model: The ART trainable model
        scenarios: List of training scenarios
        pinecone_service: Pinecone service instance
        training_config: Training configuration
        agent_config: Agent configuration
    """
    # Use iterate_dataset with real training scenarios
    training_iterator = iterate_dataset(
        scenarios,
        groups_per_step=training_config.groups_per_step,
        num_epochs=training_config.num_epochs,
    )

    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, "
            f"epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")

        # Create trajectory groups for this batch
        groups = []
        for scenario_data in batch.items:
            scenario = scenario_data
            print(f"Processing scenario: {scenario.id}")

            groups.append(
                art.TrajectoryGroup(
                    (
                        wrap_rollout(model, rollout)(
                            model,
                            FitnessScenario(
                                step=batch.step, scenario=scenario.model_dump()
                            ),
                            pinecone_service,
                            agent_config,
                        )
                        for _ in range(training_config.rollouts_per_group)
                    )
                )
            )

        print(f"Created {len(groups)} trajectory groups")

        # Gather all trajectory groups
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="gather",
            max_exceptions=training_config.rollouts_per_group * len(batch.items),
        )

        print(f"Finished gathering {len(finished_groups)} groups")

        # Train the model on the gathered trajectories
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=training_config.learning_rate),
            # Lowering the logprob_calculation_chunk_size is a memory saving measure
            # to allow longer sequences (up to 8192 tokens) to be processed on a T4.
            _config={
                "logprob_calculation_chunk_size": training_config.logprob_calculation_chunk_size
            },
        )

        print(f"Completed training step {batch.step}")

        # Stop after max_steps
        if batch.step >= training_config.max_steps:
            print(f"Reached max_steps ({training_config.max_steps}), stopping training")
            break

    print("Training completed!")

