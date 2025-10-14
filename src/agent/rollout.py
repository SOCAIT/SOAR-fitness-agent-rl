"""Rollout function for the fitness agent."""

import random
import uuid

import art
import weave
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from art.langgraph import init_chat_model

from src.agent.tools import create_agent_tools, get_payload
from src.config import AgentConfig, get_planner_prompt
from src.env.verifiers_utils import (
    verify_daily_meal_plan_macros,
    verify_meal_plan_schema,
)
from src.models import FinalAnswer, FitnessScenario
from src.services import PineconeService


class ProjectTrajectory(art.Trajectory):
    """Custom trajectory class with final answer field."""

    final_answer: FinalAnswer | None = None


@weave.op
async def rollout(
    model: art.Model,
    fitness_scenario: FitnessScenario,
    pinecone_service: PineconeService,
    agent_config: AgentConfig,
) -> ProjectTrajectory:
    """
    Execute a single rollout of the agent on a fitness scenario.

    Args:
        model: The ART trainable model
        fitness_scenario: The scenario to solve
        pinecone_service: Pinecone service for vector search
        agent_config: Agent configuration

    Returns:
        ProjectTrajectory with reward and messages
    """
    scenario = fitness_scenario.scenario

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": str(scenario.id),
            "step": fitness_scenario.step,
        },
    )

    system_prompt = get_planner_prompt(max_turns=agent_config.max_turns)

    # Store final answer (using list as mutable container)
    final_answer_ref = [None]

    # Target Nutrition
    daily_cal_target = scenario.daily_cal_target
    daily_prot_target = scenario.daily_prot_target
    daily_carb_target = scenario.daily_carb_target
    daily_fat_target = scenario.daily_fat_target

    # Create tools with logging
    tools = create_agent_tools(
        pinecone_service=pinecone_service,
        trajectory_messages=traj.messages_and_choices,
        final_answer_ref=final_answer_ref,
        top_k=agent_config.max_turns,
    )

    # Initialize chat model
    chat_model = init_chat_model(f"{model.name}", temperature=model._internal_config.init_args.temperature if hasattr(model._internal_config.init_args, 'temperature') else 1.0)

    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, tools)
    print("LangGraph agent created!")

    try:
        # Run the agent
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": agent_config.max_turns,
        }

        print(f"Human Question: {scenario.question}")

        # Run the agent to get the final result
        res = await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )

        print(f"Agent response received")

        # Check if we got a final answer
        final_answer = final_answer_ref[0]
        if final_answer:
            print("Got final answer!")
            print(final_answer)

            payload = get_payload(final_answer)  # normalize again defensively
            print(f"Payload: {payload}")

            # Calculate the total reward
            total_reward = 0.0
            traj.final_answer = final_answer

            # Score the trajectory - Schema validation
            schema_score, info = verify_meal_plan_schema(payload)
            print(f"Nutrition Schema score: {schema_score}, info: {info}")

            # Score the trajectory - Macro validation
            nutrition_score, info = verify_daily_meal_plan_macros(
                payload,
                daily_cal_target=daily_cal_target,
                daily_prot_target=daily_prot_target,
            )
            print(f"Nutrition score: {nutrition_score}, info: {info}")

            # Calculate total verifiable reward
            total_reward = 0.75 * nutrition_score + 0.25 * schema_score
            print(f"Total Verifiable Reward: {total_reward}")

            # Add small random noise to break ties
            random_noise = random.random()
            traj.reward = total_reward + random_noise * 0.005
            print(f"Total reward: {traj.reward}")
            traj.metrics["correct"] = total_reward
        else:
            print("No final answer received from agent")
            traj.reward = -1.0
            traj.metrics["correct"] = -1.0

    except Exception as e:
        print(f"Error running LangGraph agent: {e}")
        # Add error information to trajectory
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )
        traj.reward = -1.0
        traj.metrics["correct"] = -1.0

    return traj

