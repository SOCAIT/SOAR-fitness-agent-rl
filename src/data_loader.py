"""Data loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import List

from datasets import load_dataset

from src.models import Scenario


def load_fitness_scenarios(data_file: str | Path) -> List[Scenario]:
    """
    Load fitness scenarios from a JSONL file.

    Args:
        data_file: Path to the JSONL file containing scenarios

    Returns:
        List of Scenario objects
    """
    # Load the dataset from the JSONL file
    dataset = load_dataset("json", data_files=str(data_file))

    # Assuming the dataset has a "train" split
    training_scenarios = dataset["train"]

    print(f"Dataset loaded successfully! {len(training_scenarios)} scenarios found.")

    # Apply transformations
    training_scenarios = training_scenarios.map(add_one_day_meal_question)
    training_scenarios = training_scenarios.map(combine_question_and_context)
    training_scenarios = training_scenarios.map(convert_val_to_test)
    training_scenarios = training_scenarios.map(extract_target_nutrition_data)

    # Convert to Scenario objects
    scenarios_list = []
    for example in training_scenarios:
        scenario = Scenario(
            question=example["question"],
            split=example["split"],
            id=str(example["id"]),
            daily_cal_target=example["daily_cal_target"],
            daily_prot_target=example["daily_prot_target"],
            daily_carb_target=example["daily_carb_target"],
            daily_fat_target=example["daily_fat_target"],
        )
        scenarios_list.append(scenario)

    print(f"Created {len(scenarios_list)} Scenario objects.")

    # Print split distribution
    from collections import Counter

    splits = [s.split for s in scenarios_list]
    print(f"Split distribution: {Counter(splits)}")

    return scenarios_list


def add_one_day_meal_question(example: dict) -> dict:
    """Add the one day meal plan question to the example."""
    one_day_prompt = (
        "generate a one day meal plan for user that match its macros and diet"
    )
    example["input_question"] = f"{one_day_prompt}"
    return example


def combine_question_and_context(example: dict) -> dict:
    """Combine the question with context information."""
    context_str = json.dumps(example["context"])
    example["question"] = f"{example['input_question']} Context: {context_str}"
    return example


def convert_val_to_test(example: dict) -> dict:
    """Convert 'val' split to 'test' split."""
    if example["split"] == "val":
        example["split"] = "test"
    return example


def extract_target_nutrition_data(example: dict) -> dict:
    """Extract target nutrition data from context."""
    daily_cal_target = example["context"]["daily_cal_target"]
    daily_prot_target = example["context"]["daily_prot_target"]
    daily_carb_target = example["context"]["daily_carb_target"]
    daily_fat_target = example["context"]["daily_fat_target"]

    example["daily_cal_target"] = daily_cal_target
    example["daily_prot_target"] = daily_prot_target
    example["daily_carb_target"] = daily_carb_target
    example["daily_fat_target"] = daily_fat_target
    return example

