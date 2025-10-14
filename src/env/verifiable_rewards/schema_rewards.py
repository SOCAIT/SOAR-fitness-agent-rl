import jsonschema
import json
import re

from src.constants.schema import nutrition_schema, workout_one_week_schema, daily_meal_plan_schema

from jsonschema import validate, ValidationError

def is_valid_json(data):
    """
    Check if data is valid JSON (either already parsed dict or parseable string).
    Returns (bool, parsed_data_or_error_msg)
    """
    if isinstance(data, dict):
        return float(True), data
    
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return float(True), parsed
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {e.msg} at position {e.pos}"
    
    return float(False), f"Invalid input type: {type(data).__name__}"

def verify_nutrition_schema(plan_json):
    """
    Returns (bool, message) whether the JSON matches the schema.
    """
    
    
    
    try:
        validate(instance=plan_json, schema=nutrition_schema)
        verification = True
        return float(verification), "schema OK"
    except ValidationError as e:
        return -1.0, f"schema error: {e.message}"

def verify_meal_plan_schema(plan_json):
    """
    Returns (bool, message) whether the JSON matches the schema.
    """
    
    
    try:
        validate(instance=plan_json, schema=daily_meal_plan_schema)
        verification = True
        return float(verification), "schema OK"
    except ValidationError as e:
        return -1.0, f"schema error: {e.message}"


def verify_workout_schema(plan_json):
    """Check that the workout plan JSON matches 1-week workout schema."""
    
    
    try:
        validate(instance=plan_json, schema=workout_one_week_schema)
        verification = True
        return float(verification), "schema OK"
    except ValidationError as e:
        return -1.0, f"schema error: {e.message}"
