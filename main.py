from src.env.verifiers_utils import verify_nutrition_plan, verify_workout_plan, verify_nutrition_schema
from src.data_utils.create_synthetic_data import data
from src.env.verifiable_rewards.workout_rewards import verify_workout_week
from src.env.verifiable_rewards.nutrition_rewards import verify_nutrition_plan, verify_macros, verify_no_banned, verify_daily_meal_plan_macros
from src.env.verifiable_rewards.schema_rewards import verify_nutrition_schema, verify_meal_plan_schema, verify_workout_schema, is_valid_json
from src.env.verifiable_rewards.nutrition_rewards import nutrition_reward
# Example nutrition plan output from agent
plan_json = {
    "dailyMealPlans": data[0]["target_output"]["dailyMealPlans"]
}


# # Example workout weekly plan
# wk = {
#     "workouts": [ … ]  # your 7 day plan
# }
# wscore, winfo = verify_workout_plan(wk)
# print("Workout score:", wscore, "info:", winfo)

def main():
    print("Hello from fitness-reasoning-rl-agent!")

    score, info = verify_nutrition_plan(
    plan_json,
    daily_cal_target=2300,
    daily_prot_target=136,
    banned_keywords=["egg", "shellfish"]
    )
    print("Nutrition score:", score, "info:", info)

    score, info = verify_nutrition_schema(plan_json)
    print("Nutrition schema score:", score, "info:", info)

    daily_plan = plan_json["dailyMealPlans"][1]

    daily_plan = {"meals": [{"name": "Grilled Tofu and Quinoa with Sexy Veggie Veggies", "calories": 700, "proteins": 45, "carbs": 60, "fats": 20, "sequence": 1}, {"name": "Lentil Stew with Carrot Climbers", "calories": 680, "proteins": 25, "carbs": 80, "fats": 8, "sequence": 2}, {"name": "Chickpea Curry with Potato Prancers", "calories": 650, "proteins": 20, "carbs": 60, "fats": 15, "sequence": 3}, {"name": "Avocado Salad with Summer Greens", "calories": 350, "proteins": 10, "carbs": 40, "fats": 20, "sequence": 4}, {"name": "Peanut Butter Smoothie with Banana and Spinach", "calories": 500, "proteins": 20, "carbs": 30, "fats": 25, "sequence": 5}, {"name": "Oatmeal with Nuts, Raisins and Milk", "calories": 400, "proteins": 15, "carbs": 45, "fats": 8, "sequence": 6}, {"name": "Roasted Eggplant Dip with Garlic Prancers", "calories": 350, "proteins": 5, "carbs": 10, "fats": 25, "sequence": 7}]}

    score, info = verify_meal_plan_schema(daily_plan)
    print("Daily meal plan schema score:", score, "info:", info)

    score, info = verify_daily_meal_plan_macros(daily_plan, daily_cal_target=600, daily_prot_target=136)
    print("Daily meal plan macros score:", score, "info:", info)

    score, info = nutrition_reward(daily_plan, daily_cal_target=600, daily_prot_target=136, banned_keywords=None)
    print("Nutrition reward score:", score, "info:", info)



if __name__ == "__main__":
    main()
