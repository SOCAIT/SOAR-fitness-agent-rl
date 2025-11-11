from typing import Dict, List

from src.env.verifiable_rewards.nutrition_rewards import nutrition_reward_weekly
from src.env.verifiable_rewards.schema_rewards import verify_nutrition_schema


def build_weekly_plan(daily_cal_target: int, daily_prot_target: int) -> Dict:
    days: List[Dict] = []
    for d in range(7):
        meals = [
            {"name": "Oatmeal with berries", "calories": 700, "proteins": 40, "carbs": 90, "fats": 15, "sequence": 1},
            {"name": "Grilled chicken and rice", "calories": 800, "proteins": 50, "carbs": 95, "fats": 18, "sequence": 2},
            {"name": "Tofu stir-fry", "calories": 800, "proteins": 46, "carbs": 80, "fats": 20, "sequence": 3},
        ]
        # Totals per day: 2300 kcal, 136 g protein
        days.append({"day": d, "meals": meals})
    return {"dailyMealPlans": days}


def build_weekly_plan_off_macros() -> Dict:
    """Intentionally off-target macros (e.g., too low calories/protein)."""
    days: List[Dict] = []
    for d in range(7):
        meals = [
            {"name": "Light salad", "calories": 400, "proteins": 20, "carbs": 50, "fats": 10, "sequence": 1},
            {"name": "Soup and bread", "calories": 500, "proteins": 25, "carbs": 60, "fats": 12, "sequence": 2},
            {"name": "Veggie bowl", "calories": 700, "proteins": 35, "carbs": 70, "fats": 18, "sequence": 3},
        ]
        # Totals per day: 1600 kcal, 80 g protein (below targets)
        days.append({"day": d, "meals": meals})
    return {"dailyMealPlans": days}


def build_weekly_plan_with_banned(daily_cal_target: int, daily_prot_target: int) -> Dict:
    """Valid macros but with a banned ingredient in meal names to trigger penalty."""
    days: List[Dict] = []
    for d in range(7):
        meals = [
            {"name": "Egg salad sandwich", "calories": 700, "proteins": 40, "carbs": 90, "fats": 15, "sequence": 1},
            {"name": "Grilled chicken and rice", "calories": 800, "proteins": 50, "carbs": 95, "fats": 18, "sequence": 2},
            {"name": "Tofu stir-fry", "calories": 800, "proteins": 46, "carbs": 80, "fats": 20, "sequence": 3},
        ]
        # Totals per day: 2300 kcal, 136 g protein (meets targets) but contains "egg"
        days.append({"day": d, "meals": meals})
    return {"dailyMealPlans": days}


def main() -> None:
    daily_cal_target = 2300
    daily_prot_target = 136
    
    # Case A: Perfect macros, no banned keywords → reward close to 1.0
    plan_ok = build_weekly_plan(daily_cal_target, daily_prot_target)
    s_ok, d_ok = verify_nutrition_schema(plan_ok)
    print("Case A - Schema score:", s_ok)
    if s_ok == 1.0:
        r_ok, g_ok = nutrition_reward_weekly(
            plan_ok,
            daily_cal_target=daily_cal_target,
            daily_prot_target=daily_prot_target,
            banned_keywords=[],
        )
        print("Case A - Weekly reward:", round(r_ok, 3), {"R_macro": g_ok.get("R_macro"), "R_schema": g_ok.get("R_schema"), "R_banned": g_ok.get("R_banned")})

    # Case B: Off-target macros (lower calories/protein) → reward < 1.0 due to R_macro
    plan_off = build_weekly_plan_off_macros()
    s_off, d_off = verify_nutrition_schema(plan_off)
    print("Case B - Schema score:", s_off)
    if s_off == 1.0:
        r_off, g_off = nutrition_reward_weekly(
            plan_off,
            daily_cal_target=daily_cal_target,
            daily_prot_target=daily_prot_target,
            banned_keywords=[],
        )
        print("Case B - Weekly reward:", round(r_off, 3), {"R_macro": g_off.get("R_macro"), "R_schema": g_off.get("R_schema"), "R_banned": g_off.get("R_banned")})

    # Case C: Perfect macros but with banned keyword "egg" → R_banned = 0 reduces reward
    plan_ban = build_weekly_plan_with_banned(daily_cal_target, daily_prot_target)
    s_ban, d_ban = verify_nutrition_schema(plan_ban)
    print("Case C - Schema score:", s_ban)
    if s_ban == 1.0:
        r_ban, g_ban = nutrition_reward_weekly(
            plan_ban,
            daily_cal_target=daily_cal_target,
            daily_prot_target=daily_prot_target,
            banned_keywords=["egg"],
        )
        print("Case C - Weekly reward:", round(r_ban, 3), {"R_macro": g_ban.get("R_macro"), "R_schema": g_ban.get("R_schema"), "R_banned": g_ban.get("R_banned")})


if __name__ == "__main__":
    main()
