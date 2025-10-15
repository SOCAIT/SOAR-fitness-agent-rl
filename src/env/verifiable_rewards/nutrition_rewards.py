import math

from src.env.verifiable_rewards.schema_rewards import verify_meal_plan_schema, verify_workout_schema, verify_nutrition_schema

def _banded_score(err, tol=0.03, hard=0.10):
    """
    err: relative error (abs(actual-target)/target).
    1.0 inside tol; cosine-decay to 0 by 'hard'.
    """
    if err <= tol: return 1.0
    if err >= hard: return 0.0
    x = (err - tol) / (hard - tol)
    return 0.5 * (1 + math.cos(math.pi * x))

def verify_daily_meal_plan_macros(plan_json, daily_cal_target, daily_prot_target,
                                  tol_cal=0.03, hard_cal=0.10,
                                  tol_pro=0.02, hard_pro=0.08):
    """
    Returns (score in [0,1], diagnostics).
    Score is a weighted combo emphasizing protein.
    """
    meals = plan_json.get("meals", [])
    tot_c = sum(m.get("calories", 0) for m in meals)
    tot_p = sum(m.get("proteins", 0) for m in meals)

    rel_c = abs(tot_c - daily_cal_target) / max(daily_cal_target, 1)
    rel_p = abs(tot_p - daily_prot_target) / max(daily_prot_target, 1)

    s_cal = _banded_score(rel_c, tol=tol_cal, hard=hard_cal)
    s_pro = _banded_score(rel_p, tol=tol_pro, hard=hard_pro)

    # Emphasize protein, then calories
    score = 0.6 * s_pro + 0.4 * s_cal

    diag = {
        "totals": {"calories": tot_c, "proteins": tot_p},
        "targets": {"calories": daily_cal_target, "proteins": daily_prot_target},
        "rel_errors": {"cal": rel_c, "pro": rel_p},
        "component_scores": {"cal": s_cal, "pro": s_pro},
        "within_5pct": {"cal": rel_c <= 0.05, "pro": rel_p <= 0.05},
    }
    return score, diag
def verify_macros(plan_json, daily_cal_target, daily_prot_target,
                  tol_cal=0.03, hard_cal=0.10, tol_pro=0.02, hard_pro=0.08):
    """
    plan_json["dailyMealPlans"] is a list of days.
    Returns (avg_score, per_day_diags).
    """
    days = plan_json.get("dailyMealPlans", [])
    if not isinstance(days, list) or not days:
        return 0.0, {"error": "dailyMealPlans missing/empty"}

    per_day = []
    for day in days:
        meals = day.get("meals", [])
        tot_c = sum(m.get("calories", 0) for m in meals)
        tot_p = sum(m.get("proteins", 0) for m in meals)
        rel_c = abs(tot_c - daily_cal_target) / max(daily_cal_target, 1)
        rel_p = abs(tot_p - daily_prot_target) / max(daily_prot_target, 1)
        s_cal = _banded_score(rel_c, tol=tol_cal, hard=hard_cal)
        s_pro = _banded_score(rel_p, tol=tol_pro, hard=hard_pro)
        score = 0.6*s_pro + 0.4*s_cal
        per_day.append({"day": day.get("day"), "score": score, "tot_c": tot_c, "tot_p": tot_p, "rel_c": rel_c, "rel_p": rel_p})
    avg = sum(d["score"] for d in per_day) / len(per_day)
    return avg, {"per_day": per_day}


def verify_no_banned(plan_json, banned_keywords):
    """
    Ensure no banned keywords appear in meal name or description.
    Returns list of (day, meal_name, offending_keyword) if violations.
    """
    violations = []
    for day in plan_json.get("dailyMealPlans", []):
        d = day.get("day")
        for m in day.get("meals", []):
            text = (m.get("name", "") + " " + m.get("description", "")).lower()
            for kw in banned_keywords:
                if re.search(rf"\b{re.escape(kw.lower())}\b", text):
                    violations.append((d, m.get("name", ""), kw))
    return violations


# --- Combined verifier APIs ---

def verify_nutrition_plan(plan_json, daily_cal_target, daily_prot_target, banned_keywords=None):
    """
    Returns (score, diagnostic_info)
    score is in [0,1]. 1 means fully valid.
    """
    ok, msg = verify_nutrition_schema(plan_json)
    if not ok:
        return -1.0, {"schema_error": msg}

    macro_errs = verify_macros(plan_json, daily_cal_target, daily_prot_target)
    if macro_errs:
        return -1.0, {"macro_errors": macro_errs}

    if banned_keywords:
        banned_viol = verify_no_banned(plan_json, banned_keywords)
        if banned_viol:
            return -1.0, {"banned_violations": banned_viol}

    return 1.0, {"ok": True}

def verify_workout_plan(plan_json):
    """
    plan_json: dict with key "workouts"
    Returns (score, diagnostics)
    """
    ok, msg = verify_workout_schema(plan_json)
    if not ok:
        return -1.0, {"schema_error": msg}

    week = plan_json.get("workouts", [])
    errs = verify_workout_week(week)
    if errs:
        return -1.0, {"week_errors": errs}

    return 1.0, {"ok": True}



def nutrition_reward(payload, daily_cal_target, daily_prot_target, banned_keywords, traj=None):
    R_schema, diag_schema = verify_meal_plan_schema(payload)
    R_macro,  diag_macro  = verify_daily_meal_plan_macros(payload, daily_cal_target, daily_prot_target)
    if banned_keywords:
        R_banned, diag_banned = verify_no_banned(payload, banned_keywords)
    else:
        R_banned = 1.0
        diag_banned = {"ok": True}

    # Optional: finalization factor if you log tool order
    F_final = 1.0
    # if traj is not None:
    #     try:
    #         last = None
    #         for m in traj.messages_and_choices:
    #             if m.get("role") == "tool_log":
    #                 blob = json.loads(m["content"]); end = blob.get("end")
    #                 if end: last = end.get("tool")
    #         F_final = 1.0 if last == "return_final_answer_tool" else 0.4 if last else 0.1
    #     except Exception:
    #         F_final = 0.7

    # Heavier weight on macros
    if banned_keywords:
        reward_weights = {  "R_macro": 0.70, "R_schema": 0.20,  "R_banned": 0.10, }
    else:
        reward_weights = { "R_macro": 0.70, "R_schema": 0.30, "R_banned": 0.00  }

    base = reward_weights["R_macro"]*R_macro + reward_weights["R_schema"]*R_schema + reward_weights["R_banned"]*R_banned
    total = max(0.0, min(1.05, base * F_final))
    diag = {"R_macro": R_macro, "R_schema": R_schema, "R_banned": R_banned, "F_final": F_final,
            "diag_macro": diag_macro, "diag_schema": diag_schema, "diag_banned": diag_banned}
    return total, diag

