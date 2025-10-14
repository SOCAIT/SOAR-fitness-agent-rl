
def verify_macros(plan_json, daily_cal_target, daily_prot_target, tol_pct=0.05):
    """
    Checks each day’s summed calories and proteins are within ± tol_pct.
    Returns list of error messages (empty = pass).
    """
    errors = []
    for day in plan_json.get("dailyMealPlans", []):
        d = day["day"]
        meals = day.get("meals", [])
        tot_c = sum(m.get("calories", 0) for m in meals)
        tot_p = sum(m.get("proteins", 0) for m in meals)
        low_c = daily_cal_target * (1 - tol_pct)
        high_c = daily_cal_target * (1 + tol_pct)
        if not (low_c <= tot_c <= high_c):
            errors.append(f"Day {d} calories {tot_c:.1f} outside [{low_c:.1f}, {high_c:.1f}]")
        low_p = daily_prot_target * (1 - tol_pct)
        high_p = daily_prot_target * (1 + tol_pct)
        if not (low_p <= tot_p <= high_p):
            errors.append(f"Day {d} protein {tot_p:.1f} outside [{low_p:.1f}, {high_p:.1f}]")
    return errors

def verify_daily_meal_plan_macros(plan_json, daily_cal_target, daily_prot_target, tol_pct=0.05):
    """
    Checks each day’s summed calories and proteins are within ± tol_pct.
    Returns list of error messages (empty = pass).
    """
    errors = []
    daily_meal_plan = plan_json.get("meals", [])
    tot_c = 0
    tot_p = 0
    low_c = daily_cal_target * (1 - tol_pct)
    high_c = daily_cal_target * (1 + tol_pct)
    low_p = daily_prot_target * (1 - tol_pct)
    high_p = daily_prot_target * (1 + tol_pct)
    for meal in daily_meal_plan:
        c = meal.get("calories", 0)
        p = meal.get("proteins", 0)
        
        tot_c += c
        tot_p += p


    if not (low_c <= tot_c <= high_c):
            errors.append(f"Calories {tot_c:.1f} outside [{low_c:.1f}, {high_c:.1f}]")
    if not (low_p <= tot_p <= high_p):
        errors.append(f"Protein {tot_p:.1f} outside [{low_p:.1f}, {high_p:.1f}]")
    

    print(tot_c, tot_p)
    if errors:
        return -1.0, {"macro_errors": errors}
    return 1.0, {"ok": True}

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

def verify_workout_week(week_plan):
    """
    week_plan: list of day dicts (from plan_json["workouts"])
    Returns list of error messages.
    """
    errors = []
    seen_days = set()
    rest_days = 0
    for entry in week_plan:
        d = entry.get("day")
        if d is None or not (0 <= d <= 6):
            errors.append(f"Invalid day index {d}")
        else:
            if d in seen_days:
                errors.append(f"Duplicate day {d}")
            seen_days.add(d)
        exs = entry.get("exercises", [])
        if not exs:
            rest_days += 1
        else:
            for e in exs:
                if ("sets" not in e) or ("reps" not in e) or ("restTime" not in e):
                    errors.append(f"Day {d} exercise missing key: {e.get('exercise')}")
    # Check coverage: you expect one entry per day 0..6 or at least unique set
    if len(seen_days) != len(week_plan):
        errors.append("Mismatch: workout list length vs unique days")
    if rest_days < 1:
        errors.append("Less than 1 rest day")
    return errors



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

