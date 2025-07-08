def get_distillation_coefficient(step: int, schedule: str = "decrease", total_steps: int = 1_000_000, base_coef: float = 0.5) -> float:
    if schedule == "four_phase":
        if step < 250_000:
            return 0.5
        elif step < 500_000:
            return 0.5
        elif step < 750_000:
            return 0.75
        else:
            return 0.0
    elif schedule == "decrease":
        return max(0.0, base_coef * (1 - step / total_steps))
    elif schedule == "increase":
        return min(base_coef, base_coef * (step / total_steps))
    elif schedule == "constant":
        return base_coef
    else:
        raise ValueError(f"Unknown distillation schedule: {schedule}")
