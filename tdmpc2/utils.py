import math

def get_distillation_coefficient(step: int, schedule: str = "four_phase", total_steps: int = 1_000_000, base_coef: float = 0.4) -> float:
    # Experiment with different distillation schedules
    if schedule == "four_phase":
        if step < total_steps/4:
            return 0.5
        elif step <= total_steps/2:
            return 0.5
        elif step <= 3 * total_steps / 4:
            return 0.75
        else:
            return 0.0
    elif schedule == "decrease":
        return max(0.0, base_coef * (1 - step / total_steps))
    elif schedule == "increase":
        return min(base_coef, base_coef * (step / total_steps))
    elif schedule == "linear_decay":
        return base_coef * max(1 - step / total_steps, 0.1)
    elif schedule == "cosine_decay":
        return 0.5 * base_coef * (1 + math.cos(math.pi * step / total_steps))
    elif schedule == "constant":
        return base_coef
    else:
        raise ValueError(f"Unknown distillation schedule: {schedule}")
