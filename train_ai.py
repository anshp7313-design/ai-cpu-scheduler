import argparse
import json
import os
import random

from ai_scheduler import (
    DEFAULT_Q_TABLE_PATH,
    load_q_table,
    save_q_table,
    simulate_ai_scheduler,
)


DEFAULT_META_PATH = "q_table_meta.json"
TRAINING_PROFILES = (
    "all-arrival-same",
    "staggered-arrivals",
    "short-job-heavy",
    "long-job-heavy",
    "mixed-priority",
    "starvation-prone",
    "extreme-burst-outlier",
    "late-heavy-arrivals",
    "priority-inversion",
)

VERSATILE_TRAINING_PROFILE_WEIGHTS = (
    ("mixed-priority", 4),
    ("priority-inversion", 3),
    ("starvation-prone", 3),
    ("staggered-arrivals", 2),
    ("short-job-heavy", 2),
    ("long-job-heavy", 2),
    ("late-heavy-arrivals", 2),
    ("all-arrival-same", 1),
    ("extreme-burst-outlier", 1),
)


def load_training_metadata(meta_path=DEFAULT_META_PATH):
    if not os.path.exists(meta_path):
        return {"episodes_trained": 0, "last_run_episodes": 0}

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "episodes_trained": int(data.get("episodes_trained", 0)),
        "last_run_episodes": int(data.get("last_run_episodes", 0)),
    }


def save_training_metadata(metadata, meta_path=DEFAULT_META_PATH):
    payload = {
        "episodes_trained": int(metadata.get("episodes_trained", 0)),
        "last_run_episodes": int(metadata.get("last_run_episodes", 0)),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def random_process_rows(num_processes=None):
    return random_process_rows_profiled(num_processes=num_processes, profile="mixed-priority")


def _pick_burst(profile):
    if profile == "short-job-heavy":
        return random.choices([1, 2, 3, 4, 5, 6], weights=[8, 8, 7, 5, 2, 1], k=1)[0]
    if profile == "long-job-heavy":
        return random.choices([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20], weights=[1, 1, 2, 3, 4, 5, 5, 5, 4, 3, 3, 2, 2, 1], k=1)[0]
    if profile == "extreme-burst-outlier":
        return random.choices([1, 2, 3, 4, 5, 8, 12, 20, 30, 40, 50, 60], weights=[6, 6, 5, 5, 4, 3, 3, 2, 2, 2, 1, 1], k=1)[0]
    if profile == "starvation-prone":
        return random.randint(4, 12)
    return random.randint(1, 10)


def random_process_rows_profiled(num_processes=None, profile="mixed"):
    if num_processes is None:
        if profile == "all-arrival-same":
            num_processes = random.randint(4, 7)
        elif profile == "staggered-arrivals":
            num_processes = random.randint(4, 8)
        elif profile == "short-job-heavy":
            num_processes = random.randint(4, 7)
        elif profile == "long-job-heavy":
            num_processes = random.randint(4, 7)
        elif profile == "mixed-priority":
            num_processes = random.randint(4, 8)
        elif profile == "starvation-prone":
            num_processes = random.randint(5, 8)
        elif profile == "extreme-burst-outlier":
            num_processes = random.randint(5, 8)
        elif profile == "late-heavy-arrivals":
            num_processes = random.randint(4, 8)
        elif profile == "priority-inversion":
            num_processes = random.randint(5, 8)
        else:
            num_processes = random.randint(4, 8)

    rows = []
    for pid in range(1, num_processes + 1):
        if profile == "all-arrival-same":
            arrival_time = 0
            burst_time = random.randint(1, 10)
            priority = random.randint(0, 5)
        elif profile == "staggered-arrivals":
            arrival_time = pid - 1 + random.randint(0, 1)
            burst_time = random.randint(2, 9)
            priority = random.randint(0, 5)
        elif profile == "short-job-heavy":
            arrival_time = random.randint(0, 3)
            burst_time = _pick_burst(profile)
            priority = random.randint(0, 5)
        elif profile == "long-job-heavy":
            arrival_time = random.randint(0, 3)
            burst_time = _pick_burst(profile)
            priority = random.randint(0, 5)
        elif profile == "mixed-priority":
            arrival_time = random.randint(0, 5)
            burst_time = random.randint(1, 10)
            priority = random.choices([0, 1, 2, 3, 4, 5], weights=[3, 3, 2, 2, 1, 1], k=1)[0]
        elif profile == "starvation-prone":
            arrival_time = random.choice([0, 0, 0, 1, 1, 2, 3])
            burst_time = _pick_burst(profile)
            priority = random.randint(2, 5)
        elif profile == "extreme-burst-outlier":
            arrival_time = random.randint(0, 4)
            burst_time = _pick_burst(profile)
            priority = random.randint(0, 5)
        elif profile == "late-heavy-arrivals":
            arrival_time = random.choice([2, 3, 4, 5, 6, 7])
            burst_time = random.randint(1, 12)
            priority = random.randint(0, 5)
        elif profile == "priority-inversion":
            arrival_time = random.choice([0, 0, 1, 1, 2, 3])
            burst_time = random.randint(2, 15)
            priority = random.randint(0, 5)
        else:
            arrival_time = random.randint(0, 5)
            burst_time = _pick_burst(profile)
            priority = random.randint(0, 5)

        rows.append((str(pid), str(arrival_time), str(burst_time), str(priority)))

    if profile == "mixed-priority" and rows:
        focus_index = random.randrange(len(rows))
        pid, arrival_time, burst_time, _ = rows[focus_index]
        rows[focus_index] = (pid, arrival_time, burst_time, str(random.randint(0, 1)))

    if profile == "starvation-prone" and len(rows) >= 3:
        rows[0] = (rows[0][0], "0", str(max(9, int(rows[0][2]))), "5")
        rows[1] = (rows[1][0], "0", str(max(8, int(rows[1][2]))), "4")
        rows[2] = (rows[2][0], "0", str(max(7, int(rows[2][2]))), "3")
        tail = rows[3:]
        for i, row in enumerate(tail, start=3):
            pid, _, bt, _ = row
            rows[i] = (pid, str(random.randint(1, 3)), str(min(int(bt), 4)), str(random.randint(0, 2)))

    if profile == "extreme-burst-outlier" and len(rows) >= 4:
        outlier_idx = random.randrange(len(rows))
        pid, at, _, pr = rows[outlier_idx]
        rows[outlier_idx] = (pid, at, str(random.choice([30, 40, 50, 60])), pr)

    if profile == "late-heavy-arrivals" and len(rows) >= 3:
        # Keep at least one early anchor process so queue dynamics are not trivial idle-only runs.
        pid, _, bt, pr = rows[0]
        rows[0] = (pid, "0", bt, pr)

    if profile == "priority-inversion" and len(rows) >= 3:
        # A long urgent process arriving early plus short later jobs with lower urgency.
        rows[0] = (rows[0][0], "0", str(max(18, int(rows[0][2]))), "0")
        rows[1] = (rows[1][0], "1", str(min(6, int(rows[1][2]))), "4")
        rows[2] = (rows[2][0], "2", str(min(5, int(rows[2][2]))), "5")

    if profile in ("short-job-heavy", "long-job-heavy") and len(rows) >= 2:
        rows[0] = (rows[0][0], "0", rows[0][2], rows[0][3])
        rows[1] = (rows[1][0], "0", rows[1][2], rows[1][3])

    return rows


def _build_versatile_profile_schedule(total_episodes):
    if total_episodes <= 0:
        return []

    weighted_cycle = []
    for profile, weight in VERSATILE_TRAINING_PROFILE_WEIGHTS:
        weighted_cycle.extend([profile] * max(1, int(weight)))

    schedule = []
    full_cycles = total_episodes // len(weighted_cycle)
    remainder = total_episodes % len(weighted_cycle)

    for _ in range(full_cycles):
        cycle = list(weighted_cycle)
        random.shuffle(cycle)
        schedule.extend(cycle)

    if remainder:
        tail = list(weighted_cycle)
        random.shuffle(tail)
        schedule.extend(tail[:remainder])

    return schedule


def get_workload_for_episode(ep):
    if ep < 1000:
        return random.choices(
            ["all-arrival-same", "short-job-heavy", "long-job-heavy"],
            weights=[0.4, 0.3, 0.3]
        )[0]
    elif ep < 3000:
        return random.choices(
            ["mixed-priority", "staggered-arrivals", "priority-inversion", "starvation-prone"],
            weights=[0.3, 0.3, 0.2, 0.2]
        )[0]
    else:
        return random.choices(
            ["starvation-prone", "priority-inversion", "extreme-burst-outlier", "late-heavy-arrivals"],
            weights=[0.3, 0.3, 0.2, 0.2]
        )[0]


def describe_training_mix():
    return ", ".join(f"{profile}:{weight}" for profile, weight in VERSATILE_TRAINING_PROFILE_WEIGHTS)


def train_model(
    episodes=5000,
    alpha=0.20,
    gamma=0.90,
    q_table_path=DEFAULT_Q_TABLE_PATH,
    meta_path=DEFAULT_META_PATH,
):
    q_table = load_q_table(q_table_path)

    metadata = load_training_metadata(meta_path)

    for episode in range(episodes):
        profile = get_workload_for_episode(episode)
        process_rows = random_process_rows_profiled(profile=profile)
        simulate_ai_scheduler(
            process_rows,
            q_table=q_table,
            q_table_path=q_table_path,
            alpha=alpha,
            gamma=gamma,
            learning=True,
            save_q_table_to_disk=False,
        )

    save_q_table(q_table, q_table_path)

    metadata["episodes_trained"] = metadata.get("episodes_trained", 0) + episodes
    metadata["last_run_episodes"] = episodes
    save_training_metadata(metadata, meta_path)
    return q_table


def train(
    episodes=5000,
    alpha=0.20,
    gamma=0.90,
    q_table_path=DEFAULT_Q_TABLE_PATH,
    meta_path=DEFAULT_META_PATH,
):
    return train_model(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        q_table_path=q_table_path,
        meta_path=meta_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train AI model for CPU Scheduler")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of random episodes")
    parser.add_argument("--alpha", type=float, default=0.20, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor")
    parser.add_argument("--q-table", type=str, default=DEFAULT_Q_TABLE_PATH, help="Path to q-table JSON file")
    parser.add_argument("--meta-file", type=str, default=DEFAULT_META_PATH, help="Path to training metadata JSON file")
    args = parser.parse_args()

    train_model(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        q_table_path=args.q_table,
        meta_path=args.meta_file,
    )

    print(f"Training complete. Saved Q-table model to {args.q_table}")


if __name__ == "__main__":
    main()
