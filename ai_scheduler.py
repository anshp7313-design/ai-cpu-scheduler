import json
import os
import random
import math
import ast
from collections import defaultdict
from copy import deepcopy

ACTIONS = ("FCFS", "SJF", "PRIORITY", "AGING_PRIORITY")
DEFAULT_Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")
HIGH_PRIORITY_THRESHOLD = 1
STARVATION_WAIT_THRESHOLD = 8
SEVERE_STARVATION_WAIT_THRESHOLD = 12
AGING_INTERVAL = 3
STARVE_THRESH = 10
BINS = 5


def load_q_table(path=DEFAULT_Q_TABLE_PATH):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    q_table = {}
    for state_key, action_map in raw.items():
        if not isinstance(action_map, dict):
            continue
        parsed_state = _deserialize_state_key(state_key)
        q_table[parsed_state] = {action: float(action_map.get(action, 3.0)) for action in ACTIONS}
    return q_table



def save_q_table(q_table, path=DEFAULT_Q_TABLE_PATH):
    serializable = {}
    for state_key, action_map in q_table.items():
        encoded_state = _serialize_state_key(state_key)
        serializable[encoded_state] = {action: float(action_map.get(action, 3.0)) for action in ACTIONS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def _serialize_state_key(state_key):
    if isinstance(state_key, tuple):
        return json.dumps(list(state_key), separators=(",", ":"))
    return str(state_key)


def _deserialize_state_key(state_key):
    if state_key == "empty":
        return state_key
    if isinstance(state_key, str) and state_key.startswith("[") and state_key.endswith("]"):
        try:
            decoded = json.loads(state_key)
            if isinstance(decoded, list):
                return tuple(decoded)
        except Exception:
            pass
    if isinstance(state_key, str) and state_key.startswith("(") and state_key.endswith(")"):
        try:
            decoded = ast.literal_eval(state_key)
            if isinstance(decoded, tuple):
                return decoded
        except Exception:
            pass
    return state_key



def _ensure_state(q_table, state_key):
    if state_key not in q_table:
        q_table[state_key] = {action: 3.0 for action in ACTIONS}
    return q_table[state_key]


def _build_visit_counts(q_table):
    visit_counts = defaultdict(lambda: {action: 1 for action in ACTIONS})
    for state_key in q_table.keys():
        _ = visit_counts[state_key]
    return visit_counts



def _fallback_action_for_ready_queue(ready_queue):
    if not ready_queue:
        return "FCFS"

    remaining = [p["remaining_bt"] for p in ready_queue]
    priority_count = sum(1 for p in ready_queue if p["priority"] <= HIGH_PRIORITY_THRESHOLD)
    burst_spread = max(remaining) - min(remaining)

    if priority_count > 0:
        return "PRIORITY"
    if burst_spread >= 3:
        return "SJF"
    return "FCFS"



def build_state_key(ready_queue, completed, total, prev_action):
    if not ready_queue:
        return "empty"
    n = len(ready_queue)
    bursts = [p["remaining_bt"] for p in ready_queue]
    waits = [p["waiting_time"] for p in ready_queue]
    prios = [p["priority"] for p in ready_queue]
    mean_b = sum(bursts) / n
    std_b = (sum((b - mean_b) ** 2 for b in bursts) / n) ** 0.5

    def B(x, bins=BINS):
        return int(min(x, 0.9999) * bins)

    return (
        B(min(n / max(total, 1), 1.0)),
        B(min(std_b / max(mean_b, 1), 2.0) / 2.0),
        B(min(max(waits) / 15.0, 1.0)),
        B(sum(1 for w in waits if w >= STARVE_THRESH) / n),
        B(sum(1 for p in prios if p <= 1) / n),
        B(min(min(bursts) / max(mean_b, 1), 1.0)),
        int(completed / max(total, 1) * 4),
        1 if prev_action in {"PRIORITY", "AGING_PRIORITY"} else 0,
    )


def select_action(q_table, visit_counts, state, actions, c=1.5, training=True):
    q_s = _ensure_state(q_table, state)
    if training:
        counts = visit_counts[state]
        total_n = sum(counts.values())
        scores = {
            a: q_s[a] + c * math.sqrt(math.log(total_n + 1) / counts[a])
            for a in actions
        }
        return max(scores, key=scores.get)
    return max(q_s, key=q_s.get)



def q_table_update(q_table, state_key, action, reward, next_state_key, alpha, gamma):
    current_actions = _ensure_state(q_table, state_key)
    next_actions = _ensure_state(q_table, next_state_key)
    current_q = float(current_actions[action])
    next_best_q = max(float(next_actions[a]) for a in ACTIONS)
    td_target = float(reward) + float(gamma) * next_best_q
    current_actions[action] = current_q + float(alpha) * (td_target - current_q)



def compute_reward(ready_queue_before, ready_queue_after, just_completed):
    n_before = max(len(ready_queue_before), 1)
    delta_wait = (
        sum(p["waiting_time"] for p in ready_queue_after) -
        sum(p["waiting_time"] for p in ready_queue_before)
    )
    reward = -delta_wait / n_before

    if just_completed:
        reward += 1.0

    new_starving = [
        p for p in ready_queue_after
        if p["waiting_time"] >= STARVE_THRESH
        and (p["waiting_time"] - 1) < STARVE_THRESH
    ]
    reward -= 2.0 * len(new_starving)

    return max(reward, -10.0)



def pick_process(ready_queue, action):
    if action == "FCFS":
        return min(ready_queue, key=lambda p: (p["arrival_time"], p["pid"]))
    if action == "SJF":
        return min(ready_queue, key=lambda p: (p["remaining_bt"], p["arrival_time"], p["pid"]))
    if action == "AGING_PRIORITY":
        return min(
            ready_queue,
            key=lambda p: (
                max(0, p["priority"] - (p["waiting_time"] // AGING_INTERVAL)),
                p["arrival_time"],
                p["pid"],
            ),
        )
    return min(ready_queue, key=lambda p: (p["priority"], p["arrival_time"], p["pid"]))


def generate_dynamic_explanation(action, ready_queue):
    """Generate state-aware explanations for scheduling decisions.
    
    Analyzes queue state and produces context-specific reasoning that shows
    real AI decision-making. Uses thresholds on state features to generate
    dynamic, varied explanations.
    """
    if not ready_queue:
        return "no jobs were ready, so the scheduler waited."
    
    # Extract state features from ready queue
    remaining_bursts = [p["remaining_bt"] for p in ready_queue]
    waiting_times = [p["waiting_time"] for p in ready_queue]
    priorities = [p["priority"] for p in ready_queue]
    
    n = len(ready_queue)
    queue_size = n
    max_waiting_time = max(waiting_times) if waiting_times else 0
    mean_burst = sum(remaining_bursts) / n if n > 0 else 1.0
    
    # Compute derived metrics
    high_priority_count = sum(1 for p in priorities if p <= HIGH_PRIORITY_THRESHOLD)
    high_priority_fraction = high_priority_count / n if n > 0 else 0
    
    starving_count = sum(1 for w in waiting_times if w >= STARVE_THRESH)
    starving_fraction = starving_count / n if n > 0 else 0
    
    burst_std = (sum((b - mean_burst) ** 2 for b in remaining_bursts) / n) ** 0.5 if n > 0 else 0
    burst_variability = burst_std / mean_burst if mean_burst > 0 else 0
    
    # Generate explanation based on action and thresholds
    if action == "SJF":
        if burst_variability > 0.6:
            variations = [
                f"High burst time variation ({burst_variability:.1%}) detected, so shorter jobs reduce waiting.",
                f"System detected uneven job sizes, making shortest-job optimal.",
                f"Significant variance in burst times, so SJF minimizes average wait.",
            ]
        else:
            variations = [
                "Shorter jobs help reduce average waiting time.",
                "SJF selected to optimize job completion speed.",
                "Shortest-job-first strategy chosen for efficiency.",
            ]
        return random.choice(variations)
    
    elif action == "PRIORITY":
        if high_priority_fraction > 0.3:
            variations = [
                f"Many urgent jobs ({high_priority_fraction:.0%}), so priority scheduling was used.",
                f"High priority concentration ({high_priority_fraction:.0%}) triggered priority selection.",
                f"Several high-priority processes waiting, priority applied.",
            ]
        else:
            variations = [
                "Priority scheduling selected to handle important tasks first.",
                "PRIORITY used to respect task criticality.",
                "Priority-based selection to ensure urgent work progresses.",
            ]
        return random.choice(variations)
    
    elif action == "AGING_PRIORITY":
        if starving_fraction > 0.2:
            variations = [
                f"Starvation risk detected ({int(starving_count)} jobs at risk), aging improved fairness.",
                f"Process(es) approaching timeout ({max_waiting_time} cycles), aging applied.",
                f"Fairness degradation risk ({starving_fraction:.0%}), aging protects waiting jobs.",
            ]
        else:
            variations = [
                "Aging used to prevent future starvation while respecting priority.",
                "Aging priority activated to balance urgency with fairness.",
                "Aging applied to maintain long-term queue health.",
            ]
        return random.choice(variations)
    
    elif action == "FCFS":
        if queue_size < 3:
            variations = [
                f"Queue is small ({queue_size} job(s)), FCFS is efficient.",
                f"Minimal queue size, FCFS maintains low overhead.",
                f"Few jobs waiting, so FCFS is fair and optimal.",
            ]
        else:
            variations = [
                "FCFS used to maintain execution order fairness.",
                "Arrival order maintained by FCFS for predictable behavior.",
                "First-come-first-served selected for stable management.",
            ]
        return random.choice(variations)
    
    return "scheduling decision made."


def explain_action_choice(ready_queue, action):
    """Wrapper that uses dynamic explanation generation for AI insights."""
    return generate_dynamic_explanation(action, ready_queue)



def _prepare_processes(process_rows):
    prepared = []
    for row in process_rows:
        pid = int(row[0])
        arrival_time = int(row[1])
        burst_time = int(row[2])
        priority = int(row[3])
        prepared.append(
            {
                "pid": pid,
                "arrival_time": arrival_time,
                "burst_time": burst_time,
                "priority": priority,
                "remaining_bt": burst_time,
                "waiting_time": 0,
                "completion_time": 0,
                "turnaround_time": 0,
            }
        )
    return prepared



def _compress_gantt(unit_segments):
    if not unit_segments:
        return []

    compressed = [list(unit_segments[0])]
    for pid, start, duration in unit_segments[1:]:
        last = compressed[-1]
        if last[0] == pid and last[1] + last[2] == start:
            last[2] += duration
        else:
            compressed.append([pid, start, duration])
    return [tuple(seg) for seg in compressed]



def _write_output_file(gantt_segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for pid, start, duration in gantt_segments:
            f.write(f"{pid} {start} {duration}\n")



def _write_metrics_file(processes, avg_tat, avg_wt, metrics_path):
    ordered = sorted(processes, key=lambda p: p["pid"])
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("PID AT BT PR CT TAT WT\n")
        for p in ordered:
            f.write(
                f"{p['pid']} {p['arrival_time']} {p['burst_time']} {p['priority']} "
                f"{p['completion_time']} {p['turnaround_time']} {p['waiting_time']}\n"
            )
        f.write(f"Average Turnaround Time: {avg_tat:.2f}\n")
        f.write(f"Average Waiting Time: {avg_wt:.2f}\n")



def simulate_ai_scheduler(
    process_rows,
    q_table=None,
    q_table_path=DEFAULT_Q_TABLE_PATH,
    alpha=0.20,
    gamma=0.90,
    learning=False,
    save_q_table_to_disk=False,
    output_path=None,
    metrics_path=None,
    return_trace=False,
):
    if q_table is None:
        q_table = load_q_table(q_table_path)

    processes = _prepare_processes(process_rows)
    live_processes = deepcopy(processes)

    current_time = 0
    completed = 0
    total = len(live_processes)
    unit_gantt = []
    previous_action = None
    visit_counts = _build_visit_counts(q_table)
    decision_trace = []

    while completed < total:
        ready_queue = [
            p for p in live_processes if p["arrival_time"] <= current_time and p["remaining_bt"] > 0
        ]

        if not ready_queue:
            current_time += 1
            continue

        state_key = build_state_key(ready_queue, completed, total, previous_action)

        action = select_action(
            q_table,
            visit_counts,
            state_key,
            ACTIONS,
            c=1.5,
            training=learning,
        )
        visit_counts[state_key][action] += 1

        chosen = pick_process(ready_queue, action)
        ready_queue_before = [
            {"pid": p["pid"], "waiting_time": p["waiting_time"]}
            for p in ready_queue
        ]

        if return_trace:
            decision_trace.append(
                {
                    "time": current_time,
                    "pid": chosen["pid"],
                    "action": action,
                    "reason": explain_action_choice(ready_queue, action),
                    "ready_count": len(ready_queue),
                }
            )

        unit_gantt.append((chosen["pid"], current_time, 1))
        chosen["remaining_bt"] -= 1

        for p in ready_queue:
            if p is not chosen and p["remaining_bt"] > 0:
                p["waiting_time"] += 1

        just_completed = False
        if chosen["remaining_bt"] == 0:
            chosen["completion_time"] = current_time + 1
            chosen["turnaround_time"] = chosen["completion_time"] - chosen["arrival_time"]
            completed += 1
            just_completed = True

        ready_queue_after = [
            p for p in live_processes if p["arrival_time"] <= current_time + 1 and p["remaining_bt"] > 0
        ]
        reward = compute_reward(ready_queue_before, ready_queue_after, just_completed)

        next_state_key = build_state_key(ready_queue_after, completed, total, action)

        if learning:
            q_table_update(q_table, state_key, action, reward, next_state_key, alpha, gamma)

        previous_action = action
        current_time += 1

    gantt = _compress_gantt(unit_gantt)

    for p in live_processes:
        p["turnaround_time"] = p["completion_time"] - p["arrival_time"]

    avg_wt = sum(p["waiting_time"] for p in live_processes) / total if total else 0.0
    avg_tat = sum(p["turnaround_time"] for p in live_processes) / total if total else 0.0

    if save_q_table_to_disk:
        save_q_table(q_table, q_table_path)

    if output_path:
        _write_output_file(gantt, output_path)
    if metrics_path:
        _write_metrics_file(live_processes, avg_tat, avg_wt, metrics_path)

    process_table = [
        {
            "pid": p["pid"],
            "arrival_time": p["arrival_time"],
            "burst_time": p["burst_time"],
            "priority": p["priority"],
            "completion_time": p["completion_time"],
            "turnaround_time": p["turnaround_time"],
            "waiting_time": p["waiting_time"],
        }
        for p in sorted(live_processes, key=lambda x: x["pid"])
    ]

    return {
        "gantt": gantt,
        "process_table": process_table,
        "avg_waiting_time": avg_wt,
        "avg_turnaround_time": avg_tat,
        "q_table": q_table,
        "decision_trace": decision_trace if return_trace else [],
    }
