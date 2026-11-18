import tkinter as tk
from tkinter import ttk
import subprocess
import os
import shutil
import threading
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from ai_scheduler import simulate_ai_scheduler
from train_ai import load_training_metadata, train_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Q_TABLE_PATH = os.path.join(BASE_DIR, "q_table.json")
MODEL_META_PATH = os.path.join(BASE_DIR, "q_table_meta.json")

ai_training_in_progress = False
ai_model_ready = os.path.exists(Q_TABLE_PATH)
last_loaded_sample_rows = None


def get_model_status_text():
    if not os.path.exists(Q_TABLE_PATH):
        return "Model Status: Not Trained | Train Q-table model to enable stronger AI inference"

    metadata = load_training_metadata(MODEL_META_PATH)
    episodes_trained = int(metadata.get("episodes_trained", 0))

    if episodes_trained >= 5000:
        return f"Model Status: Q-table READY | approx. {episodes_trained}+ cumulative episodes"
    if episodes_trained > 0:
        return f"Model Status: Q-table READY | {episodes_trained} cumulative episodes"
    return "Model Status: Q-table READY"


def update_model_status_ui():
    if "model_status_val" not in globals():
        return
    model_status_val.config(text=get_model_status_text())


def get_q_table_stats(model_path):
    if not os.path.exists(model_path):
        return {"state_count": 0, "entry_count": 0}

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            table = json.load(f)
        state_count = len(table)
        entry_count = sum(len(v) for v in table.values() if isinstance(v, dict))
        return {"state_count": state_count, "entry_count": entry_count}
    except Exception:
        return {"state_count": 0, "entry_count": 0}


def scheduler_binary_path():
    # Prefer platform-native binary names in project directory.
    if os.name == "nt":
        exe_path = os.path.join(BASE_DIR, "scheduler.exe")
        if os.path.exists(exe_path):
            return exe_path
        alt_path = os.path.join(BASE_DIR, "scheduler")
        if os.path.exists(alt_path):
            return alt_path
        return exe_path

    # Linux/macOS expected binary path.
    return os.path.join(BASE_DIR, "scheduler")


def ensure_scheduler_binary():
    bin_path = scheduler_binary_path()
    src_path = os.path.join(BASE_DIR, "scheduler.c")

    if os.path.exists(bin_path):
        if not os.path.exists(src_path):
            raise FileNotFoundError("scheduler.c not found")

        # Rebuild automatically when source changed after binary.
        if os.path.getmtime(src_path) <= os.path.getmtime(bin_path):
            if os.name != "nt":
                # Ensure executable permission on Linux/macOS.
                os.chmod(bin_path, 0o755)
            return bin_path

    if not os.path.exists(src_path):
        raise FileNotFoundError("scheduler.c not found")

    if shutil.which("gcc") is None:
        raise FileNotFoundError("gcc not found")

    out_name = "scheduler.exe" if os.name == "nt" else "scheduler"
    subprocess.run(["gcc", "scheduler.c", "-o", out_name], check=True, cwd=BASE_DIR)
    bin_path = scheduler_binary_path()
    if os.name != "nt" and os.path.exists(bin_path):
        os.chmod(bin_path, 0o755)
    return bin_path


def run_scheduler(algo, param=None):
    scheduler_path = ensure_scheduler_binary()
    cmd = [scheduler_path, algo]
    if algo in ("RR", "PRIORITY_AGING"):
        value = str(param) if param is not None else "2"
        cmd.append(value)
    result = subprocess.run(cmd, check=False, cwd=BASE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Scheduler failed for {algo}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def run_ai_scheduler(process_rows, learning=False, save_q_table_to_disk=False):
    return simulate_ai_scheduler(
        process_rows,
        q_table_path=Q_TABLE_PATH,
        alpha=0.20,
        gamma=0.90,
        learning=learning,
        save_q_table_to_disk=save_q_table_to_disk,
        output_path=os.path.join(BASE_DIR, "output.txt"),
        metrics_path=os.path.join(BASE_DIR, "metrics.txt"),
        return_trace=True,
    )


def read_metrics_file():
    avg_tat = None
    avg_wt = None

    with open(os.path.join(BASE_DIR, "metrics.txt"), "r") as m:
        table_lines = [line.rstrip("\n") for line in m]

    for line in table_lines:
        if line.lower().startswith("average turnaround"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                avg_tat = float(parts[1].strip())
        elif line.lower().startswith("average waiting"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                avg_wt = float(parts[1].strip())

    return avg_tat, avg_wt, table_lines


def evaluate_candidates(candidates, process_rows):
    results = []
    for algo, q in candidates:
        if algo == "AI Scheduler":
            run_ai_scheduler(process_rows, learning=False, save_q_table_to_disk=False)
        else:
            run_scheduler(algo, q)
        avg_tat, avg_wt, _ = read_metrics_file()
        if avg_tat is None or avg_wt is None:
            continue
        results.append((algo, q, avg_tat, avg_wt))
    return results


def evaluate_algorithms(rr_quantum, process_rows):
    # Use the provided rr_quantum for both RR and PRIORITY_AGING so
    # comparison runs match the UI's selected quantum/aging value.
    return evaluate_candidates([
        ("FCFS", None),
        ("SJF", None),
        ("SRTF", None),
        ("RR", rr_quantum),
        ("PRIORITY", None),
        ("PRIORITY_AGING", rr_quantum),
        ("AI Scheduler", None),
    ], process_rows)


def snapshot_artifacts(names=("output.txt", "metrics.txt")):
    snapshots = {}
    for name in names:
        path = os.path.join(BASE_DIR, name)
        try:
            with open(path, "rb") as f:
                snapshots[name] = f.read()
        except FileNotFoundError:
            snapshots[name] = None
    return snapshots


def restore_artifacts(snapshots):
    for name, data in snapshots.items():
        path = os.path.join(BASE_DIR, name)
        if data is None:
            if os.path.exists(path):
                os.remove(path)
            continue
        with open(path, "wb") as f:
            f.write(data)


def build_suggested_algorithm(results):
    if not results:
        return "Recommended: unavailable"

    best_algo, best_q, best_tat, best_wt = min(results, key=lambda x: (x[3], x[2]))
    if best_algo == "RR":
        winner = f"RR (q={best_q})"
    else:
        winner = best_algo

    return f">> Recommended: {winner}  (WT {best_wt:.2f}, TAT {best_tat:.2f})"


def build_ai_explanation_log(ai_result):
    trace = (ai_result or {}).get("decision_trace", [])
    if not trace:
        return ["AI explanation unavailable for this run."]

    action_label = {
        "FCFS": "First Come First Serve",
        "SJF": "Shortest Job First",
        "PRIORITY": "Priority",
        "AGING_PRIORITY": "Aging Priority",
    }
    simple_reason = {
        "a high-priority process was waiting.": "A high-priority job was waiting.",
        "priority pressure remained high for the ready queue.": "Many urgent jobs were waiting.",
        "burst times were similar, so shortest-job scheduling stayed stable.": "Jobs were similar, so shortest first was stable.",
        "shorter jobs reduce waiting time when the queue starts to build up.": "Shorter jobs helped reduce waiting.",
        "shorter jobs were the safer choice for reducing waiting time.": "Shortest jobs helped reduce waiting.",
        "arrival order was a stable fallback when no urgent job stood out.": "No urgent job stood out, so arrival order was used.",
        "aging was used to protect long-waiting jobs from starvation.": "Aging protected long-waiting jobs.",
        "aging-balanced priority improved fairness without ignoring urgency.": "Aging improved fairness while handling urgent jobs.",
    }

    logs = ["Simple AI Explanation:"]
    for step in trace:
        action = action_label.get(step.get("action", ""), step.get("action", "Scheduling Rule"))
        reason = step.get("reason", "it was the best option at that moment").strip()
        reason = simple_reason.get(reason.lower(), reason)
        if reason:
            reason = reason[0].upper() + reason[1:]
        if len(reason) > 70:
            reason = reason[:67].rstrip() + "..."
        logs.append(
            f"Time {step['time']}: AI chose P{step['pid']} using {action}. Reason: {reason}"
        )

    if not logs:
        logs.append("AI explanation unavailable for this run.")

    return logs


def build_comparison_summary(results):
    if not results:
        return "Comparison summary unavailable."

    ai_result = next((row for row in results if row[0] == "AI Scheduler"), None)
    if not ai_result:
        return "Comparison summary unavailable."

    others = [row for row in results if row[0] != "AI Scheduler"]
    best_other = min(others, key=lambda row: (row[3], row[2])) if others else None
    best_overall = min(results, key=lambda row: (row[3], row[2]))

    if best_overall[0] == "AI Scheduler":
        return "AI Scheduler performed best for this workload."

    if best_overall[0] == "SJF":
        return "SJF performed best for this workload; AI was competitive."

    if best_other and ai_result[3] <= best_other[3] and ai_result[2] <= best_other[2]:
        return f"AI Scheduler matched or beat the best non-AI baseline ({best_other[0]})."

    return f"AI adapted dynamically, but {best_overall[0]} achieved the strongest metrics in this case."


def set_ai_log_lines(lines):
    if "ai_log_text" not in globals():
        return

    ai_log_text.config(state="normal")
    ai_log_text.delete("1.0", tk.END)
    ai_log_text.insert(tk.END, "\n".join(lines))
    ai_log_text.config(
        padx=8,
        pady=4,
        background="#FFFFFF",
        foreground="#1F3550",
        highlightthickness=1,
        highlightbackground="#D7E2F0",
        highlightcolor="#AAC2E0",
        spacing1=1,
        spacing3=2,
    )
    # Always show fresh logs from the top; avoids mid-line clipped view after reruns.
    ai_log_text.yview_moveto(0.0)
    ai_log_text.config(state="disabled")


def set_simulation_status(text, status="ready"):
    if "sim_status_val" not in globals():
        return

    style_name = {
        "ready": "StatusReady.TLabel",
        "running": "StatusRunning.TLabel",
        "completed": "StatusCompleted.TLabel",
        "error": "StatusError.TLabel",
    }.get(status, "StatusReady.TLabel")
    sim_status_val.config(text=text, style=style_name)
    if "status_value" in globals():
        value_map = {
            "ready": ("READY", "#475569"),
            "running": ("RUNNING", WARNING_ORANGE),
            "completed": ("COMPLETED", SUCCESS_GREEN),
            "error": ("ERROR", ERROR_RED),
        }
        val, color = value_map.get(status, ("READY", "#475569"))
        status_value.config(text=val, fg=color)
    if "status_badge" in globals():
        badge_map = {
            "ready": ("READY", "#475569"),
            "running": ("RUNNING", WARNING_ORANGE),
            "completed": ("COMPLETED", SUCCESS_GREEN),
            "error": ("ERROR", ERROR_RED),
        }
        badge_text, badge_color = badge_map.get(status, ("READY", "#475569"))
        status_badge.config(text=f" {badge_text} ", bg=badge_color)


def update_comparison_table(results):
    tree = globals().get("comparison_tree")
    if tree is None:
        return

    for item in tree.get_children():
        tree.delete(item)

    wanted = ("FCFS", "SJF", "SRTF", "RR", "PRIORITY", "PRIORITY_AGING", "AI Scheduler")
    by_algo = {algo: (q, wt, tat) for algo, q, tat, wt in results}
    best_algo = None
    if results:
        best_algo = min(results, key=lambda row: (row[3], row[2]))[0]

    for idx, algo in enumerate(wanted):
        base_tag = "even" if idx % 2 == 0 else "odd"
        if algo in by_algo:
            q, wt, tat = by_algo[algo]
            tags = [base_tag]
            if best_algo == algo:
                tags.append("best_algo")
            display_name = f"RR (q={q})" if algo == "RR" and q is not None else algo
            tree.insert("", "end", values=(display_name, f"{wt:.2f}", f"{tat:.2f}"), tags=tuple(tags))
        else:
            display_name = "RR" if algo == "RR" else algo
            tree.insert("", "end", values=(display_name, "-", "-"), tags=(base_tag,))


def render_comparison_chart(results, container):
    for child in container.winfo_children():
        child.destroy()

    if not results:
        empty_state = tk.Frame(container, bg="#ffffff", highlightthickness=1, highlightbackground=BORDER_COLOR)
        empty_state.pack(fill="both", expand=True)
        ttk.Label(
            empty_state,
            text="Run a comparison to see the chart.",
            style="Suggestion.TLabel",
            justify="center",
        ).pack(fill="both", expand=True, padx=24, pady=24)
        return

    normalized = []
    for row in results:
        if len(row) == 4:
            algo, q, tat, wt = row
            label = f"RR (q={q})" if algo == "RR" and q is not None else algo
        elif len(row) == 3:
            label, wt, tat = row
        else:
            continue
        normalized.append((label, float(wt), float(tat)))

    labels = [label for label, _, _ in normalized]
    wt_values = [wt for _, wt, _ in normalized]
    tat_values = [tat for _, _, tat in normalized]

    fig = plt.Figure(figsize=(8.2, 3.6), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f8fafc")
    x = list(range(len(labels)))
    width = 0.34

    wt_bars = ax.bar([i - width / 2 for i in x], wt_values, width=width, label="Avg WT", color="#2563eb", edgecolor="#1e3a8a", linewidth=0.8)
    tat_bars = ax.bar([i + width / 2 for i in x], tat_values, width=width, label="Avg TAT", color="#f59e0b", edgecolor="#b45309", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Time", labelpad=8, color="#334155")
    ax.set_title("Algorithm Comparison Graph", pad=12, fontweight="bold", color="#0f172a")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    max_value = max(wt_values + tat_values) if wt_values and tat_values else 0
    ax.set_ylim(0, max_value * 1.28 if max_value else 1)

    def annotate_bars(bars, color):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    annotate_bars(wt_bars, "#1d4ed8")
    annotate_bars(tat_bars, "#b45309")

    ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 1.02), borderaxespad=0.0)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    container._comparison_canvas = canvas


def load_sample_data():
    global last_loaded_sample_rows

    selected = int(process_count_var.get())
    selected_algo = algorithm_var.get()

    # Curated demo workloads per algorithm so behavior differences are visible in class/project showcases.
    sample_by_algo = {
        "FCFS": [
            ("1", "0", "8", "2"),
            ("2", "1", "4", "1"),
            ("3", "2", "2", "3"),
            ("4", "3", "6", "0"),
            ("5", "4", "3", "4"),
        ],
        "SJF": [
            ("1", "0", "9", "2"),
            ("2", "0", "2", "1"),
            ("3", "0", "5", "3"),
            ("4", "1", "1", "0"),
            ("5", "2", "4", "4"),
        ],
        "SRTF": [
            ("1", "0", "10", "2"),
            ("2", "1", "3", "1"),
            ("3", "2", "2", "3"),
            ("4", "3", "1", "0"),
            ("5", "4", "5", "4"),
        ],
        "RR": [
            ("1", "0", "9", "2"),
            ("2", "0", "4", "1"),
            ("3", "1", "7", "3"),
            ("4", "2", "5", "0"),
            ("5", "3", "3", "4"),
        ],
        "PRIORITY": [
            ("1", "0", "8", "4"),
            ("2", "0", "4", "0"),
            ("3", "1", "6", "1"),
            ("4", "2", "3", "2"),
            ("5", "3", "5", "3"),
        ],
        "PRIORITY_AGING": [
            ("1", "0", "10", "5"),
            ("2", "0", "3", "0"),
            ("3", "1", "8", "4"),
            ("4", "2", "2", "1"),
            ("5", "3", "4", "2"),
        ],
        "AI Scheduler": [
            ("1", "0", "9", "4"),
            ("2", "0", "2", "0"),
            ("3", "1", "5", "1"),
            ("4", "2", "7", "3"),
            ("5", "3", "3", "2"),
        ],
    }

    sample_rows = sample_by_algo.get(selected_algo, sample_by_algo["FCFS"])
    last_loaded_sample_rows = sample_rows

    for i in range(5):
        pid_entries[i].delete(0, tk.END)
        at_entries[i].delete(0, tk.END)
        bt_entries[i].delete(0, tk.END)
        pr_entries[i].delete(0, tk.END)

        if i < selected:
            pid_val, at_val, bt_val, pr_val = sample_rows[i]
            pid_entries[i].insert(0, pid_val)
            at_entries[i].insert(0, at_val)
            bt_entries[i].insert(0, bt_val)
            pr_entries[i].insert(0, pr_val)

    update_process_rows()
    set_simulation_status(
        f"Simulation Status: Ready ({selected_algo} {selected}-process sample loaded)",
        "ready",
    )
    show_toast("Sample data loaded")


def clear_inputs():
    current_process_count = process_count_var.get()

    for i in range(5):
        for widget in (pid_entries[i], at_entries[i], bt_entries[i], pr_entries[i]):
            widget.delete(0, tk.END)
            widget.configure(highlightthickness=1, highlightbackground=BORDER_COLOR, highlightcolor=BORDER_COLOR)

    # Keep the user's current process-count selection instead of forcing 3.
    process_count_var.set(current_process_count)
    quantum_var.set("2")
    update_process_rows()
    avg_tat_val.config(text="-")
    avg_wt_val.config(text="-")
    suggested_val.config(text="Recommended: run simulation to generate a suggestion")
    set_simulation_status("Simulation Status: Ready", "ready")
    for r in metrics_tree.get_children():
        metrics_tree.delete(r)
    update_comparison_table([])
    render_gantt_preview([])
    show_toast("Inputs cleared")


def update_ai_mode_ui():
    if "ai_mode_val" not in globals() or "ai_status_val" not in globals():
        return

    selected_algo = algorithm_var.get() if "algorithm_var" in globals() else ""

    if "train_ai_btn" in globals():
        train_ai_btn.config(state="disabled" if ai_training_in_progress else "normal")

    if ai_training_in_progress:
        ai_mode_val.config(text="AI Mode: Training")
        ai_status_val.config(text="AI Features: Training model...")
        if "ai_mode_badge" in globals():
            ai_mode_badge.config(bg=WARNING_ORANGE, text=" AI MODE: TRAINING ")
        return

    if selected_algo == "AI Scheduler":
        ai_mode_val.config(text="AI Mode: Inference (Read Only)")
        ai_status_val.config(text="AI Features: Active")
        if "ai_mode_badge" in globals():
            ai_mode_badge.config(bg=SUCCESS_GREEN, text=" AI MODE: ACTIVE ")
        return

    if ai_model_ready:
        ai_mode_val.config(text="AI Mode: Read Only Suggestion (Ready)")
        ai_status_val.config(text="AI Features: Read Only Suggestion")
        if "ai_mode_badge" in globals():
            ai_mode_badge.config(bg="#0ea5e9", text=" AI MODE: READY ")
    else:
        ai_mode_val.config(text="AI Mode: Off")
        ai_status_val.config(text="AI Features: Inactive")
        if "ai_mode_badge" in globals():
            ai_mode_badge.config(bg="#e2e8f0", text=" AI MODE: OFF ")


def on_train_ai_model_click():
    global ai_training_in_progress

    if ai_training_in_progress:
        return

    ai_training_in_progress = True
    if "train_ai_btn" in globals():
        train_ai_btn.config(state="disabled")
    update_ai_mode_ui()

    def worker():
        try:
            train_model(q_table_path=Q_TABLE_PATH)
        except Exception as exc:
            root.after(0, lambda: finish_train_ai(False, str(exc)))
            return
        root.after(0, lambda: finish_train_ai(True, None))

    threading.Thread(target=worker, daemon=True).start()


def finish_train_ai(success, error_text=None):
    global ai_training_in_progress, ai_model_ready

    ai_training_in_progress = False
    if success:
        ai_model_ready = True

    if "train_ai_btn" in globals():
        train_ai_btn.config(state="normal")

    update_model_status_ui()
    update_ai_mode_ui()

    if success:
        metadata = load_training_metadata(MODEL_META_PATH)
        last_run_episodes = int(metadata.get("last_run_episodes", 0))
        messagebox.showinfo(
            "AI Training",
            "Q-table Model Trained Successfully\n"
            f"Training completed using ~{last_run_episodes} episodes.\n"
            "Q-table model saved successfully.\n"
            "Training used a curated versatile workload mix.",
        )
    else:
        messagebox.showerror("AI Training", f"AI model training failed: {error_text}")


def on_reset_ai_model_click():
    global ai_model_ready

    confirm = messagebox.askyesno(
        "Reset AI Model",
        "This will delete the saved Q-table model and reset the AI model. Continue?",
    )
    if not confirm:
        return

    try:
        if os.path.exists(Q_TABLE_PATH):
            os.remove(Q_TABLE_PATH)
        if os.path.exists(MODEL_META_PATH):
            os.remove(MODEL_META_PATH)
        ai_model_ready = False
        update_model_status_ui()
        update_ai_mode_ui()
        set_ai_log_lines(["AI model reset. Train AI Model to create a new Q-table model."])
        global comparison_summary_val_text
        comparison_summary_val_text = "Insight: comparison summary will appear here after simulation."
        messagebox.showinfo("Reset AI Model", "AI Model Reset Successfully")
    except Exception as exc:
        messagebox.showerror("Reset AI Model", f"Failed to reset AI model: {exc}")


def collect_process_rows(requires_priority=False):
    process_count = int(process_count_var.get())
    process_rows = []
    invalid_at_processes = []
    invalid_bt_processes = []
    invalid_widgets = []

    for i in range(5):
        for widget in (pid_entries[i], at_entries[i], bt_entries[i], pr_entries[i]):
            widget.configure(highlightthickness=1, highlightbackground=BORDER_COLOR, highlightcolor=PRIMARY_BLUE)

    for i in range(process_count):
        pid_val = pid_entries[i].get().strip()
        at_val = at_entries[i].get().strip()
        bt_val = bt_entries[i].get().strip()
        pr_val = pr_entries[i].get().strip()

        if not pid_val or not at_val or not bt_val:
            invalid_widgets.extend((pid_entries[i], at_entries[i], bt_entries[i]))
            for widget in invalid_widgets:
                widget.configure(highlightbackground=ERROR_RED, highlightcolor=ERROR_RED)
            show_toast("Invalid input detected")
            messagebox.showerror("Input Error", f"Process {i + 1} has empty fields")
            return None

        if requires_priority and not pr_val:
            pr_entries[i].configure(highlightbackground=ERROR_RED, highlightcolor=ERROR_RED)
            show_toast("Invalid input detected")
            messagebox.showerror("Input Error", f"Process {i + 1} has empty priority")
            return None

        if not pid_val.isdigit():
            pid_entries[i].configure(highlightbackground=ERROR_RED, highlightcolor=ERROR_RED)
            show_toast("Invalid input detected")
            messagebox.showerror("Input Error", f"Process {i + 1} PID must be a non-negative integer")
            return None

        row_valid = True

        try:
            at_int = int(at_val)
            if at_int < 0:
                invalid_at_processes.append(str(i + 1))
                invalid_widgets.append(at_entries[i])
                row_valid = False
        except ValueError:
            invalid_at_processes.append(str(i + 1))
            invalid_widgets.append(at_entries[i])
            row_valid = False

        try:
            bt_int = int(bt_val)
            if bt_int <= 0:
                invalid_bt_processes.append(str(i + 1))
                invalid_widgets.append(bt_entries[i])
                row_valid = False
        except ValueError:
            invalid_bt_processes.append(str(i + 1))
            invalid_widgets.append(bt_entries[i])
            row_valid = False

        if requires_priority and not pr_val.isdigit():
            pr_entries[i].configure(highlightbackground=ERROR_RED, highlightcolor=ERROR_RED)
            show_toast("Invalid input detected")
            messagebox.showerror("Input Error", f"Process {i + 1} priority must be a non-negative integer")
            return None

        if not requires_priority:
            pr_val = "0"

        if row_valid:
            process_rows.append((pid_val, at_val, bt_val, pr_val))

    validation_errors = []
    if invalid_at_processes:
        validation_errors.append(
            f"Process(es) {', '.join(invalid_at_processes)} arrival time must be non negative value"
        )
    if invalid_bt_processes:
        validation_errors.append(
            f"Process(es) {', '.join(invalid_bt_processes)} burst time must be greater than 0"
        )

    if validation_errors:
        for widget in invalid_widgets:
            widget.configure(highlightbackground=ERROR_RED, highlightcolor=ERROR_RED)
        show_toast("Invalid input detected")
        messagebox.showerror("Input Error", "\n".join(validation_errors))
        return None

    return process_rows


def write_input_rows(process_rows):
    with open(os.path.join(BASE_DIR, "input.txt"), "w") as f:
        f.write(f"{len(process_rows)}\n")
        for pid_val, at_val, bt_val, pr_val in process_rows:
            f.write(f"{pid_val} {at_val} {bt_val} {pr_val}\n")


def draw_gantt(save_path=None, animate=True, step_delay=0.35):

    schedule = []
    with open(os.path.join(BASE_DIR, "output.txt"), "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            data = line.split()
            if len(data) < 3:
                continue
            schedule.append((int(data[0]), int(data[1]), int(data[2])))

    schedule.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(11, 4.1))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    # Premium color palette: deep navy, teal, amber, royal purple, crimson, emerald
    colors = ["#0b3d91", "#0ea5a4", "#d97706", "#7c3aed", "#dc2626", "#059669"]

    end_times = [seg_start + seg_duration for _, seg_start, seg_duration in schedule]
    makespan = int(max(end_times)) if end_times else 0
    x_axis_max = makespan if makespan > 0 else 1
    x_axis_pad = 0.18 if x_axis_max > 1 else 0.35

    ax.set_xlabel("Time", color="#334155", labelpad=8)
    ax.set_xlim(-x_axis_pad, x_axis_max + x_axis_pad)
    ax.set_xticks(list(range(0, x_axis_max + 1)))
    ax.set_yticks([])
    ax.set_title("CPU Scheduling Gantt Chart", pad=12, fontweight="bold", color="#0f172a")
    ax.grid(axis="x", linestyle="--", alpha=0.22, color="#94a3b8")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", colors="#475569")
    ax.margins(x=0.01)
    ax.axvline(0, color="#111827", linewidth=1.2, zorder=0)
    ax.axvline(x_axis_max, color="#111827", linewidth=1.2, zorder=0)

    timeline = []
    current_time = 0
    for proc_pid, proc_start, proc_duration in schedule:
        if proc_start > current_time:
            timeline.append(("idle", "IDLE", current_time, proc_start - current_time, "#F1F5F9"))
        timeline.append(("proc", f"P{proc_pid}", proc_start, proc_duration, colors[proc_pid % len(colors)]))
        current_time = max(current_time, proc_start + proc_duration)

    def draw_segment(kind, label, seg_start, seg_duration, color):
        text_color = "#334155" if kind == "idle" else "#ffffff"
        label_fontsize = 9
        if seg_duration <= 1:
            label_fontsize = 6
        elif seg_duration <= 2:
            label_fontsize = 7
        elif seg_duration <= 3:
            label_fontsize = 8

        if kind == "idle":
            bar = ax.barh(
                0,
                seg_duration,
                left=seg_start,
                height=0.62,
                color="#f8fafc",
                edgecolor="#cbd5e1",
                hatch="///",
            )[0]
            text = ax.text(
                seg_start + seg_duration / 2,
                0,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=label_fontsize,
                rotation=0,
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="#ffffff", edgecolor="none", alpha=0.72),
            )
            text.set_clip_path(bar)
            return

        bar = ax.barh(
            0,
            seg_duration,
            left=seg_start,
            height=0.62,
            color=color,
            edgecolor="#071428",
            linewidth=0.9,
        )[0]
        text = ax.text(
            seg_start + seg_duration / 2,
            0,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontweight="bold",
            fontsize=label_fontsize,
            rotation=0,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="#071428", edgecolor="none", alpha=0.14),
        )
        text.set_clip_path(bar)

    if animate and timeline:
        plt.show(block=False)
        for segment in timeline:
            draw_segment(*segment)
            fig.canvas.draw_idle()
            plt.pause(step_delay)
        fig.canvas.draw_idle()
    else:
        for segment in timeline:
            draw_segment(*segment)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


def run_simulation():
    set_simulation_status("Running simulation...", "running")
    process_rows = collect_process_rows(requires_priority=algorithm_var.get() in ("PRIORITY", "PRIORITY_AGING", "AI Scheduler"))
    if process_rows is None:
        set_simulation_status("Simulation Status: Ready", "ready")
        return

    algo = algorithm_var.get()
    write_input_rows(process_rows)

    q = quantum_var.get()
    if not q.isdigit() or int(q) <= 0:
        q = "2"
    run_param = q if algo in ("RR", "PRIORITY_AGING") else None

    if algo in ("RR", "PRIORITY_AGING"):
        quantum_var.set(q)

    ai_result = None
    try:
        if algo == "AI Scheduler":
            ai_result = run_ai_scheduler(process_rows)
        else:
            run_scheduler(algo, run_param)
    except FileNotFoundError:
        if os.name == "nt":
            build_help = "Install gcc (MinGW) or build manually: gcc scheduler.c -o scheduler.exe"
        else:
            build_help = "Install gcc (Ubuntu: sudo apt install build-essential)"
        messagebox.showerror("Scheduler Error", f"Scheduler build failed. {build_help}")
        set_simulation_status("Simulation Status: Error", "error")
        return
    except Exception as e:
        messagebox.showerror("Scheduler Error", f"Failed to run scheduler: {e}")
        set_simulation_status("Simulation Status: Error", "error")
        return

    if algo == "AI Scheduler":
        set_ai_log_lines(build_ai_explanation_log(ai_result))
    else:
        set_ai_log_lines(["AI explanation log is available when AI Scheduler runs."])

    try:
        avg_tat, avg_wt, table_lines = read_metrics_file()
        if avg_tat is not None:
            avg_tat_val.config(text=f"{avg_tat:.2f}")
        if avg_wt is not None:
            avg_wt_val.config(text=f"{avg_wt:.2f}")

        metrics_rows = []
        header_found = False
        for line in table_lines:
            if not header_found:
                if line.strip().upper().startswith("PID"):
                    header_found = True
                continue
            if line.lower().startswith("average"):
                break
            parts = line.split()
            if len(parts) >= 7:
                metrics_rows.append(parts[:7])

        for r in metrics_tree.get_children():
            metrics_tree.delete(r)
        for idx, row in enumerate(metrics_rows):
            metrics_tree.insert('', 'end', values=row, tags=(("even" if idx % 2 == 0 else "odd"),))

        global comparison_summary_val_text, last_run_process_rows, last_run_quantum
        comparison_results = []
        artifacts_before_compare = None
        try:
            # Preserve current run artifacts so the dashboard keeps selected algorithm details visible.
            artifacts_before_compare = snapshot_artifacts()
            comparison_results = evaluate_algorithms(q, process_rows)
        except Exception:
            comparison_results = []
        finally:
            if artifacts_before_compare is not None:
                restore_artifacts(artifacts_before_compare)

        if comparison_results:
            suggested_val.config(text=build_suggested_algorithm(comparison_results))
            update_comparison_table(comparison_results)
            comparison_summary_val_text = build_comparison_summary(comparison_results)
        else:
            suggested_val.config(text="Recommended: unavailable")
            update_comparison_table([])
            comparison_summary_val_text = "Comparison summary unavailable."

        last_run_process_rows = process_rows
        last_run_quantum = q

        set_simulation_status("Simulation Completed", "completed")
        animate_result_cards()
        animate_gantt_preview(parse_output_schedule())
        show_toast("Simulation completed successfully")
    except FileNotFoundError:
        avg_tat_val.config(text="-")
        avg_wt_val.config(text="-")
        suggested_val.config(text="Recommended: unavailable")
        update_comparison_table([])
        comparison_summary_val_text = "Comparison summary unavailable."
        set_simulation_status("Simulation Status: Error", "error")
        show_toast("Simulation failed")


def show_chart():
    # show current gantt without saving to disk
    try:
        draw_gantt()
    except Exception as e:
        messagebox.showerror("Chart Error", f"Failed to show chart: {e}")


# Global reference for modal windows
modal_windows = {}


def apply_button_hover(widget, primary=False):
    normal_bg = PRIMARY_BLUE if primary else SIDEBAR_BG
    hover_bg = HOVER_BLUE if primary else "#f3f4f6"
    fg = "#ffffff" if primary else CARD_TEXT
    widget.configure(
        bg=normal_bg,
        fg=fg,
        activebackground=PRESS_BLUE if primary else "#e5e7eb",
        activeforeground=fg,
        relief="flat",
        bd=0,
        cursor="hand2",
    )
    widget.bind("<Enter>", lambda _e: widget.configure(bg=hover_bg))
    widget.bind("<Leave>", lambda _e: widget.configure(bg=normal_bg))


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            bg="#111827",
            fg="#e5e7eb",
            font=(base_font, 9),
            padx=8,
            pady=4,
        )
        label.pack()

    def hide(self, _event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def show_toast(message):
    if "root" not in globals():
        return
    toast = tk.Toplevel(root)
    toast.overrideredirect(True)
    toast.attributes("-topmost", True)
    toast.configure(bg="#111827")
    x = root.winfo_rootx() + root.winfo_width() - 280
    y = root.winfo_rooty() + 40
    toast.geometry(f"260x44+{x}+{y}")
    tk.Label(
        toast,
        text=message,
        bg="#111827",
        fg="#f8fafc",
        font=(base_font, 9, "bold"),
    ).pack(fill="both", expand=True, padx=10, pady=8)
    toast.after(1800, toast.destroy)


def parse_output_schedule():
    schedule = []
    try:
        with open(os.path.join(BASE_DIR, "output.txt"), "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    schedule.append((int(parts[0]), int(parts[1]), int(parts[2])))
    except FileNotFoundError:
        return []
    return sorted(schedule, key=lambda x: x[1])


def render_gantt_preview(schedule, progress=1.0):
    canvas = globals().get("gantt_canvas")
    if canvas is None:
        return
    canvas.delete("all")
    width = max(canvas.winfo_width(), 320)
    height = max(canvas.winfo_height(), 120)
    canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="#e5e7eb")
    if not schedule:
        canvas.create_text(width / 2, height / 2, text="Run simulation to preview Gantt chart", fill=MUTED_TEXT, font=(base_font, 10))
        return

    end_time = max(start + dur for _, start, dur in schedule)
    end_time = max(end_time, 1)
    left_pad, right_pad = 30, 20
    bar_y, bar_h = 46, 28
    usable = width - left_pad - right_pad
    count = max(1, int(len(schedule) * progress))
    view = schedule[:count]
    # Premium preview palette matching the main Gantt chart
    palette = ["#0b3d91", "#0ea5a4", "#d97706", "#7c3aed", "#dc2626", "#059669"]

    for pid, start, dur in view:
        x0 = left_pad + (start / end_time) * usable
        x1 = left_pad + ((start + dur) / end_time) * usable
        color = palette[pid % len(palette)]
        canvas.create_rectangle(x0, bar_y, x1, bar_y + bar_h, fill=color, outline="#0f172a")
        canvas.create_text((x0 + x1) / 2, bar_y + (bar_h / 2), text=f"P{pid}", fill="white", font=(base_font, 9, "bold"))
        canvas.create_text(x0, bar_y + bar_h + 14, text=str(start), fill=MUTED_TEXT, font=(base_font, 8))

    canvas.create_text(left_pad + usable, bar_y + bar_h + 14, text=str(end_time), fill=MUTED_TEXT, font=(base_font, 8))
    canvas.create_text(left_pad, 24, text="Gantt Preview", anchor="w", fill=CARD_TEXT, font=(base_font, 10, "bold"))


def animate_gantt_preview(schedule):
    if not schedule:
        render_gantt_preview([])
        return
    steps = 6

    def draw_step(step=1):
        render_gantt_preview(schedule, progress=step / steps)
        if step < steps:
            root.after(30, lambda: draw_step(step + 1))

    draw_step()


def animate_result_cards():
    cards = [status_card, wt_card, tat_card, suggested_frame]
    for card in cards:
        card.configure(highlightbackground="#93c5fd", highlightthickness=2)
    root.after(140, lambda: [c.configure(highlightbackground=BORDER_COLOR, highlightthickness=1) for c in cards])


def show_metrics_window():
    """Open a Toplevel window displaying per-process metrics table."""
    if "metrics_modal" in modal_windows and modal_windows["metrics_modal"].winfo_exists():
        modal_windows["metrics_modal"].lift()
        return

    metrics_win = tk.Toplevel(root)
    metrics_win.title("Per-process Metrics")
    metrics_win.geometry("650x350")
    metrics_win.configure(bg="#f5f7fa")
    modal_windows["metrics_modal"] = metrics_win

    # Header
    header = ttk.Label(metrics_win, text="Per-process Metrics", font=(base_font, 12, 'bold'))
    header.pack(side='top', fill='x', padx=12, pady=(12, 8))

    # Treeview frame
    frame = ttk.Frame(metrics_win)
    frame.pack(side='top', fill='both', expand=True, padx=12, pady=(0, 12))
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    modal_metrics_tree = ttk.Treeview(frame, style="MetricsTree.Treeview", columns=("PID","AT","BT","PR","CT","TAT","WT"), show='headings', height=12)
    for col in ("PID","AT","BT","PR","CT","TAT","WT"):
        modal_metrics_tree.heading(col, text=col)
    modal_metrics_tree.column("PID", width=78, anchor='center')
    modal_metrics_tree.column("AT", width=78, anchor='center')
    modal_metrics_tree.column("BT", width=78, anchor='center')
    modal_metrics_tree.column("PR", width=62, anchor='center')
    modal_metrics_tree.column("CT", width=95, anchor='center')
    modal_metrics_tree.column("TAT", width=95, anchor='center')
    modal_metrics_tree.column("WT", width=95, anchor='center')

    # Copy current data from main tree
    modal_metrics_tree.tag_configure("odd", background="#f8fafc")
    modal_metrics_tree.tag_configure("even", background="#ffffff")
    if "metrics_tree" in globals():
        for idx, item in enumerate(metrics_tree.get_children()):
            values = metrics_tree.item(item, 'values')
            modal_metrics_tree.insert('', 'end', values=values, tags=("even" if idx % 2 == 0 else "odd",))

    scrollbar = ttk.Scrollbar(frame, orient='vertical', command=modal_metrics_tree.yview)
    modal_metrics_tree.configure(yscrollcommand=scrollbar.set)
    modal_metrics_tree.grid(row=0, column=0, sticky='nsew')
    scrollbar.grid(row=0, column=1, sticky='ns')

    # Close button
    btn_frame = ttk.Frame(metrics_win)
    btn_frame.pack(side='bottom', fill='x', padx=12, pady=(0, 12))
    close_btn = tk.Button(btn_frame, text='Close', command=lambda: metrics_win.destroy(), font=(base_font, 9))
    close_btn.pack(side='right')
    apply_button_hover(close_btn, primary=False)
    metrics_win.bind("<Escape>", lambda _e: metrics_win.destroy())


def show_comparison_window():
    """Open a Toplevel window displaying algorithm comparison table."""
    global comparison_summary_val_text
    if "comparison_modal" in modal_windows and modal_windows["comparison_modal"].winfo_exists():
        modal_windows["comparison_modal"].lift()
        return

    comp_win = tk.Toplevel(root)
    comp_win.title("Algorithm Comparison")
    comp_win.geometry("980x700")
    comp_win.configure(bg="#f5f7fa")
    modal_windows["comparison_modal"] = comp_win

    # Header
    header = ttk.Label(comp_win, text="Algorithm Comparison (Read Only)", font=(base_font, 12, 'bold'))
    header.pack(side='top', fill='x', padx=12, pady=(12, 8))

    # Table + chart container
    content = ttk.Frame(comp_win)
    content.pack(side='top', fill='both', expand=True, padx=12, pady=(0, 12))
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)
    content.rowconfigure(1, weight=1)

    # Treeview frame
    frame = ttk.Frame(content)
    frame.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    modal_comparison_tree = ttk.Treeview(frame, columns=("ALGO", "WT", "TAT"), show='headings', height=7)
    modal_comparison_tree.heading("ALGO", text="Algorithm")
    modal_comparison_tree.heading("WT", text="Avg WT")
    modal_comparison_tree.heading("TAT", text="Avg TAT")
    modal_comparison_tree.column("ALGO", width=280, anchor='w')
    modal_comparison_tree.column("WT", width=180, anchor='center')
    modal_comparison_tree.column("TAT", width=180, anchor='center')
    modal_comparison_tree.tag_configure("best_algo", background="#e8f5e9", foreground="#14532d")
    modal_comparison_tree.tag_configure("odd", background="#f8fafc")
    modal_comparison_tree.tag_configure("even", background="#ffffff")

    # Compute comparison on demand if not already available.
    if "comparison_tree" in globals() and len(comparison_tree.get_children()) == 0:
        if "last_run_process_rows" in globals() and last_run_process_rows:
            artifacts_before_compare = None
            try:
                artifacts_before_compare = snapshot_artifacts()
                results = evaluate_algorithms(last_run_quantum, last_run_process_rows)
                update_comparison_table(results)
                comparison_summary_val_text = build_comparison_summary(results)
            except Exception:
                comparison_summary_val_text = "Comparison summary unavailable."
            finally:
                if artifacts_before_compare is not None:
                    restore_artifacts(artifacts_before_compare)

    if "comparison_tree" in globals():
        for idx, item in enumerate(comparison_tree.get_children()):
            values = comparison_tree.item(item, 'values')
            tags = comparison_tree.item(item, 'tags')
            base_tag = "even" if idx % 2 == 0 else "odd"
            row_tags = tuple(list(tags) + [base_tag]) if tags else (base_tag,)
            modal_comparison_tree.insert('', 'end', values=values, tags=row_tags)

    scrollbar = ttk.Scrollbar(frame, orient='vertical', command=modal_comparison_tree.yview)
    modal_comparison_tree.configure(yscrollcommand=scrollbar.set)
    modal_comparison_tree.grid(row=0, column=0, sticky='nsew')
    scrollbar.grid(row=0, column=1, sticky='ns')

    # Comparison chart frame
    chart_card = tk.Frame(content, bg="#ffffff", highlightthickness=1, highlightbackground=BORDER_COLOR)
    chart_card.grid(row=1, column=0, sticky='nsew')
    chart_card.grid_columnconfigure(0, weight=1)
    chart_card.grid_rowconfigure(1, weight=1)
    ttk.Label(chart_card, text="Visual Comparison", font=(base_font, 11, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

    chart_panel = ttk.Frame(chart_card)
    chart_panel.grid(row=1, column=0, sticky='nsew', padx=8, pady=(0, 8))
    chart_panel.columnconfigure(0, weight=1)
    chart_panel.rowconfigure(0, weight=1)
    chart_results = []
    if "comparison_tree" in globals():
        for item in comparison_tree.get_children():
            values = comparison_tree.item(item, 'values')
            if len(values) >= 3 and values[1] not in ("-", "") and values[2] not in ("-", ""):
                chart_results.append((values[0], values[1], values[2]))
    render_comparison_chart(chart_results, chart_panel)

    # Insight label
    insight_text = comparison_summary_val_text

    insight = ttk.Label(comp_win, text=insight_text, style='Suggestion.TLabel', justify='left', wraplength=550)
    insight.pack(side='top', fill='x', padx=12, pady=(0, 8))

    # Close button
    btn_frame = ttk.Frame(comp_win)
    btn_frame.pack(side='bottom', fill='x', padx=12, pady=(0, 12))
    close_btn = tk.Button(btn_frame, text='Close', command=lambda: comp_win.destroy(), font=(base_font, 9))
    close_btn.pack(side='right')
    apply_button_hover(close_btn, primary=False)
    comp_win.bind("<Escape>", lambda _e: comp_win.destroy())


def show_ai_log_window():
    """Open a Toplevel window displaying AI explanation log."""
    if "ai_log_modal" in modal_windows and modal_windows["ai_log_modal"].winfo_exists():
        modal_windows["ai_log_modal"].lift()
        return

    log_win = tk.Toplevel(root)
    log_win.title("AI Explanation Log")
    log_win.geometry("700x450")
    log_win.configure(bg="#f5f7fa")
    modal_windows["ai_log_modal"] = log_win

    # Header
    header = ttk.Label(log_win, text="AI Scheduling Decisions", font=(base_font, 12, 'bold'))
    header.pack(side='top', fill='x', padx=12, pady=(12, 8))

    # Text widget
    text_frame = ttk.Frame(log_win)
    text_frame.pack(side='top', fill='both', expand=True, padx=12, pady=(0, 12))
    text_frame.columnconfigure(0, weight=1)
    text_frame.rowconfigure(0, weight=1)

    modal_log_text = tk.Text(text_frame, height=15, wrap='word', font=(base_font, 10), borderwidth=1)
    modal_log_text.grid(row=0, column=0, sticky='nsew')

    scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=modal_log_text.yview)
    scrollbar.grid(row=0, column=1, sticky='ns')
    modal_log_text.configure(yscrollcommand=scrollbar.set)

    # Copy current content
    if "ai_log_text" in globals():
        content = ai_log_text.get("1.0", tk.END)
        modal_log_text.insert("1.0", content)

    modal_log_text.config(
        padx=8,
        pady=4,
        background="#FFFFFF",
        foreground="#1F3550",
        highlightthickness=1,
        highlightbackground="#D7E2F0",
        highlightcolor="#AAC2E0",
        spacing1=1,
        spacing3=2,
    )
    modal_log_text.config(state="disabled")

    # Close button
    btn_frame = ttk.Frame(log_win)
    btn_frame.pack(side='bottom', fill='x', padx=12, pady=(0, 12))
    close_btn = tk.Button(btn_frame, text='Close', command=lambda: log_win.destroy(), font=(base_font, 9))
    close_btn.pack(side='right')
    apply_button_hover(close_btn, primary=False)
    log_win.bind("<Escape>", lambda _e: log_win.destroy())


root = tk.Tk()
root.title("CPU Scheduling Visualizer")
root.geometry("1220x760")
root.minsize(1080, 680)

BASE_BG = "#f5f7fa"
SIDEBAR_BG = "#ffffff"
CARD_BG = "#ffffff"
CARD_TEXT = "#0f172a"
MUTED_TEXT = "#64748b"
PRIMARY_BLUE = "#2563eb"
HOVER_BLUE = "#1d4ed8"
PRESS_BLUE = "#1e40af"
SUCCESS_GREEN = "#16a34a"
WARNING_ORANGE = "#f59e0b"
ERROR_RED = "#dc2626"
BORDER_COLOR = "#e5e7eb"

base_font = 'Segoe UI' if os.name == 'nt' else 'DejaVu Sans'
root.configure(bg=BASE_BG)

style = ttk.Style(root)
try:
    style.theme_use("clam")
except tk.TclError:
    pass
style.configure("TFrame", background=BASE_BG)
style.configure("TLabel", background=BASE_BG, foreground=CARD_TEXT, font=(base_font, 10))
style.configure(
    "Selector.TCombobox",
    fieldbackground="#f8fafc",
    background="#edf2ff",
    foreground=CARD_TEXT,
    arrowcolor=PRIMARY_BLUE,
    padding=4,
)
style.map(
    "Selector.TCombobox",
    fieldbackground=[("readonly", "#f8fafc"), ("focus", "#ffffff")],
    foreground=[("readonly", CARD_TEXT)],
)
style.configure("DashboardTree.Treeview", rowheight=24, font=(base_font, 10), background="#ffffff", fieldbackground="#ffffff", foreground=CARD_TEXT)
style.configure("DashboardTree.Treeview.Heading", font=(base_font, 10, "bold"), background="#f8fafc", foreground=CARD_TEXT)
style.configure("MetricsTree.Treeview", rowheight=30, font=(base_font, 11), background="#ffffff", fieldbackground="#ffffff", foreground="#0b1324")
style.configure("MetricsTree.Treeview.Heading", font=(base_font, 11, "bold"), background="#eef2f7", foreground="#0f172a")
style.configure("StatusReady.TLabel", background=SIDEBAR_BG, foreground=MUTED_TEXT, font=(base_font, 9, "bold"))
style.configure("StatusRunning.TLabel", background=SIDEBAR_BG, foreground=WARNING_ORANGE, font=(base_font, 9, "bold"))
style.configure("StatusCompleted.TLabel", background=SIDEBAR_BG, foreground=SUCCESS_GREEN, font=(base_font, 9, "bold"))
style.configure("StatusError.TLabel", background=SIDEBAR_BG, foreground=ERROR_RED, font=(base_font, 9, "bold"))


def validate_numeric_input(value):
    if value == "" or value == "-":
        return True
    if value.startswith("-"):
        return value[1:].isdigit() and value.count("-") == 1
    return value.isdigit()


def make_card(parent, title, row, column, colspan=1):
    shadow = tk.Frame(parent, bg="#dbe3ec")
    shadow.grid(row=row, column=column, columnspan=colspan, sticky="nsew", padx=8, pady=8)
    card = tk.Frame(shadow, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER_COLOR)
    card.pack(fill="both", expand=True, padx=(0, 2), pady=(0, 2))
    tk.Label(card, text=title, bg=CARD_BG, fg=MUTED_TEXT, font=(base_font, 10, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
    return card


header = tk.Frame(root, bg="#ffffff", height=62, highlightthickness=1, highlightbackground=BORDER_COLOR)
header.grid(row=0, column=0, columnspan=2, sticky="nsew")
header.grid_columnconfigure(0, weight=1)
tk.Label(header, text="CPU Scheduling Simulator", bg="#ffffff", fg=CARD_TEXT, font=(base_font, 16, "bold")).grid(row=0, column=0, sticky="w", padx=16, pady=14)

content = tk.PanedWindow(root, orient="horizontal", sashwidth=6, bg=BASE_BG, bd=0, relief="flat")
content.grid(row=1, column=0, columnspan=2, sticky="nsew")
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

sidebar = tk.Frame(content, bg=SIDEBAR_BG, width=355)
dashboard = tk.Frame(content, bg=BASE_BG)
content.add(sidebar, minsize=330)
content.add(dashboard, minsize=620)

sidebar.grid_columnconfigure(0, weight=1)
dashboard.grid_columnconfigure(0, weight=1)
dashboard.grid_rowconfigure(2, weight=1)

inputs_card = tk.Frame(sidebar, bg=SIDEBAR_BG, highlightthickness=1, highlightbackground=BORDER_COLOR)
inputs_card.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
inputs_card.grid_columnconfigure(1, weight=1)
tk.Label(inputs_card, text="Inputs", bg=SIDEBAR_BG, fg=CARD_TEXT, font=(base_font, 12, "bold")).grid(row=0, column=0, columnspan=6, sticky="w", padx=10, pady=(10, 10))

tk.Label(inputs_card, text="Algorithm", bg=SIDEBAR_BG, fg=MUTED_TEXT).grid(row=1, column=0, sticky="w", padx=(10, 6))
algorithm_var = tk.StringVar()
algorithm = ttk.Combobox(inputs_card, textvariable=algorithm_var, state="readonly", values=("FCFS", "SJF", "SRTF", "RR", "PRIORITY", "PRIORITY_AGING", "AI Scheduler"), style="Selector.TCombobox")
algorithm.current(0)
algorithm.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 10), pady=(0, 8))

tk.Label(inputs_card, text="Processes", bg=SIDEBAR_BG, fg=MUTED_TEXT).grid(row=1, column=3, sticky="e")
process_count_var = tk.StringVar(value="3")
vcmd = (root.register(validate_numeric_input), "%P")
process_count = ttk.Combobox(
    inputs_card,
    textvariable=process_count_var,
    state="readonly",
    values=("1", "2", "3", "4", "5"),
    width=4,
    style="Selector.TCombobox",
)
process_count.grid(row=1, column=4, sticky="w", padx=(8, 10), pady=(0, 8))

quantum_label = tk.Label(inputs_card, text="Quantum", bg=SIDEBAR_BG, fg=MUTED_TEXT)
quantum_var = tk.StringVar(value="2")
quantum_entry = tk.Entry(inputs_card, textvariable=quantum_var, width=6, font=(base_font, 10), relief="flat", highlightthickness=1, highlightbackground=BORDER_COLOR, validate="key", validatecommand=vcmd)
quantum_label.grid(row=1, column=5, sticky="e")
quantum_entry.grid(row=1, column=6, sticky="w", padx=(6, 10), pady=(0, 8))
quantum_label.grid_remove()
quantum_entry.grid_remove()

proc_frame = tk.Frame(inputs_card, bg=SIDEBAR_BG)
proc_frame.grid(row=2, column=0, columnspan=7, sticky="ew", padx=10, pady=(4, 10))
for c, text in enumerate(("PID", "Arrival", "Burst")):
    tk.Label(proc_frame, text=text, width=10, bg=SIDEBAR_BG, fg=MUTED_TEXT, font=(base_font, 9, "bold")).grid(row=0, column=c, padx=4, pady=(0, 6))
priority_header = tk.Label(proc_frame, text="Priority", width=10, bg=SIDEBAR_BG, fg=MUTED_TEXT, font=(base_font, 9, "bold"))

pid_entries, at_entries, bt_entries, pr_entries = [], [], [], []
for row in range(5):
    pid_entry = tk.Entry(proc_frame, width=9, font=(base_font, 10), relief="flat", highlightthickness=1, highlightbackground=BORDER_COLOR, highlightcolor=PRIMARY_BLUE, validate="key", validatecommand=vcmd)
    at_entry = tk.Entry(proc_frame, width=11, font=(base_font, 10), relief="flat", highlightthickness=1, highlightbackground=BORDER_COLOR, highlightcolor=PRIMARY_BLUE, validate="key", validatecommand=vcmd)
    bt_entry = tk.Entry(proc_frame, width=11, font=(base_font, 10), relief="flat", highlightthickness=1, highlightbackground=BORDER_COLOR, highlightcolor=PRIMARY_BLUE, validate="key", validatecommand=vcmd)
    pr_entry = tk.Entry(proc_frame, width=11, font=(base_font, 10), relief="flat", highlightthickness=1, highlightbackground=BORDER_COLOR, highlightcolor=PRIMARY_BLUE, validate="key", validatecommand=vcmd)
    pid_entry.grid(row=row + 1, column=0, padx=4, pady=4)
    at_entry.grid(row=row + 1, column=1, padx=4, pady=4)
    bt_entry.grid(row=row + 1, column=2, padx=4, pady=4)
    pr_entry.grid(row=row + 1, column=3, padx=4, pady=4)
    # priority inputs are shown/hidden by update_process_rows depending on selected algorithm
    pid_entries.append(pid_entry); at_entries.append(at_entry); bt_entries.append(bt_entry); pr_entries.append(pr_entry)

Tooltip(proc_frame.grid_slaves(row=0, column=1)[0], "Arrival time: non-negative integer")
Tooltip(proc_frame.grid_slaves(row=0, column=2)[0], "Burst time: positive integer")
Tooltip(priority_header, "Priority: lower number means higher priority")

pid_entries[0].insert(0, '1'); at_entries[0].insert(0, '0'); bt_entries[0].insert(0, '7')
pid_entries[1].insert(0, '2'); at_entries[1].insert(0, '0'); bt_entries[1].insert(0, '3')
pid_entries[2].insert(0, '3'); at_entries[2].insert(0, '0'); bt_entries[2].insert(0, '5')
pid_entries[3].insert(0, '4'); at_entries[3].insert(0, '1'); bt_entries[3].insert(0, '4')
pid_entries[4].insert(0, '5'); at_entries[4].insert(0, '2'); bt_entries[4].insert(0, '6')
pr_entries[0].insert(0, '2'); pr_entries[1].insert(0, '1'); pr_entries[2].insert(0, '3'); pr_entries[3].insert(0, '0'); pr_entries[4].insert(0, '4')


def update_process_rows(event=None):
    selected = int(process_count_var.get())
    show_priority = algorithm_var.get() in ("PRIORITY", "PRIORITY_AGING", "AI Scheduler")
    if show_priority:
        priority_header.grid(row=0, column=3, padx=4, pady=(0, 6))
    else:
        priority_header.grid_remove()
    for i in range(5):
        core_widgets = (pid_entries[i], at_entries[i], bt_entries[i])
        if i < selected:
            for widget in core_widgets:
                widget.grid()
            if show_priority:
                pr_entries[i].grid(row=i + 1, column=3, padx=4, pady=4)
            else:
                pr_entries[i].grid_remove()
        else:
            for widget in (pid_entries[i], at_entries[i], bt_entries[i], pr_entries[i]):
                widget.grid_remove()


def on_algorithm_change(event=None):
    sel = algorithm_var.get()
    if sel == "RR":
        quantum_label.config(text="Quantum")
        quantum_label.grid(); quantum_entry.grid()
    elif sel == "PRIORITY_AGING":
        quantum_label.config(text="Aging")
        quantum_label.grid(); quantum_entry.grid()
    else:
        quantum_label.grid_remove(); quantum_entry.grid_remove()
    update_process_rows()
    if "metrics_tree" in globals():
        if sel in ('PRIORITY', 'PRIORITY_AGING', 'AI Scheduler'):
            metrics_tree.configure(displaycolumns=("PID", "AT", "BT", "PR", "CT", "TAT", "WT"))
        else:
            metrics_tree.configure(displaycolumns=("PID", "AT", "BT", "CT", "TAT", "WT"))
    update_ai_mode_ui()


action_row = tk.Frame(inputs_card, bg=SIDEBAR_BG)
action_row.grid(row=3, column=0, columnspan=7, sticky="ew", padx=10, pady=(0, 8))
action_row.grid_columnconfigure((0, 1, 2), weight=1)

run_btn = tk.Button(action_row, text="Run Simulation", command=run_simulation, font=(base_font, 10, "bold"), padx=8, pady=8)
run_btn.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
load_sample_btn = tk.Button(action_row, text="Load Sample Data", command=load_sample_data, font=(base_font, 9), padx=8, pady=6)
load_sample_btn.grid(row=1, column=0, sticky="ew", padx=(0, 4))
clear_inputs_btn = tk.Button(action_row, text="Clear Inputs", command=clear_inputs, font=(base_font, 9), padx=8, pady=6)
clear_inputs_btn.grid(row=1, column=1, sticky="ew", padx=4)
gantt_btn = tk.Button(action_row, text="Show Gantt Chart", command=show_chart, font=(base_font, 9), padx=8, pady=6)
gantt_btn.grid(row=1, column=2, sticky="ew", padx=(4, 0))
apply_button_hover(run_btn, primary=True)
for btn in (load_sample_btn, clear_inputs_btn, gantt_btn):
    apply_button_hover(btn, primary=False)

ai_panel = tk.Frame(inputs_card, bg=SIDEBAR_BG, highlightthickness=1, highlightbackground=BORDER_COLOR)
ai_panel.grid(row=4, column=0, columnspan=7, sticky="ew", padx=10, pady=(6, 10))
tk.Label(ai_panel, text="AI Controls", bg=SIDEBAR_BG, fg=CARD_TEXT, font=(base_font, 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
model_status_val = tk.Label(ai_panel, text=get_model_status_text(), bg=SIDEBAR_BG, fg=MUTED_TEXT, font=(base_font, 9))
model_status_val.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)
ai_mode_badge = tk.Label(ai_panel, text=" AI MODE ", bg="#e2e8f0", fg=CARD_TEXT, font=(base_font, 8, "bold"), padx=8, pady=3)
ai_mode_badge.grid(row=2, column=0, sticky="w", padx=10, pady=(6, 4))
sim_status_val = ttk.Label(ai_panel, text="Simulation Status: Ready", style='StatusReady.TLabel')
sim_status_val.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))
ai_mode_val = tk.Label(ai_panel, text='AI Mode: Off', bg=SIDEBAR_BG, fg=MUTED_TEXT, font=(base_font, 9))
ai_mode_val.grid(row=4, column=0, columnspan=2, sticky='w', padx=10)
ai_status_val = tk.Label(ai_panel, text='AI Features: Inactive', bg=SIDEBAR_BG, fg=MUTED_TEXT, font=(base_font, 9))
ai_status_val.grid(row=5, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 8))
train_ai_btn = tk.Button(ai_panel, text='Train AI Model', command=on_train_ai_model_click, font=(base_font, 9), padx=8, pady=5)
train_ai_btn.grid(row=6, column=0, sticky='ew', padx=(10, 5), pady=(0, 10))
reset_ai_btn = tk.Button(ai_panel, text='Reset AI Model', command=on_reset_ai_model_click, font=(base_font, 9), padx=8, pady=5)
reset_ai_btn.grid(row=6, column=1, sticky='ew', padx=(5, 10), pady=(0, 10))
apply_button_hover(train_ai_btn, primary=False)
apply_button_hover(reset_ai_btn, primary=False)

status_bar = tk.Frame(dashboard, bg=BASE_BG)
status_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 2))
status_bar.grid_columnconfigure(2, weight=1)
status_badge = tk.Label(status_bar, text=" READY ", bg="#475569", fg="white", font=(base_font, 9, "bold"), padx=10, pady=4)
status_badge.grid(row=0, column=0, sticky="w")
tk.Label(status_bar, text="Dashboard Summary", bg=BASE_BG, fg=MUTED_TEXT, font=(base_font, 11, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0))

cards_frame = tk.Frame(dashboard, bg=BASE_BG)
cards_frame.grid(row=1, column=0, sticky="ew", padx=6)
cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

status_card = make_card(cards_frame, "Status", 0, 0)
status_value = tk.Label(status_card, text="READY", bg=CARD_BG, fg=SUCCESS_GREEN, font=(base_font, 16, "bold"))
status_value.pack(anchor="w", padx=14, pady=(0, 12))

wt_card = make_card(cards_frame, "Average Waiting Time", 0, 1)
avg_wt_val = tk.Label(wt_card, text="-", bg=CARD_BG, fg=CARD_TEXT, font=(base_font, 18, "bold"))
avg_wt_val.pack(anchor="w", padx=14, pady=(0, 12))

tat_card = make_card(cards_frame, "Average Turnaround Time", 0, 2)
avg_tat_val = tk.Label(tat_card, text="-", bg=CARD_BG, fg=CARD_TEXT, font=(base_font, 18, "bold"))
avg_tat_val.pack(anchor="w", padx=14, pady=(0, 12))

suggested_frame = make_card(cards_frame, "Recommended Algorithm", 0, 3)
suggested_val = tk.Label(suggested_frame, text='Recommended: run simulation to generate a suggestion', bg=CARD_BG, fg=CARD_TEXT, wraplength=280, justify="left", font=(base_font, 11, "bold"))
suggested_val.pack(anchor="w", padx=14, pady=(0, 12))

bottom = tk.Frame(dashboard, bg=BASE_BG)
bottom.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 10))
bottom.grid_columnconfigure(0, weight=1)
bottom.grid_rowconfigure(0, weight=1)
bottom.grid_rowconfigure(1, weight=0)

metrics_section = tk.Frame(bottom, bg="#ffffff", highlightthickness=1, highlightbackground=BORDER_COLOR)
metrics_section.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
metrics_section.grid_columnconfigure(0, weight=1)
metrics_section.grid_rowconfigure(1, weight=1)
tk.Label(metrics_section, text="Per-process Metrics", bg="#ffffff", fg=CARD_TEXT, font=(base_font, 11, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

details = tk.Frame(bottom, bg=BASE_BG)
details.grid(row=1, column=0, sticky="ew")
details.grid_columnconfigure((0, 1), weight=1)
tk.Label(details, text="Detailed Views", bg=BASE_BG, fg=MUTED_TEXT, font=(base_font, 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(2, 8))
comparison_btn = tk.Button(details, text='Algorithm Comparison', command=show_comparison_window, font=(base_font, 9), padx=8, pady=7)
ai_log_btn = tk.Button(details, text='AI Explanation Log', command=show_ai_log_window, font=(base_font, 9), padx=8, pady=7)
view_chart_btn = tk.Button(details, text='Open Full Gantt Chart', command=show_chart, font=(base_font, 9), padx=8, pady=7)
comparison_btn.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=4)
ai_log_btn.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=4)
view_chart_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
for btn in (comparison_btn, ai_log_btn, view_chart_btn):
    apply_button_hover(btn, primary=False)

metrics_tree = ttk.Treeview(metrics_section, style="MetricsTree.Treeview", columns=("PID","AT","BT","PR","CT","TAT","WT"), show='headings', height=8)
for col in ("PID","AT","BT","PR","CT","TAT","WT"):
    metrics_tree.heading(col, text=col)
metrics_tree.column("PID", width=78, anchor='center')
metrics_tree.column("AT", width=78, anchor='center')
metrics_tree.column("BT", width=78, anchor='center')
metrics_tree.column("PR", width=62, anchor='center')
metrics_tree.column("CT", width=95, anchor='center')
metrics_tree.column("TAT", width=95, anchor='center')
metrics_tree.column("WT", width=95, anchor='center')
metrics_tree.configure(displaycolumns=("PID", "AT", "BT", "CT", "TAT", "WT"))
metrics_tree.tag_configure("odd", background="#f8fafc")
metrics_tree.tag_configure("even", background="#ffffff")
metrics_tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 10))
metrics_scroll = ttk.Scrollbar(metrics_section, orient='vertical', command=metrics_tree.yview)
metrics_scroll.grid(row=1, column=1, sticky='ns', padx=(0, 10), pady=(0, 10))
metrics_tree.configure(yscrollcommand=metrics_scroll.set)

comparison_tree = ttk.Treeview(root, style="DashboardTree.Treeview", columns=("ALGO", "WT", "TAT"), show='headings', height=0)
comparison_tree.heading("ALGO", text="Algorithm")
comparison_tree.heading("WT", text="Avg WT")
comparison_tree.heading("TAT", text="Avg TAT")
comparison_tree.column("ALGO", width=170, anchor='w')
comparison_tree.column("WT", width=90, anchor='center')
comparison_tree.column("TAT", width=90, anchor='center')
comparison_tree.tag_configure("best_algo", background="#dcfce7", foreground="#166534")
comparison_tree.tag_configure("odd", background="#f8fafc")
comparison_tree.tag_configure("even", background="#ffffff")

ai_log_text = tk.Text(root, height=0, wrap='word', font=(base_font, 10), borderwidth=0)
comparison_summary_val_text = "Insight: comparison summary will appear here after simulation."
last_run_process_rows = None
last_run_quantum = "2"

def menu_about():
    messagebox.showinfo("About Project", "CPU Scheduling Visualizer\nProfessional dashboard UI with AI-assisted insights.")


menubar = tk.Menu(root)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Load Sample", command=load_sample_data, accelerator="Ctrl+L")
file_menu.add_command(label="Clear Inputs", command=clear_inputs, accelerator="Ctrl+C")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.destroy)
menubar.add_cascade(label="File", menu=file_menu)

view_menu = tk.Menu(menubar, tearoff=0)
view_menu.add_command(label="Metrics", command=show_metrics_window)
view_menu.add_command(label="Comparison", command=show_comparison_window)
view_menu.add_command(label="AI Log", command=show_ai_log_window)
view_menu.add_command(label="Gantt Chart", command=show_chart)
menubar.add_cascade(label="View", menu=view_menu)

help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="About Project", command=menu_about)
menubar.add_cascade(label="Help", menu=help_menu)
root.config(menu=menubar)

root.bind_all("<Control-r>", lambda _e: run_simulation())
root.bind_all("<Control-l>", lambda _e: load_sample_data())
root.bind_all("<Control-c>", lambda _e: clear_inputs())

algorithm.bind('<<ComboboxSelected>>', on_algorithm_change)
process_count.bind("<<ComboboxSelected>>", update_process_rows)
algorithm_var.trace_add('write', lambda *a: on_algorithm_change())

update_process_rows()
update_model_status_ui()
on_algorithm_change()
set_simulation_status('Simulation Status: Ready', 'ready')
render_gantt_preview([])

root.mainloop()
