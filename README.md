# CPU Scheduler & AI Scheduling Visualizer

An interactive CPU scheduling simulator and visualizer combining a C scheduling engine with a Q-learning AI scheduler and a Tkinter/Matplotlib desktop dashboard.

## Features

- **Classic CPU Scheduling Algorithms (C Engine — `scheduler.c`)**:
  - **FCFS** — First-Come, First-Served
  - **SJF** — Shortest Job First (non-preemptive)
  - **SRTF** — Shortest Remaining Time First (preemptive)
  - **RR** — Round Robin (configurable time quantum)
  - **PRIORITY** — Preemptive Priority Scheduling (lower numeric value = higher priority)
  - **PRIORITY_AGING** — Preemptive Priority Scheduling with starvation-preventing aging
- **Reinforcement Learning AI Scheduler (`ai_scheduler.py`, `train_ai.py`)**:
  - Uses a Q-learning agent that dynamically selects between `FCFS`, `SJF`, `PRIORITY`, and `AGING_PRIORITY` based on live ready-queue state features.
  - Includes a pre-trained Q-table (`q_table.json`, `q_table_meta.json`) trained across 20,000 curriculum episodes covering mixed priorities, starvation-prone workloads, priority inversion, and extreme burst outliers.
- **Interactive Desktop Dashboard (`visualizer.py`)**:
  - Visual Gantt chart preview and full-screen Gantt chart view
  - Per-process metrics table (Arrival Time, Burst Time, Priority, Completion Time, Turnaround Time, Waiting Time)
  - Side-by-side Algorithm Comparison view
  - AI Decision Explanation Log
  - Built-in Q-table training controls

## Project Structure

```text
Cpu_Scheduler_Project/
├── scheduler.c          # C implementation of classical CPU scheduling algorithms
├── ai_scheduler.py      # Q-learning AI scheduler and simulation engine
├── train_ai.py          # CLI & module for training the Q-table across workload profiles
├── visualizer.py        # Tkinter + Matplotlib GUI dashboard
├── q_table.json         # Pre-trained Q-learning state-action table
├── q_table_meta.json    # Training episode metadata
└── requirements.txt     # Python dependencies
```

## Prerequisites

- **Python 3.10+** (with `tkinter` included)
- **GCC** (required to compile `scheduler.c`; `visualizer.py` compiles it automatically on first run)

## Setup & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/anshp7313-design/ai-cpu-scheduler.git
   cd ai-cpu-scheduler
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   # source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Run the GUI Visualizer**
   ```bash
   python visualizer.py
   ```
   The visualizer automatically compiles `scheduler.c` into `scheduler.exe` (Windows) or `scheduler` (Linux/macOS) when needed.

4. **(Optional) Train the AI Model via CLI**
   ```bash
   python train_ai.py --episodes 5000 --alpha 0.20 --gamma 0.90
   ```

5. **(Optional) Run the C Scheduler directly**
   ```bash
   gcc scheduler.c -o scheduler
   ./scheduler FCFS
   ./scheduler RR 2
   ./scheduler PRIORITY_AGING 2
   ```
   Reads process rows from `input.txt` (`n` on line 1, followed by `pid at bt [priority]` lines) and writes `output.txt` and `metrics.txt`.
