#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define AGING_DEFAULT_INTERVAL 2

#define MAX 100

int main(int argc, char *argv[])
{
    const char *algorithm = (argc > 1) ? argv[1] : "FCFS";

    FILE *fp = fopen("input.txt", "r");
    if(!fp) {
        fprintf(stderr, "Failed to open input.txt\n");
        return 1;
    }

    int n;
    if(fscanf(fp, "%d", &n) != 1) {
        fprintf(stderr, "Invalid input format (missing n)\n");
        fclose(fp);
        return 1;
    }

    int pid[MAX], at[MAX], bt[MAX], pr[MAX];

    char line[256];
    fgets(line, sizeof(line), fp); // consume end of first line after n

    for(int i=0;i<n;i++) {
        if(!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "Invalid input format on line %d\n", i + 2);
            fclose(fp);
            return 1;
        }

        // Supports both "pid at bt" and "pid at bt priority" input formats.
        int parsed = sscanf(line, "%d %d %d %d", &pid[i], &at[i], &bt[i], &pr[i]);
        if(parsed == 3) {
            pr[i] = 0;
        } else if(parsed != 4) {
            fprintf(stderr, "Invalid input format on line %d\n", i + 2);
            fclose(fp);
            return 1;
        }

        // Validate arrival time cannot be negative
        if(at[i] < 0) {
            fprintf(stderr, "Error on line %d: Arrival time cannot be negative, got %d\n", i + 2, at[i]);
            fclose(fp);
            return 1;
        }
        
        // Validate burst time must be greater than 0
        if(bt[i] <= 0) {
            fprintf(stderr, "Error on line %d: Burst time must be greater than 0, got %d\n", i + 2, bt[i]);
            fclose(fp);
            return 1;
        }
    }

    fclose(fp);

    FILE *out = fopen("output.txt", "w");
    if(!out) {
        fprintf(stderr, "Failed to open output.txt for writing\n");
        return 1;
    }

    int comp_time[MAX];
    for(int i=0;i<MAX;i++) comp_time[i] = -1;

    // We'll produce schedule as sequence of segments: pid start duration
    if(strcmp(algorithm, "FCFS") == 0) {
        int current = 0;
        // Sort by arrival time while preserving pid/burst
        int order[MAX];
        for(int i=0;i<n;i++) order[i] = i;
        // simple stable sort by at
        for(int i=0;i<n-1;i++) {
            for(int j=i+1;j<n;j++) {
                if(at[order[i]] > at[order[j]]) {
                    int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
                }
            }
        }

        for(int idx=0; idx<n; idx++) {
            int i = order[idx];
            int start = (current < at[i]) ? at[i] : current;
            int dur = bt[i];
            fprintf(out, "%d %d %d\n", pid[i], start, dur);
            current = start + dur;
            comp_time[i] = current;
        }
    }
    else if(strcmp(algorithm, "SJF") == 0) {
        int completed[MAX] = {0};
        int completed_count = 0;
        int current = 0;

        while(completed_count < n) {
            // find shortest job among arrived and not completed
            int best = -1;
            for(int i=0;i<n;i++) {
                if(!completed[i] && at[i] <= current) {
                    if(best == -1 || bt[i] < bt[best]) best = i;
                }
            }

            if(best == -1) {
                // no process has arrived yet; advance to next arrival
                int next = -1;
                for(int i=0;i<n;i++) if(!completed[i]) {
                    if(next == -1 || at[i] < at[next]) next = i;
                }
                current = at[next];
                continue;
            }

            int start = current;
            int dur = bt[best];
            fprintf(out, "%d %d %d\n", pid[best], start, dur);
            current += dur;
            comp_time[best] = current;
            completed[best] = 1;
            completed_count++;
        }
    }
    else if(strcmp(algorithm, "SRTF") == 0) {
        int remaining[MAX];
        int completed_count = 0;
        int time = 0;
        int last_pid = -1;
        int seg_start = 0;
        int seg_dur = 0;

        for(int i=0;i<n;i++) remaining[i] = bt[i];

        while(completed_count < n) {
            int best = -1;

            for(int i=0;i<n;i++) {
                if(remaining[i] <= 0 || at[i] > time) continue;

                if(best == -1 ||
                   remaining[i] < remaining[best] ||
                   (remaining[i] == remaining[best] && at[i] < at[best]) ||
                   (remaining[i] == remaining[best] && at[i] == at[best] && pid[i] < pid[best])) {
                    best = i;
                }
            }

            if(best == -1) {
                // CPU idle: flush current segment and jump to next arrival.
                if(seg_dur > 0) {
                    fprintf(out, "%d %d %d\n", last_pid, seg_start, seg_dur);
                    seg_dur = 0;
                    last_pid = -1;
                }

                int next = -1;
                for(int i=0;i<n;i++) {
                    if(remaining[i] > 0 && (next == -1 || at[i] < at[next])) next = i;
                }
                if(next == -1) break;
                time = at[next];
                continue;
            }

            if(last_pid != pid[best]) {
                if(seg_dur > 0) {
                    fprintf(out, "%d %d %d\n", last_pid, seg_start, seg_dur);
                }
                last_pid = pid[best];
                seg_start = time;
                seg_dur = 1;
            } else {
                seg_dur++;
            }

            remaining[best]--;
            time++;

            if(remaining[best] == 0) {
                comp_time[best] = time;
                completed_count++;
            }
        }

        if(seg_dur > 0) {
            fprintf(out, "%d %d %d\n", last_pid, seg_start, seg_dur);
        }
    }
    else if(strcmp(algorithm, "RR") == 0 || strcmp(algorithm, "R R") == 0) {
        int quantum = 2;
        if(argc > 2) quantum = atoi(argv[2]);
        if(quantum <= 0) quantum = 2;

        int remaining[MAX];
        for(int i=0;i<n;i++) remaining[i] = bt[i];

        int in_queue[MAX] = {0};
        int queue[MAX];
        int q_front = 0, q_rear = 0, q_size = 0;
        int completed_count = 0;

        int time = -1;
        for(int i=0;i<n;i++) {
            if(time == -1 || at[i] < time) time = at[i];
        }
        if(time < 0) time = 0;

        while(completed_count < n) {
            for(int i=0;i<n;i++) {
                if(remaining[i] > 0 && !in_queue[i] && at[i] <= time) {
                    queue[q_rear] = i;
                    q_rear = (q_rear + 1) % MAX;
                    q_size++;
                    in_queue[i] = 1;
                }
            }

            if(q_size == 0) {
                int next_time = -1;
                for(int i=0;i<n;i++) {
                    if(remaining[i] > 0 && at[i] > time && (next_time == -1 || at[i] < next_time)) {
                        next_time = at[i];
                    }
                }

                if(next_time == -1) break;
                time = next_time;
                continue;
            }

            int current = queue[q_front];
            q_front = (q_front + 1) % MAX;
            q_size--;
            in_queue[current] = 0;

            int use = (remaining[current] > quantum) ? quantum : remaining[current];
            fprintf(out, "%d %d %d\n", pid[current], time, use);
            time += use;
            remaining[current] -= use;

            // Arrivals during the executed time slice should be served before re-queueing current.
            for(int i=0;i<n;i++) {
                if(remaining[i] > 0 && !in_queue[i] && at[i] <= time) {
                    queue[q_rear] = i;
                    q_rear = (q_rear + 1) % MAX;
                    q_size++;
                    in_queue[i] = 1;
                }
            }

            if(remaining[current] > 0) {
                queue[q_rear] = current;
                q_rear = (q_rear + 1) % MAX;
                q_size++;
                in_queue[current] = 1;
            } else {
                comp_time[current] = time;
                completed_count++;
            }
        }
    }
    else if(strcmp(algorithm, "PRIORITY") == 0 || strcmp(algorithm, "PRIORITY_AGING") == 0) {
        int aging_enabled = (strcmp(algorithm, "PRIORITY_AGING") == 0);
        int aging_interval = AGING_DEFAULT_INTERVAL;
        if(aging_enabled) {
            if(argc > 2) aging_interval = atoi(argv[2]);
            if(aging_interval <= 0) aging_interval = AGING_DEFAULT_INTERVAL;
        }

        int remaining[MAX];
        int completed_count = 0;
        int time = 0;
        int last_pid = -1;
        int seg_start = 0;
        int seg_dur = 0;

        for(int i=0;i<n;i++) remaining[i] = bt[i];

        while(completed_count < n) {
            int best = -1;

            for(int i=0;i<n;i++) {
                if(remaining[i] <= 0 || at[i] > time) continue;

                int effective_priority = pr[i];
                if(aging_enabled) {
                    int executed = bt[i] - remaining[i];
                    int waited = time - at[i] - executed;
                    if(waited < 0) waited = 0;
                    // Lower numeric priority means better priority in this project.
                    // Apply a capped aging bonus so long-waiting jobs improve gradually.
                    int aging_bonus = waited / aging_interval;
                    if(aging_bonus > 2) aging_bonus = 2;
                    effective_priority -= aging_bonus;
                }

                if(best == -1) {
                    best = i;
                    continue;
                }

                int best_effective_priority = pr[best];
                if(aging_enabled) {
                    int best_executed = bt[best] - remaining[best];
                    int best_waited = time - at[best] - best_executed;
                    if(best_waited < 0) best_waited = 0;
                    int best_aging_bonus = best_waited / aging_interval;
                    if(best_aging_bonus > 2) best_aging_bonus = 2;
                    best_effective_priority -= best_aging_bonus;
                }

                if(effective_priority < best_effective_priority ||
                   (effective_priority == best_effective_priority && at[i] < at[best]) ||
                   (effective_priority == best_effective_priority && at[i] == at[best] && pid[i] < pid[best])) {
                    best = i;
                }
            }

            if(best == -1) {
                // CPU idle: flush current segment and jump to next arrival.
                if(seg_dur > 0) {
                    fprintf(out, "%d %d %d\n", last_pid, seg_start, seg_dur);
                    seg_dur = 0;
                    last_pid = -1;
                }

                int next = -1;
                for(int i=0;i<n;i++) {
                    if(remaining[i] > 0 && (next == -1 || at[i] < at[next])) next = i;
                }
                if(next == -1) break;
                time = at[next];
                continue;
            }

            if(last_pid != pid[best]) {
                if(seg_dur > 0) {
                    fprintf(out, "%d %d %d\n", last_pid, seg_start, seg_dur);
                }
                last_pid = pid[best];
                seg_start = time;
                seg_dur = 1;
            } else {
                seg_dur++;
            }

            remaining[best]--;
            time++;

            if(remaining[best] == 0) {
                comp_time[best] = time;
                completed_count++;
            }
        }

        if(seg_dur > 0) {
            fprintf(out, "%d %d %d\n", last_pid, seg_start, seg_dur);
        }
    }
    else {
        // unknown algorithm, default to FCFS
        int current = 0;
        for(int i=0;i<n;i++) {
            int start = (current < at[i]) ? at[i] : current;
            fprintf(out, "%d %d %d\n", pid[i], start, bt[i]);
            current = start + bt[i];
            comp_time[i] = current;
        }
    }

    fclose(out);

    // compute per-process completion, turnaround, waiting
    double total_tat = 0.0;
    double total_wt = 0.0;
    int tat[MAX], wt[MAX];

    for(int i=0;i<n;i++) {
        if(comp_time[i] < 0) {
            // process never scheduled (shouldn't happen), set completion as arrival
            comp_time[i] = at[i];
        }
        tat[i] = comp_time[i] - at[i];
        wt[i] = tat[i] - bt[i];
        
        // Ensure waiting time is never negative
        if(wt[i] < 0) {
            wt[i] = 0;
        }
        
        total_tat += tat[i];
        total_wt += wt[i];
    }

    double avg_tat = total_tat / n;
    double avg_wt = total_wt / n;

    // write detailed metrics
    FILE *metrics = fopen("metrics.txt", "w");
    if(metrics) {
        fprintf(metrics, "PID AT BT PR CT TAT WT\n");
        for(int i=0;i<n;i++) {
            fprintf(metrics, "%d %d %d %d %d %d %d\n", pid[i], at[i], bt[i], pr[i], comp_time[i], tat[i], wt[i]);
        }
        fprintf(metrics, "Average Turnaround: %.2f\n", avg_tat);
        fprintf(metrics, "Average Waiting: %.2f\n", avg_wt);
        fclose(metrics);
    }

    printf("Scheduling completed\n");
    printf("Average Turnaround: %.2f\n", avg_tat);
    printf("Average Waiting: %.2f\n", avg_wt);

    return 0;
}