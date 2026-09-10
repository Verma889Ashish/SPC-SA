

import os
import itertools
import math
import time
import random
from collections import defaultdict

import pandas as pd

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

MU = 0.4
G = 9.81
INF_PENALTY = 1e18
ARENA_SIZE = 100


# ============================================================
# 1) GENERATE ROBOTS
# ============================================================
def generate_robots(
    num_robots,
    num_skills,
    arena_size=ARENA_SIZE,
    min_skills_per_robot=8,
    max_skills_per_robot=15,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    if num_skills is None or num_skills <= 0:
        raise ValueError(f"num_skills must be >= 1, but got {num_skills}")

    skills = [f"Skill_{i+1}" for i in range(num_skills)]
    S = len(skills)

    min_k = max(1, min(min_skills_per_robot, S))
    max_k = max(1, min(max_skills_per_robot, S))
    if min_k > max_k:
        min_k, max_k = max_k, min_k

    robots = {}
    for i in range(num_robots):
        rid = f"Robot_{i+1}"
        velocity = random.uniform(0.5, 2.0)
        k = random.randint(min_k, max_k)
        robot_skills = random.sample(skills, k=k)
        robots[rid] = {
            "skills": robot_skills,
            "location": (random.randint(0, arena_size), random.randint(0, arena_size)),
            "velocity": velocity,
        }

    return robots, skills


# ============================================================
# 2) BUILD TASKS + PERFECT INIT MATCHING (use all robots once)
# ============================================================
def build_tasks_and_perfect_matching_use_all_robots(
    robots,
    skills,
    num_tasks,
    arena_size=ARENA_SIZE,
    min_skills_per_task=2,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    robot_ids = list(robots.keys())
    random.shuffle(robot_ids)

    tasks = []
    for _ in range(num_tasks):
        tasks.append({
            "skills": [],
            "location": (random.randint(0, arena_size), random.randint(0, arena_size)),
            "duration": random.uniform(2.0, 6.0),
            "priority": random.randint(1, 100),
        })

    matching = {f"Task_{t+1}": {} for t in range(num_tasks)}
    slot_req = {}
    slots = []

    # assign each robot to exactly one task skill slot
    for idx, rid in enumerate(robot_ids):
        t = idx % num_tasks
        used = set(tasks[t]["skills"])

        chosen = None
        for _ in range(10):
            cand = random.choice(robots[rid]["skills"])
            if cand not in used:
                chosen = cand
                break

        if chosen is None:
            cand = random.choice(robots[rid]["skills"])
            for shift in range(1, num_tasks + 1):
                tt = (t + shift) % num_tasks
                if cand not in set(tasks[tt]["skills"]):
                    t = tt
                    chosen = cand
                    break
            if chosen is None:
                chosen = cand

        tasks[t]["skills"].append(chosen)
        matching[f"Task_{t+1}"][chosen] = rid
        slot_req[(t, chosen)] = chosen
        slots.append((t, chosen, chosen))

    def cnt(ti): return len(tasks[ti]["skills"])

    # enforce each task has at least min_skills_per_task
    changed = True
    while changed:
        changed = False
        for t in range(num_tasks):
            if cnt(t) < min_skills_per_task:
                donor = None
                for d in range(num_tasks):
                    if cnt(d) > min_skills_per_task:
                        donor = d
                        break
                if donor is None:
                    break

                moved_skill = tasks[donor]["skills"].pop()
                if moved_skill in set(tasks[t]["skills"]):
                    tasks[donor]["skills"].append(moved_skill)
                    continue

                tasks[t]["skills"].append(moved_skill)

                rid = matching[f"Task_{donor+1}"].pop(moved_skill)
                matching[f"Task_{t+1}"][moved_skill] = rid

                req = slot_req.pop((donor, moved_skill))
                slot_req[(t, moved_skill)] = req

                for i in range(len(slots)):
                    if slots[i][0] == donor and slots[i][1] == moved_skill:
                        slots[i] = (t, moved_skill, req)
                        break

                changed = True

    return tasks, matching, slot_req, slots


# ============================================================
# 3) DEADLINES (FIXED)
# ============================================================
def compute_deadlines(robots, tasks, seed=None, factor=1.0):
    """
    FIX: base_deadline should be computed AFTER scanning all robots (nearest_dist).
    Your previous code computed/assigned inside the robot loop by mistake.

    deadline = factor * (nearest_dist / slowest_v)
    """
    if seed is not None:
        random.seed(seed)

    slowest_v = max(1e-9, min(r["velocity"] for r in robots.values()))

    deadlines = {}
    for i, task in enumerate(tasks, start=1):
        tid = f"Task_{i}"
        tx, ty = task["location"]

        nearest_dist = float("inf")
        for r in robots.values():
            rx, ry = r["location"]
            nearest_dist = min(nearest_dist, math.hypot(tx - rx, ty - ry))

        base_deadline = nearest_dist / slowest_v
        deadlines[tid] = factor * base_deadline
        # deadlines[tid] = random.uniform(base_deadline, 10*base_deadline)

    return deadlines



def build_fast_structures(robots, tasks):
    robot_ids = list(robots.keys())
    rid2i = {rid: i for i, rid in enumerate(robot_ids)}
    R = len(robot_ids)
    Tn = len(tasks)

    robot_skillset = {rid: set(info["skills"]) for rid, info in robots.items()}

    travel_t = [[0.0] * Tn for _ in range(R)]
    for i, rid in enumerate(robot_ids):
        r = robots[rid]
        rx, ry = r["location"]
        v = max(1e-9, r["velocity"])
        for t in range(Tn):
            tx, ty = tasks[t]["location"]
            dist = math.hypot(tx - rx, ty - ry)
            travel_t[i][t] = dist / v

    return robot_ids, rid2i, robot_skillset, travel_t


# ============================================================
# 5) PENALTY (NON-NEGATIVE GUARANTEE)
# ============================================================
def task_penalty_fast(t, tasks, task_assignment, deadlines, rid2i, travel_t):
    tid = f"Task_{t+1}"
    task = tasks[t]
    p = task["priority"]

    longest = 0.0
    for _, rid in task_assignment.items():
        if rid == "unfulfilled":
            return p * INF_PENALTY
        ri = rid2i[rid]
        longest = max(longest, travel_t[ri][t])

    omega = longest + task["duration"]
    tard = max(0.0, omega - deadlines[tid])  # MUST be nonnegative
    return p * tard


def calculate_penalty_all(tasks, matching, deadlines, rid2i, travel_t):
    total = 0.0
    for t in range(len(tasks)):
        tid = f"Task_{t+1}"
        total += task_penalty_fast(t, tasks, matching[tid], deadlines, rid2i, travel_t)

    # hard safety: penalty should never be negative
    if total < 0:
        raise ValueError(f"Penalty went negative (BUG): {total}")
    return total


# ============================================================
# 6) UNIQUE MOVED ROBOTS
# ============================================================
def count_unique_robots_moved(initial_matching, final_matching):
    init_pos = {}
    fin_pos = {}

    for tid, mp in initial_matching.items():
        for skill_key, rid in mp.items():
            init_pos[rid] = (tid, skill_key)

    for tid, mp in final_matching.items():
        for skill_key, rid in mp.items():
            fin_pos[rid] = (tid, skill_key)

    moved = set()
    for rid, p0 in init_pos.items():
        p1 = fin_pos.get(rid, None)
        if p1 is not None and p1 != p0:
            moved.add(rid)

    return len(moved), moved


# ============================================================
# 7) CF-HMRTA 
# ============================================================
def CF_HMRTA(robots, tasks, matching, deadlines, *, rid2i=None, use_priority=False):
    total_switches = 0
    CF_star = {tid: dict(mp) for tid, mp in matching.items()}

    pruned_G = defaultdict(set)
    for task_map in CF_star.values():
        for s, r in task_map.items():
            if r != "unfulfilled":
                pruned_G[s].add(r)

    Z = True
    I = 1
    m = len(tasks)

    while Z:
        I += 1

        task_penalties = []
        for task_id in CF_star:
            penalty = 0.0

            task_index = int(task_id.split("_")[1]) - 1
            task_location = tasks[task_index]["location"]
            max_time = deadlines[task_id]

            p = tasks[task_index].get("priority", 1) if use_priority else 1

            for skill, robot_id in CF_star[task_id].items():
                if robot_id == "unfulfilled":
                    penalty += max_time
                    continue

                robot = robots[robot_id]
                distance = math.hypot(
                    task_location[0] - robot["location"][0],
                    task_location[1] - robot["location"][1]
                )
                travel_time = distance / max(1e-9, robot["velocity"])
                if travel_time > max_time:
                    penalty += (travel_time - max_time)

            penalty *= p
            task_penalties.append((task_id, penalty))

        task_penalties.sort(key=lambda x: x[1], reverse=True)

        for task_id, _ in task_penalties:
            max_distance = -math.inf
            ri = None
            ssm = None

            task_index = int(task_id.split("_")[1]) - 1
            task_location = tasks[task_index]["location"]

            for skill, robot_id in CF_star[task_id].items():
                if robot_id == "unfulfilled":
                    continue

                robot = robots[robot_id]
                robot_location = robot["location"]
                distance = math.hypot(task_location[0] - robot_location[0], task_location[1] - robot_location[1])
                if distance > max_distance:
                    max_distance = distance
                    ri = robot_id
                    ssm = skill

            if ri is None:
                continue

            for ssn in pruned_G:
                if ri in pruned_G[ssn]:
                    for tj_id in CF_star:
                        if tj_id == task_id:
                            continue
                        if ssn in CF_star[tj_id]:
                            rj = CF_star[tj_id][ssn]
                            if rj == "unfulfilled":
                                continue

                            if ssm in robots[rj]["skills"] and ssn in robots[ri]["skills"]:
                                ti_index = task_index
                                tj_index = int(tj_id.split("_")[1]) - 1

                                ti_location = tasks[ti_index]["location"]
                                tj_location = tasks[tj_index]["location"]
                                ri_location = robots[ri]["location"]
                                rj_location = robots[rj]["location"]

                                cost_ri_ti = math.hypot(ti_location[0] - ri_location[0], ti_location[1] - ri_location[1]) / max(1e-9, robots[ri]["velocity"])
                                cost_rj_tj = math.hypot(tj_location[0] - rj_location[0], tj_location[1] - rj_location[1]) / max(1e-9, robots[rj]["velocity"])

                                cost_ri_tj = math.hypot(tj_location[0] - ri_location[0], tj_location[1] - ri_location[1]) / max(1e-9, robots[ri]["velocity"])
                                cost_rj_ti = math.hypot(ti_location[0] - rj_location[0], ti_location[1] - rj_location[1]) / max(1e-9, robots[rj]["velocity"])

                                if (cost_ri_tj < cost_ri_ti) and (cost_rj_ti < cost_rj_tj):
                                    CF_star[task_id][ssm] = rj
                                    CF_star[tj_id][ssn] = ri
                                    total_switches += 1

                                    pruned_G[ssn].remove(ri)
                                    pruned_G[ssn].add(rj)
                                    pruned_G[ssm].remove(rj)
                                    pruned_G[ssm].add(ri)
                                    break

            if I == m:
                Z = False
                break

    return CF_star, total_switches



def simulated_annealing_kbest_2swap_3cycle(
    tasks, matching, slots, slot_req, deadlines,
    robot_skillset, rid2i, travel_t,
    iters_per_temp=3500, max_temps=80, alpha=0.985, Tmin=1e-4,
    seed=None,
    p_guided=0.55,
    p_3cycle=0.40,
    K=25,
    target_accept=0.60,
    reset_to_best_each_temp=True
):
    if seed is not None:
        random.seed(seed)

    Tn = len(tasks)
    curr = {tid: dict(skmap) for tid, skmap in matching.items()}
    flat = [(t, f"Task_{t+1}", slot_key) for (t, slot_key, _) in slots]
    req_of = {(t, slot_key): slot_req[(t, slot_key)] for (t, slot_key, _) in slots}

    # ---------- init per-task + curr_cost ----------
    per_task = [0.0] * Tn
    for t in range(Tn):
        tid = f"Task_{t+1}"
        per_task[t] = task_penalty_fast(t, tasks, curr[tid], deadlines, rid2i, travel_t)
        if not math.isfinite(per_task[t]) or per_task[t] < 0:
            raise ValueError(f"Non-finite/negative task penalty at init: t={t} val={per_task[t]}")
    curr_cost = sum(per_task)

    best = {tid: dict(skmap) for tid, skmap in curr.items()}
    best_cost = curr_cost

    def pick_bad_slot():
        worst = None
        worst_val = -1.0
        for _ in range(30):
            t, tid, k = random.choice(flat)
            rid = curr[tid][k]
            val = travel_t[rid2i[rid]][t]
            if val > worst_val:
                worst_val = val
                worst = (t, tid, k)
        return worst

    # ---------- auto temperature init (unchanged) ----------
    pos_deltas = []
    for _ in range(400):
        t1, tid1, k1 = random.choice(flat)
        t2, tid2, k2 = random.choice(flat)
        if t1 == t2 and k1 == k2:
            continue
        r1 = curr[tid1][k1]
        r2 = curr[tid2][k2]
        if r1 == r2:
            continue
        req1 = req_of[(t1, k1)]
        req2 = req_of[(t2, k2)]
        if (req1 not in robot_skillset[r2]) or (req2 not in robot_skillset[r1]):
            continue

        old1, old2 = per_task[t1], per_task[t2]
        curr[tid1][k1], curr[tid2][k2] = r2, r1
        new1 = task_penalty_fast(t1, tasks, curr[tid1], deadlines, rid2i, travel_t)
        new2 = task_penalty_fast(t2, tasks, curr[tid2], deadlines, rid2i, travel_t)
        curr[tid1][k1], curr[tid2][k2] = r1, r2

        delta = (new1 + new2) - (old1 + old2)
        if delta > 0:
            pos_deltas.append(delta)

    if pos_deltas:
        pos_deltas.sort()
        med = pos_deltas[len(pos_deltas)//2]
        T = -med / math.log(target_accept)
        T = max(T, 1e-6)
    else:
        T = 1.0

    accepted_moves = 0
    robot_reassignments = 0
    temps_used = 0

  
    def eval_2swap(t1, tid1, k1, t2, tid2, k2):
        r1 = curr[tid1][k1]
        r2 = curr[tid2][k2]
        if r1 == r2:
            return None
        req1 = req_of[(t1, k1)]
        req2 = req_of[(t2, k2)]
        if (req1 not in robot_skillset[r2]) or (req2 not in robot_skillset[r1]):
            return None

        old1, old2 = per_task[t1], per_task[t2]
        curr[tid1][k1], curr[tid2][k2] = r2, r1
        new1 = task_penalty_fast(t1, tasks, curr[tid1], deadlines, rid2i, travel_t)
        new2 = task_penalty_fast(t2, tasks, curr[tid2], deadlines, rid2i, travel_t)
        curr[tid1][k1], curr[tid2][k2] = r1, r2

        if (not math.isfinite(new1)) or (not math.isfinite(new2)) or new1 < 0 or new2 < 0:
            return None

        
        new_cost = curr_cost - old1 - old2 + new1 + new2
        return ("2swap", (t1, tid1, k1, t2, tid2, k2), (new1, new2), new_cost)

    def eval_3cycle(a, b, c):
        (t1, tid1, k1) = a
        (t2, tid2, k2) = b
        (t3, tid3, k3) = c

        r1 = curr[tid1][k1]
        r2 = curr[tid2][k2]
        r3 = curr[tid3][k3]
        if len({r1, r2, r3}) < 3:
            return None

        req1 = req_of[(t1, k1)]
        req2 = req_of[(t2, k2)]
        req3 = req_of[(t3, k3)]
        if (req2 not in robot_skillset[r1]) or (req3 not in robot_skillset[r2]) or (req1 not in robot_skillset[r3]):
            return None

        old1, old2, old3 = per_task[t1], per_task[t2], per_task[t3]

        curr[tid1][k1] = r3
        curr[tid2][k2] = r1
        curr[tid3][k3] = r2

        new1 = task_penalty_fast(t1, tasks, curr[tid1], deadlines, rid2i, travel_t)
        new2 = task_penalty_fast(t2, tasks, curr[tid2], deadlines, rid2i, travel_t)
        new3 = task_penalty_fast(t3, tasks, curr[tid3], deadlines, rid2i, travel_t)

        curr[tid1][k1] = r1
        curr[tid2][k2] = r2
        curr[tid3][k3] = r3

        if (not math.isfinite(new1)) or (not math.isfinite(new2)) or (not math.isfinite(new3)) or new1 < 0 or new2 < 0 or new3 < 0:
            return None

        new_cost = curr_cost - old1 - old2 - old3 + new1 + new2 + new3
        return ("3cycle", (a, b, c), (new1, new2, new3), new_cost)

    while T > Tmin and temps_used < max_temps:
        temps_used += 1

        for _ in range(iters_per_temp):
            if random.random() < p_guided:
                t1, tid1, k1 = pick_bad_slot()
            else:
                t1, tid1, k1 = random.choice(flat)

            best_prop = None
            best_new_cost = float("inf")

            for _k in range(K):
                if random.random() < p_3cycle:
                    a = (t1, tid1, k1)
                    b = random.choice(flat)
                    c = random.choice(flat)
                    if b == a or c == a or b == c:
                        continue
                    prop = eval_3cycle(a, b, c)
                else:
                    t2, tid2, k2 = random.choice(flat)
                    if t2 == t1 and k2 == k1:
                        continue
                    prop = eval_2swap(t1, tid1, k1, t2, tid2, k2)

                if prop is None:
                    continue

                new_cost = prop[3]
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_prop = prop

            if best_prop is None:
                continue

            delta = best_new_cost - curr_cost
            accept = (delta <= 0.0) or (random.random() < math.exp(-delta / max(1e-12, T)))
            if not accept:
                continue

         
            if best_prop[0] == "2swap":
                _, (t1, tid1, k1, t2, tid2, k2), (new1, new2), _ = best_prop
                r1 = curr[tid1][k1]
                r2 = curr[tid2][k2]
                curr[tid1][k1], curr[tid2][k2] = r2, r1
                per_task[t1], per_task[t2] = new1, new2
                accepted_moves += 1
                robot_reassignments += 2
            else:
                _, (a, b, c), (new1, new2, new3), _ = best_prop
                (t1, tid1, k1) = a
                (t2, tid2, k2) = b
                (t3, tid3, k3) = c
                r1 = curr[tid1][k1]
                r2 = curr[tid2][k2]
                r3 = curr[tid3][k3]

                curr[tid1][k1] = r3
                curr[tid2][k2] = r1
                curr[tid3][k3] = r2
                per_task[t1], per_task[t2], per_task[t3] = new1, new2, new3
                accepted_moves += 1
                robot_reassignments += 3

            # ✅ CRITICAL FIX: re-sync true cost to avoid drift (prevents negative)
            curr_cost = sum(per_task)

            if not math.isfinite(curr_cost) or curr_cost < 0:
                raise ValueError(f"SA curr_cost became invalid: {curr_cost}")

            if curr_cost < best_cost:
                best_cost = curr_cost
                best = {tid: dict(skmap) for tid, skmap in curr.items()}

        # optional reset
        if reset_to_best_each_temp and curr_cost > best_cost:
            curr = {tid: dict(skmap) for tid, skmap in best.items()}
            for t in range(Tn):
                tid = f"Task_{t+1}"
                per_task[t] = task_penalty_fast(t, tasks, curr[tid], deadlines, rid2i, travel_t)
                if not math.isfinite(per_task[t]) or per_task[t] < 0:
                    raise ValueError(f"Non-finite/negative task penalty during reset: t={t} val={per_task[t]}")
            curr_cost = sum(per_task)

        T *= alpha

    return best, best_cost, accepted_moves, robot_reassignments, temps_used



# ============================================================
# 9) BRUTE FORCE (CORRECT + PRUNED)
# ============================================================
# def brute_force_method(tasks, robots, deadlines, rid2i, travel_t, *, show_progress=True):
#     """
#     Brute-force assignment aligned with your representation.

#     NOTE: tasks[t]["skills"] entries are treated as slot_keys.
#     If you later use "Skill_k__RRobot_x", req_skill extraction still works.
#     """

#     robot_ids = list(robots.keys())

#     # Build list of required slots: (task_id, slot_key, req_skill)
#     skill_requirements = []
#     for t_idx, task in enumerate(tasks):
#         tid = f"Task_{t_idx+1}"
#         for slot_key in task["skills"]:
#             req_skill = slot_key.split("__R", 1)[0]
#             skill_requirements.append((tid, slot_key, req_skill))

#     S = len(skill_requirements)
#     R = len(robot_ids)
#     if S > R:
#         raise ValueError(f"Brute-force impossible: required slots={S} > robots={R}")

#     # Feasible robot sets per slot (prune early)
#     feasible_sets = []
#     for (_, _, req_skill) in skill_requirements:
#         feas = {rid for rid in robot_ids if req_skill in robots[rid]["skills"]}
#         if not feas:
#             return None, float("inf")
#         feasible_sets.append(feas)

#     total_perm = math.perm(R, S)  # upper bound, not exact after pruning

#     best_assignment = None
#     min_penalty = float("inf")

#     pbar = None
#     if show_progress and tqdm is not None:
#         pbar = tqdm(total=total_perm, desc="BruteForce", ncols=100)

#     for robot_tuple in itertools.permutations(robot_ids, S):
#         # permutations already guarantee uniqueness -> no assigned_robots set needed

#         # feasibility check
#         ok = True
#         for i in range(S):
#             if robot_tuple[i] not in feasible_sets[i]:
#                 ok = False
#                 break

#         if pbar is not None:
#             pbar.update(1)

#         if not ok:
#             continue

#         assignment = defaultdict(dict)
#         for i, (task_id, slot_key, _) in enumerate(skill_requirements):
#             assignment[task_id][slot_key] = robot_tuple[i]

#         penalty = calculate_penalty_all(tasks, assignment, deadlines, rid2i, travel_t)
#         if penalty < min_penalty:
#             min_penalty = penalty
#             best_assignment = assignment

#     if pbar is not None:
#         pbar.close()

#     return best_assignment, min_penalty


# ============================================================
# 10) SAVE PROGRESS
# ============================================================
def save_progress(df, base_dir, filename_no_ext):
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, filename_no_ext + ".csv")
    df.to_csv(csv_path, index=False)

    xlsx_path = None
    try:
        import openpyxl  # noqa: F401
        xlsx_path = os.path.join(base_dir, filename_no_ext + ".xlsx")
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass

    return csv_path, xlsx_path


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    RESULTS_DIR = r"C:\Users\zu155765\OneDrive - LUT University\Ashish Verma"

    N_RUNS = 50
    BASE_SEED = 20260120

    robots_list = [500, 750, 1000, 1250, 1500, 1750, 2000]
    tasks_list = [25, 50, 100, 200, 400]
    num_skills = 80

    # SA settings (defaults; will be overwritten per m below)
    SA_ALPHA = 0.975
    SA_K = 10
    SA_P_GUIDED = 0.75
    SA_P_3CYCLE = 0.10
    SA_TARGET_ACCEPT = 0.35

    def pct_reduce(base, new):
        return 0.0 if base <= 0 else ((base - new) / base) * 100.0

    for num_tasks in tasks_list:

        # ----------------------------
        # SA scaling by m (= num_tasks)
        # ----------------------------
        m = num_tasks
        if m <= 50:
            SA_MAX_TEMPS = 30
            SA_ITERS_PER_TEMP = 16
        elif m <= 150:
            SA_MAX_TEMPS = 40
            SA_ITERS_PER_TEMP = 20
        elif m <= 300:
            SA_MAX_TEMPS = 60
            SA_ITERS_PER_TEMP = 30
        else:  # 301..400
            SA_MAX_TEMPS = 90
            SA_ITERS_PER_TEMP = 50
    for num_tasks in tasks_list:
        for num_robots in robots_list:

            if num_robots < (num_tasks * 2):
                print(f"\n[SKIP CONFIG] robots={num_robots}, tasks={num_tasks} because 2*tasks={2*num_tasks} > robots")
                continue

            print(f"\n===============================")
            print(f"CONFIG: robots={num_robots}, tasks={num_tasks}, skills={num_skills}")
            print(f"===============================")

            results = []
            filename = f"CASE1_penalty_only_R{num_robots}_T{num_tasks}_S{num_skills}_runs{N_RUNS}"

            try:
                for run in range(1, N_RUNS + 1):
                    seed = BASE_SEED + run

                    robots, skills = generate_robots(
                        num_robots, num_skills,
                        min_skills_per_robot=8, max_skills_per_robot=15,
                        seed=seed
                    )

                    tasks, init_matching, slot_req, slots = build_tasks_and_perfect_matching_use_all_robots(
                        robots, skills, num_tasks,
                        min_skills_per_task=2,
                        seed=seed
                    )

                    deadlines = compute_deadlines(robots, tasks, seed=seed, factor=2.0)
                    robot_ids, rid2i, robot_skillset, travel_t = build_fast_structures(robots, tasks)

                    # Ratio (baseline = init matching)
                    t0 = time.perf_counter()
                    ratio_match = init_matching
                    ratio_penalty = calculate_penalty_all(tasks, ratio_match, deadlines, rid2i, travel_t)
                    ratio_time = time.perf_counter() - t0

                    # CF-HMRTA
                    t1 = time.perf_counter()
                    cf_match, cf_switches = CF_HMRTA(robots, tasks, ratio_match, deadlines)
                    cf_penalty = calculate_penalty_all(tasks, cf_match, deadlines, rid2i, travel_t)
                    cf_time = time.perf_counter() - t1
                    cf_unique_moved, _ = count_unique_robots_moved(ratio_match, cf_match)

                    # SA
                    t2 = time.perf_counter()
                    sa_match, sa_penalty, sa_moves, sa_robot_reassign, sa_temps = simulated_annealing_kbest_2swap_3cycle(
                        tasks, ratio_match, slots, slot_req, deadlines,
                        robot_skillset, rid2i, travel_t,
                        iters_per_temp=SA_ITERS_PER_TEMP,
                        max_temps=SA_MAX_TEMPS,
                        alpha=SA_ALPHA,
                        seed=seed,
                        p_guided=SA_P_GUIDED,
                        p_3cycle=SA_P_3CYCLE,
                        K=SA_K,
                        target_accept=SA_TARGET_ACCEPT,
                        reset_to_best_each_temp=True
                    )
                    sa_time = time.perf_counter() - t2
                    sa_unique_moved, _ = count_unique_robots_moved(ratio_match, sa_match)

                    # Brute force (ONLY meaningful for very small cases)
                    # t3 = time.perf_counter()
                    # best_assignment, brute_force_penalty = brute_force_method(
                    #     tasks, robots, deadlines, rid2i, travel_t,
                    #     show_progress=False
                    # )
                    # brute_force_time = time.perf_counter() - t3

                    print(f"\nRUN {run} (seed={seed})")
                    print(f"  Ratio: time={ratio_time:.9f}s  penalty={ratio_penalty:.6e}")
                    print(f"  CF   : time={cf_time:.9f}s  penalty={cf_penalty:.6e}"
                          f"  switches={cf_switches}  unique_moved_final={cf_unique_moved}"
                          f"  improve={pct_reduce(ratio_penalty, cf_penalty):.2f}%")
                    print(f"  SA   : time={sa_time:.9f}s  penalty={sa_penalty:.6e}"
                          f"  accepted_moves={sa_moves}  robot_reassignments={sa_robot_reassign}"
                          f"  unique_moved_final={sa_unique_moved}  temps={sa_temps}"
                          f"  improve={pct_reduce(ratio_penalty, sa_penalty):.2f}%")
                    # print(f"  BF   : time={brute_force_time:.9f}s  penalty={brute_force_penalty:.6e}")

                    results.append({
                        "run": run,
                        "seed": seed,
                        "num_robots": num_robots,
                        "num_tasks": num_tasks,
                        "num_skills": num_skills,

                        "ratio_time_sec": ratio_time,
                        "ratio_penalty": ratio_penalty,

                        "cf_time_sec": cf_time,
                        "cf_penalty": cf_penalty,
                        "cf_switches": cf_switches,
                        "cf_unique_moved_final": cf_unique_moved,
                        "cf_improve_pct_vs_ratio": pct_reduce(ratio_penalty, cf_penalty),

                        "sa_time_sec": sa_time,
                        "sa_penalty": sa_penalty,
                        "sa_accepted_moves": sa_moves,
                        "sa_robot_reassignments": sa_robot_reassign,
                        "sa_unique_moved_final": sa_unique_moved,
                        "sa_temps_used": sa_temps,
                        "sa_improve_pct_vs_ratio": pct_reduce(ratio_penalty, sa_penalty),

                        "sa_iters_per_temp": SA_ITERS_PER_TEMP,
                        "sa_max_temps": SA_MAX_TEMPS,
                        "sa_K": SA_K,
                        "sa_alpha": SA_ALPHA,

                        # "brute_force_pen": brute_force_penalty,
                        # "brute_force_time": brute_force_time,
                    })

                    df = pd.DataFrame(results)
                    csv_path, xlsx_path = save_progress(df, RESULTS_DIR, filename)
                    print(f"  Saved progress: {csv_path}" + (f" | {xlsx_path}" if xlsx_path else ""))

            except KeyboardInterrupt:
                print("\nStopped by user (Ctrl+C). Saving partial config results...")
                df = pd.DataFrame(results)
                csv_path, xlsx_path = save_progress(df, RESULTS_DIR, filename)
                print(f"Partial saved: {csv_path}" + (f" | {xlsx_path}" if xlsx_path else ""))
                raise
