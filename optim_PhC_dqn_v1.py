"""DQN optimization of a COMSOL PCSEL model.

This version keeps the COMSOL environment interface unchanged while adding:

1. Periodic RBF-style history snapshots rather than one file per step.
2. Resumable DQN checkpoints for Slurm/HPC runs.
3. Cumulative input.pt and objective.pt files for later analysis.
4. A detailed CSV training history and plain-text training log.
5. A compact partner-style training_summary.txt report.
6. All quantities needed to reproduce the BO-style plots later:
   Q progression, wavelength convergence, Pareto front, hypervolume,
   feasibility, parameter sensitivity/convergence/distributions, and
   optimized-versus-default comparisons.

The script intentionally does not generate plots during training.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import deque, namedtuple
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.envs.registration import register
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------
# User-editable run configuration
# -----------------------------------------------------------------------------

ENV_ID = "ComsolDQNGev1-v0"
ENV_ENTRY_POINT = "comsol_env_v1:ComsolEnv"

NUM_EPISODES = 200
MAX_EPISODE_STEPS = 25

# Save history/checkpoints after this many actual COMSOL evaluations.
# This is analogous to the RBF script's ``if iteration % 25 == 0`` block.
SAVE_INTERVAL_EVALUATIONS = 25

# Resume from checkpoints/latest_checkpoint.pt when it exists.
RESUME_IF_AVAILABLE = True

# Analysis definitions chosen to match the partner's BO reporting.
TARGET_WAVELENGTH_NM = 2200.0
WAVELENGTH_TOLERANCE_NM = 100.0
Q_GOAL = 1.0e9

# DQN hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 300
TARGET_UPDATE_EPISODES = 5
UPDATE_FREQ_EVALUATIONS = 4
MEMORY_CAPACITY = 5000
LEARNING_RATE = 2.5e-4
HIDDEN_SIZE = 128

# Reproducibility. COMSOL itself may still introduce numerical variation.
RANDOM_SEED = 7

# Output layout. The default creates one run folder in the Slurm working dir.
RUN_ROOT = Path(os.environ.get("PCSEL_RUN_DIR", "dqn_run"))
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
PLOT_DATA_DIR = RUN_ROOT / "plot_data"
TENSORBOARD_DIR = RUN_ROOT / "tensorboard"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"
BEST_REWARD_CHECKPOINT = CHECKPOINT_DIR / "best_reward_model.pt"
BEST_FEASIBLE_Q_CHECKPOINT = CHECKPOINT_DIR / "best_feasible_Q_model.pt"
INPUT_PT = PLOT_DATA_DIR / "input.pt"
OBJECTIVE_PT = PLOT_DATA_DIR / "objective.pt"
METRICS_PT = PLOT_DATA_DIR / "metrics.pt"
RUN_METADATA_JSON = PLOT_DATA_DIR / "run_metadata.json"
HISTORY_CSV = LOG_DIR / "training_history.csv"
TRAINING_LOG = LOG_DIR / "training.log"
SUMMARY_TXT = LOG_DIR / "training_summary.txt"


# -----------------------------------------------------------------------------
# Environment registration and utility helpers
# -----------------------------------------------------------------------------

try:
    register(
        id=ENV_ID,
        entry_point=ENV_ENTRY_POINT,
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=200.0,
    )
except Exception:
    # Gym raises if the environment was already registered in this process.
    pass


def setup_directories() -> None:
    for directory in (RUN_ROOT, CHECKPOINT_DIR, LOG_DIR, PLOT_DATA_DIR, TENSORBOARD_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("pcsel_dqn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(TRAINING_LOG, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy_1d(value: Any) -> np.ndarray:
    """Convert an environment state to a flat float64 NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def extract_info_value(info: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in info:
            return info[key]
    return default


def extract_metrics(info: Dict[str, Any]) -> Tuple[float, float, int]:
    """Read Q, wavelength and mode index without changing the environment."""
    q_factor = finite_float(
        extract_info_value(info, ("Qfactor", "Q_factor", "Q", "qfactor"), float("nan"))
    )
    wavelength_nm = finite_float(
        extract_info_value(
            info,
            ("wavelength_nm", "wavelength", "lambda_nm", "lam_nm", "lambda"),
            float("nan"),
        )
    )
    mode_raw = extract_info_value(info, ("mode_index", "mode", "selected_mode"), -1)
    try:
        mode_index = int(mode_raw)
    except (TypeError, ValueError):
        mode_index = -1
    return q_factor, wavelength_nm, mode_index


def wavelength_metrics(wavelength_nm: float) -> Tuple[float, float, bool]:
    if not math.isfinite(wavelength_nm):
        return float("nan"), float("nan"), False
    error_nm = abs(wavelength_nm - TARGET_WAVELENGTH_NM)
    match_score = 1.0 - error_nm / TARGET_WAVELENGTH_NM
    feasible = error_nm <= WAVELENGTH_TOLERANCE_NM
    return error_nm, match_score, feasible


def atomic_torch_save(payload: Any, destination: Path) -> None:
    """Write a torch file atomically so a Slurm interruption does not corrupt it."""
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, destination)


def atomic_text_write(text: str, destination: Path) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, destination)


def reset_env(env: gym.Env) -> np.ndarray:
    result = env.reset()
    # Compatible with both old Gym and newer reset() -> (obs, info).
    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    return to_numpy_1d(result)


def step_env(env: gym.Env, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise TypeError("env.step(action) must return a tuple")

    if len(result) == 4:
        obs, reward, done, info = result
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        raise ValueError(f"Unexpected env.step return length: {len(result)}")

    return to_numpy_1d(obs), float(reward), bool(done), dict(info or {})


# -----------------------------------------------------------------------------
# DQN model and replay memory
# -----------------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> Dict[str, Any]:
        return {"capacity": self.memory.maxlen, "memory": list(self.memory)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        capacity = int(state.get("capacity", MEMORY_CAPACITY))
        self.memory = deque(state.get("memory", []), maxlen=capacity)


class Net(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=device, dtype=torch.float32)
        x = x.view(-1, state_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class BestRecord:
    value: float = -float("inf")
    global_step: int = -1
    episode: int = -1
    step_in_episode: int = -1
    state: Optional[List[float]] = None
    info: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Initialize runtime
# -----------------------------------------------------------------------------

setup_directories()
logger = setup_logging()
set_random_seeds(RANDOM_SEED)

logger.info("Starting COMSOL PCSEL DQN run")
logger.info("Run directory: %s", RUN_ROOT.resolve())
logger.info("Python: %s", sys.version.replace("\n", " "))
logger.info("PyTorch: %s", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

env = gym.make(ENV_ID).unwrapped
n_actions = int(env.action_space.n)
state_dim = int(env.observation_space.shape[0])
param_names = list(getattr(env, "param_names", [f"param_{i}" for i in range(state_dim)]))
if len(param_names) != state_dim:
    logger.warning(
        "env.param_names has %d names but state dimension is %d; using generic names.",
        len(param_names),
        state_dim,
    )
    param_names = [f"param_{i}" for i in range(state_dim)]

logger.info("State dimension: %d", state_dim)
logger.info("Number of actions: %d", n_actions)
logger.info("Parameters: %s", ", ".join(param_names))

policy_net = Net(state_dim, n_actions).to(device)
target_net = Net(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(
    policy_net.parameters(),
    lr=LEARNING_RATE,
    alpha=0.95,
    momentum=0.95,
)
memory = ReplayMemory(MEMORY_CAPACITY)
writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

history_rows: List[Dict[str, Any]] = []
steps_done = 0
completed_episodes = 0
start_episode = 0
last_loss = float("nan")
failed_evaluations = 0

best_reward = BestRecord()
best_feasible_q = BestRecord()
best_wavelength_match = BestRecord()
best_combined = BestRecord()


# -----------------------------------------------------------------------------
# DQN operations
# -----------------------------------------------------------------------------

def epsilon_at_step(step: int) -> float:
    return EPS_END + (EPS_START - EPS_END) * math.exp(-float(step) / EPS_DECAY)


def select_action(state: torch.Tensor) -> Tuple[torch.Tensor, float]:
    global steps_done
    epsilon = epsilon_at_step(steps_done)
    steps_done += 1

    if random.random() > epsilon:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
    else:
        action = torch.tensor(
            [[random.randrange(n_actions)]],
            device=device,
            dtype=torch.long,
        )
    return action, epsilon


def optimize_model() -> Optional[float]:
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(state is not None for state in batch.next_state),
        device=device,
        dtype=torch.bool,
    )
    non_final_items = [state for state in batch.next_state if state is not None]
    non_final_next_states = torch.cat(non_final_items) if non_final_items else None

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states is not None:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_values = reward_batch + GAMMA * next_state_values
    loss = nn.SmoothL1Loss()(state_action_values, expected_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


# -----------------------------------------------------------------------------
# History, summary and checkpoint saving
# -----------------------------------------------------------------------------

def history_fieldnames() -> List[str]:
    fixed = [
        "episode",
        "step_in_episode",
        "global_step",
        "action",
        "reward",
        "loss",
        "epsilon",
        "Qfactor",
        "wavelength_nm",
        "wavelength_error_nm",
        "wavelength_match_r4",
        "Q_normalized_r3",
        "combined_r5",
        "feasible",
        "mode_index",
        "done",
        "evaluation_ok",
        "elapsed_seconds",
    ]
    return fixed + param_names


def write_history_csv() -> None:
    fieldnames = history_fieldnames()
    temp_path = HISTORY_CSV.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer_csv.writeheader()
        for row in history_rows:
            writer_csv.writerow(row)
    os.replace(temp_path, HISTORY_CSV)


def tensor_rows() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    valid_rows = [row for row in history_rows if bool(row.get("evaluation_ok", False))]
    if not valid_rows:
        empty_x = torch.empty((0, state_dim), dtype=torch.float64)
        empty_y = torch.empty((0, 3), dtype=torch.float64)
        metrics = {
            "global_step": torch.empty((0,), dtype=torch.long),
            "episode": torch.empty((0,), dtype=torch.long),
            "step_in_episode": torch.empty((0,), dtype=torch.long),
            "action": torch.empty((0,), dtype=torch.long),
            "reward": torch.empty((0,), dtype=torch.float64),
            "loss": torch.empty((0,), dtype=torch.float64),
            "epsilon": torch.empty((0,), dtype=torch.float64),
            "Qfactor": torch.empty((0,), dtype=torch.float64),
            "wavelength_nm": torch.empty((0,), dtype=torch.float64),
            "wavelength_error_nm": torch.empty((0,), dtype=torch.float64),
            "feasible": torch.empty((0,), dtype=torch.bool),
            "mode_index": torch.empty((0,), dtype=torch.long),
        }
        return empty_x, empty_y, metrics

    input_tensor = torch.tensor(
        [[float(row[name]) for name in param_names] for row in valid_rows],
        dtype=torch.float64,
    )
    objective_tensor = torch.tensor(
        [
            [
                float(row["Q_normalized_r3"]),
                float(row["wavelength_match_r4"]),
                float(row["combined_r5"]),
            ]
            for row in valid_rows
        ],
        dtype=torch.float64,
    )
    metrics = {
        "global_step": torch.tensor([int(row["global_step"]) for row in valid_rows]),
        "episode": torch.tensor([int(row["episode"]) for row in valid_rows]),
        "step_in_episode": torch.tensor([int(row["step_in_episode"]) for row in valid_rows]),
        "action": torch.tensor([int(row["action"]) for row in valid_rows]),
        "reward": torch.tensor([float(row["reward"]) for row in valid_rows], dtype=torch.float64),
        "loss": torch.tensor([float(row["loss"]) for row in valid_rows], dtype=torch.float64),
        "epsilon": torch.tensor([float(row["epsilon"]) for row in valid_rows], dtype=torch.float64),
        "Qfactor": torch.tensor([float(row["Qfactor"]) for row in valid_rows], dtype=torch.float64),
        "wavelength_nm": torch.tensor(
            [float(row["wavelength_nm"]) for row in valid_rows], dtype=torch.float64
        ),
        "wavelength_error_nm": torch.tensor(
            [float(row["wavelength_error_nm"]) for row in valid_rows], dtype=torch.float64
        ),
        "feasible": torch.tensor([bool(row["feasible"]) for row in valid_rows], dtype=torch.bool),
        "mode_index": torch.tensor([int(row["mode_index"]) for row in valid_rows]),
    }
    return input_tensor, objective_tensor, metrics


def save_plot_data() -> None:
    input_tensor, objective_tensor, metrics = tensor_rows()
    atomic_torch_save(input_tensor, INPUT_PT)
    atomic_torch_save(objective_tensor, OBJECTIVE_PT)
    atomic_torch_save(metrics, METRICS_PT)

    metadata = {
        "parameter_names": param_names,
        "state_dim": state_dim,
        "number_of_actions": n_actions,
        "target_wavelength_nm": TARGET_WAVELENGTH_NM,
        "wavelength_tolerance_nm": WAVELENGTH_TOLERANCE_NM,
        "Q_goal": Q_GOAL,
        "objective_columns": ["r3_Q_over_Q_goal", "r4_wavelength_match", "r5_r3_times_r4"],
        "input_file": str(INPUT_PT),
        "objective_file": str(OBJECTIVE_PT),
        "metrics_file": str(METRICS_PT),
        "history_csv": str(HISTORY_CSV),
        "save_interval_evaluations": SAVE_INTERVAL_EVALUATIONS,
        "random_seed": RANDOM_SEED,
    }
    atomic_text_write(json.dumps(metadata, indent=2), RUN_METADATA_JSON)


def checkpoint_payload(next_episode: int) -> Dict[str, Any]:
    return {
        "version": 2,
        "next_episode": next_episode,
        "steps_done": steps_done,
        "completed_episodes": completed_episodes,
        "failed_evaluations": failed_evaluations,
        "last_loss": last_loss,
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_memory": memory.state_dict(),
        "history_rows": history_rows,
        "best_reward": asdict(best_reward),
        "best_feasible_q": asdict(best_feasible_q),
        "best_wavelength_match": asdict(best_wavelength_match),
        "best_combined": asdict(best_combined),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": {
            "num_episodes": NUM_EPISODES,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "save_interval": SAVE_INTERVAL_EVALUATIONS,
            "target_wavelength_nm": TARGET_WAVELENGTH_NM,
            "wavelength_tolerance_nm": WAVELENGTH_TOLERANCE_NM,
            "Q_goal": Q_GOAL,
            "state_dim": state_dim,
            "n_actions": n_actions,
            "param_names": param_names,
        },
    }


def save_checkpoint(next_episode: int, reason: str) -> None:
    write_history_csv()
    save_plot_data()
    atomic_torch_save(checkpoint_payload(next_episode), LATEST_CHECKPOINT)
    logger.info(
        "Saved periodic snapshot (%s): %d evaluations -> %s",
        reason,
        len(history_rows),
        RUN_ROOT,
    )


def restore_best_record(data: Dict[str, Any]) -> BestRecord:
    return BestRecord(
        value=float(data.get("value", -float("inf"))),
        global_step=int(data.get("global_step", -1)),
        episode=int(data.get("episode", -1)),
        step_in_episode=int(data.get("step_in_episode", -1)),
        state=data.get("state"),
        info=data.get("info"),
    )


def load_checkpoint_if_available() -> int:
    global steps_done, completed_episodes, failed_evaluations, last_loss
    global history_rows, best_reward, best_feasible_q, best_wavelength_match, best_combined

    if not RESUME_IF_AVAILABLE or not LATEST_CHECKPOINT.exists():
        return 0

    logger.info("Loading checkpoint: %s", LATEST_CHECKPOINT)
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)

    saved_config = checkpoint.get("config", {})
    if int(saved_config.get("state_dim", state_dim)) != state_dim:
        raise RuntimeError("Checkpoint state dimension does not match the current COMSOL environment.")
    if int(saved_config.get("n_actions", n_actions)) != n_actions:
        raise RuntimeError("Checkpoint action count does not match the current COMSOL environment.")

    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    target_net.load_state_dict(checkpoint["target_net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    memory.load_state_dict(checkpoint.get("replay_memory", {}))

    steps_done = int(checkpoint.get("steps_done", 0))
    completed_episodes = int(checkpoint.get("completed_episodes", 0))
    failed_evaluations = int(checkpoint.get("failed_evaluations", 0))
    last_loss = float(checkpoint.get("last_loss", float("nan")))
    history_rows = list(checkpoint.get("history_rows", []))

    best_reward = restore_best_record(checkpoint.get("best_reward", {}))
    best_feasible_q = restore_best_record(checkpoint.get("best_feasible_q", {}))
    best_wavelength_match = restore_best_record(checkpoint.get("best_wavelength_match", {}))
    best_combined = restore_best_record(checkpoint.get("best_combined", {}))

    if "random_state" in checkpoint:
        random.setstate(checkpoint["random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    next_episode = int(checkpoint.get("next_episode", completed_episodes))
    logger.info(
        "Resumed at episode %d with %d saved evaluations and %d replay transitions.",
        next_episode + 1,
        len(history_rows),
        len(memory),
    )
    return next_episode


def update_best_records(row: Dict[str, Any], state_values: np.ndarray, info: Dict[str, Any]) -> None:
    state_list = [float(value) for value in state_values]
    base = {
        "global_step": int(row["global_step"]),
        "episode": int(row["episode"]),
        "step_in_episode": int(row["step_in_episode"]),
        "state": state_list,
        "info": dict(info),
    }

    if float(row["reward"]) > best_reward.value:
        best_reward.value = float(row["reward"])
        best_reward.global_step = base["global_step"]
        best_reward.episode = base["episode"]
        best_reward.step_in_episode = base["step_in_episode"]
        best_reward.state = base["state"]
        best_reward.info = base["info"]
        atomic_torch_save(
            {
                "record": asdict(best_reward),
                "policy_net_state_dict": policy_net.state_dict(),
                "target_net_state_dict": target_net.state_dict(),
            },
            BEST_REWARD_CHECKPOINT,
        )

    if bool(row["feasible"]) and float(row["Qfactor"]) > best_feasible_q.value:
        best_feasible_q.value = float(row["Qfactor"])
        best_feasible_q.global_step = base["global_step"]
        best_feasible_q.episode = base["episode"]
        best_feasible_q.step_in_episode = base["step_in_episode"]
        best_feasible_q.state = base["state"]
        best_feasible_q.info = base["info"]
        atomic_torch_save(
            {
                "record": asdict(best_feasible_q),
                "policy_net_state_dict": policy_net.state_dict(),
                "target_net_state_dict": target_net.state_dict(),
            },
            BEST_FEASIBLE_Q_CHECKPOINT,
        )

    if float(row["wavelength_match_r4"]) > best_wavelength_match.value:
        best_wavelength_match.value = float(row["wavelength_match_r4"])
        best_wavelength_match.global_step = base["global_step"]
        best_wavelength_match.episode = base["episode"]
        best_wavelength_match.step_in_episode = base["step_in_episode"]
        best_wavelength_match.state = base["state"]
        best_wavelength_match.info = base["info"]

    if bool(row["feasible"]) and float(row["combined_r5"]) > best_combined.value:
        best_combined.value = float(row["combined_r5"])
        best_combined.global_step = base["global_step"]
        best_combined.episode = base["episode"]
        best_combined.step_in_episode = base["step_in_episode"]
        best_combined.state = base["state"]
        best_combined.info = base["info"]


def format_parameters(state_values: Optional[Sequence[float]], indent: str = "  ") -> List[str]:
    if state_values is None:
        return [f"{indent}<not available>"]
    return [
        f"{indent}{name:<24}: {float(value):.10f}"
        for name, value in zip(param_names, state_values)
    ]


def row_for_record(record: BestRecord) -> Optional[Dict[str, Any]]:
    for row in history_rows:
        if int(row.get("global_step", -2)) == record.global_step:
            return row
    return None


def top_rows(key: str, count: int = 5, feasible_only: bool = False) -> List[Dict[str, Any]]:
    rows = [
        row
        for row in history_rows
        if bool(row.get("evaluation_ok", False))
        and (not feasible_only or bool(row.get("feasible", False)))
        and math.isfinite(finite_float(row.get(key)))
    ]
    return sorted(rows, key=lambda item: float(item[key]), reverse=True)[:count]


def generate_summary() -> str:
    valid_rows = [row for row in history_rows if bool(row.get("evaluation_ok", False))]
    feasible_rows = [row for row in valid_rows if bool(row.get("feasible", False))]
    input_tensor, objective_tensor, _ = tensor_rows()

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DQN COMSOL PCSEL TRAINING SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Raw history rows:          {len(history_rows)}")
    lines.append(f"Real COMSOL evaluations:   {len(valid_rows)}")
    lines.append(f"Failed evaluations:        {failed_evaluations}")
    lines.append(f"Feasible (lambda ±{WAVELENGTH_TOLERANCE_NM:.0f} nm): {len(feasible_rows)}")
    lines.append(f"Completed episodes:        {completed_episodes}")
    lines.append(f"Input shape:               {tuple(input_tensor.shape)}")
    lines.append(f"Objective shape:           {tuple(objective_tensor.shape)}")
    lines.append(f"Target wavelength:         {TARGET_WAVELENGTH_NM:.3f} nm")
    lines.append(f"Q normalization goal:      {Q_GOAL:.6e}")
    lines.append("")

    if valid_rows:
        lines.append("-- Parameter ranges explored (successful COMSOL runs) --")
        for name in param_names:
            values = np.asarray([float(row[name]) for row in valid_rows], dtype=float)
            lines.append(
                f"  {name:<24}: min={np.min(values):.10g}  "
                f"max={np.max(values):.10g}  mean={np.mean(values):.10g}"
            )
        lines.append("")

        lines.append("-- Objective ranges --")
        objective_items = (
            ("r3 (Q/Q_goal)", "Q_normalized_r3"),
            ("r4 (wavelength match)", "wavelength_match_r4"),
            ("r5 (r3*r4)", "combined_r5"),
        )
        for label, key in objective_items:
            values = np.asarray([float(row[key]) for row in valid_rows], dtype=float)
            lines.append(
                f"  {label:<24}: min={np.nanmin(values):.6g}  "
                f"max={np.nanmax(values):.6g}  mean={np.nanmean(values):.6g}"
            )
        lines.append("")

        all_q = np.asarray([float(row["Qfactor"]) for row in valid_rows], dtype=float)
        lines.append("-- Q-factor range --")
        lines.append(f"  All successful: {np.nanmin(all_q):.6e} - {np.nanmax(all_q):.6e}")
        if feasible_rows:
            feasible_q = np.asarray([float(row["Qfactor"]) for row in feasible_rows], dtype=float)
            lines.append(
                f"  Feasible:       {np.nanmin(feasible_q):.6e} - {np.nanmax(feasible_q):.6e}"
            )
        else:
            lines.append("  Feasible:       none")
        lines.append("")

        lines.append("-- Top 5 designs by Q-factor (all successful runs) --")
        for rank, row in enumerate(top_rows("Qfactor", 5), start=1):
            lines.append(
                f"  #{rank} step={int(row['global_step'])} episode={int(row['episode'])} "
                f"Q={float(row['Qfactor']):.6e} r4={float(row['wavelength_match_r4']):.6f} "
                f"lambda_err=±{float(row['wavelength_error_nm']):.3f} nm "
                f"feasible={bool(row['feasible'])}"
            )
        lines.append("")

        lines.append("-- Top 5 designs by wavelength match --")
        for rank, row in enumerate(top_rows("wavelength_match_r4", 5), start=1):
            lines.append(
                f"  #{rank} step={int(row['global_step'])} episode={int(row['episode'])} "
                f"Q={float(row['Qfactor']):.6e} r4={float(row['wavelength_match_r4']):.6f} "
                f"lambda_err=±{float(row['wavelength_error_nm']):.3f} nm"
            )
        lines.append("")

    def append_record(title: str, record: BestRecord) -> None:
        lines.append(f"-- {title} --")
        row = row_for_record(record)
        if row is None:
            lines.append("  Not available")
            lines.append("")
            return
        lines.append(f"  Global step:        {int(row['global_step'])}")
        lines.append(f"  Episode:            {int(row['episode'])}")
        lines.append(f"  Step in episode:    {int(row['step_in_episode'])}")
        lines.append(f"  Reward:             {float(row['reward']):.10g}")
        lines.append(f"  Q:                  {float(row['Qfactor']):.10e}")
        lines.append(f"  Wavelength:         {float(row['wavelength_nm']):.10f} nm")
        lines.append(f"  Wavelength error:   ±{float(row['wavelength_error_nm']):.10f} nm")
        lines.append(f"  r3 = Q/Q_goal:      {float(row['Q_normalized_r3']):.10f}")
        lines.append(f"  r4 = lambda match:  {float(row['wavelength_match_r4']):.10f}")
        lines.append(f"  r5 = r3*r4:         {float(row['combined_r5']):.10f}")
        lines.append(f"  Feasible:           {bool(row['feasible'])}")
        lines.append(f"  Mode index:         {int(row['mode_index'])}")
        lines.append("  Input parameters (exact state sent forward by environment):")
        lines.extend(format_parameters(record.state, indent="    "))
        lines.append("")

    append_record("Best reward design", best_reward)
    append_record("Best feasible Q design", best_feasible_q)
    append_record("Best wavelength-match design", best_wavelength_match)
    append_record("Best combined feasible design", best_combined)

    lines.append("-- Output files --")
    lines.append(f"  Full training log:       {TRAINING_LOG}")
    lines.append(f"  Full CSV history:        {HISTORY_CSV}")
    lines.append(f"  Input tensor:            {INPUT_PT}")
    lines.append(f"  Objective tensor:        {OBJECTIVE_PT}")
    lines.append(f"  Additional metrics:      {METRICS_PT}")
    lines.append(f"  Latest DQN checkpoint:   {LATEST_CHECKPOINT}")
    lines.append(f"  TensorBoard directory:   {TENSORBOARD_DIR}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

start_episode = load_checkpoint_if_available()
run_start_time = time.time()

try:
    for i_episode in range(start_episode, NUM_EPISODES):
        logger.info("Starting episode %d/%d", i_episode + 1, NUM_EPISODES)

        state_np = reset_env(env)
        if state_np.size != state_dim:
            raise RuntimeError(
                f"Environment reset returned {state_np.size} values; expected {state_dim}."
            )
        state = torch.from_numpy(state_np).unsqueeze(0).to(device=device, dtype=torch.float32)

        for step_index in range(MAX_EPISODE_STEPS):
            action, epsilon = select_action(state)
            evaluation_start = time.time()

            try:
                obs_np, reward_value, done, info = step_env(env, int(action.item()))
                q_factor, wavelength_nm, mode_index = extract_metrics(info)
                wavelength_error_nm, r4, feasible = wavelength_metrics(wavelength_nm)
                evaluation_ok = (
                    obs_np.size == state_dim
                    and math.isfinite(reward_value)
                    and math.isfinite(q_factor)
                    and math.isfinite(wavelength_nm)
                )
                if not evaluation_ok:
                    failed_evaluations += 1
            except Exception:
                failed_evaluations += 1
                logger.exception(
                    "COMSOL/environment evaluation failed at episode %d, step %d.",
                    i_episode + 1,
                    step_index + 1,
                )
                # Save everything collected before propagating the error to Slurm.
                save_checkpoint(i_episode, reason="evaluation failure")
                atomic_text_write(generate_summary(), SUMMARY_TXT)
                raise

            if obs_np.size != state_dim:
                raise RuntimeError(
                    f"Environment step returned {obs_np.size} state values; expected {state_dim}."
                )

            reward_tensor = torch.tensor([reward_value], device=device, dtype=torch.float32)
            next_state = None if done else torch.from_numpy(obs_np).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )

            memory.push(state, action, next_state, reward_tensor)
            if next_state is not None:
                state = next_state

            # Train only at the configured update frequency.
            if steps_done % UPDATE_FREQ_EVALUATIONS == 0:
                new_loss = optimize_model()
                if new_loss is not None:
                    last_loss = new_loss

            q_normalized = q_factor / Q_GOAL if math.isfinite(q_factor) else float("nan")
            combined_score = q_normalized * r4 if math.isfinite(q_normalized) and math.isfinite(r4) else float("nan")
            global_step = len(history_rows) + 1

            row: Dict[str, Any] = {
                "episode": i_episode + 1,
                "step_in_episode": step_index + 1,
                "global_step": global_step,
                "action": int(action.item()),
                "reward": reward_value,
                "loss": last_loss,
                "epsilon": epsilon,
                "Qfactor": q_factor,
                "wavelength_nm": wavelength_nm,
                "wavelength_error_nm": wavelength_error_nm,
                "wavelength_match_r4": r4,
                "Q_normalized_r3": q_normalized,
                "combined_r5": combined_score,
                "feasible": feasible,
                "mode_index": mode_index,
                "done": done,
                "evaluation_ok": evaluation_ok,
                "elapsed_seconds": time.time() - evaluation_start,
            }
            for index, name in enumerate(param_names):
                row[name] = float(obs_np[index])
            history_rows.append(row)

            if evaluation_ok:
                update_best_records(row, obs_np, info)

            writer.add_scalar("training/reward", reward_value, global_step)
            if math.isfinite(last_loss):
                writer.add_scalar("training/loss", last_loss, global_step)
            writer.add_scalar("training/epsilon", epsilon, global_step)
            if math.isfinite(q_factor):
                writer.add_scalar("metrics/Qfactor", q_factor, global_step)
            if math.isfinite(wavelength_nm):
                writer.add_scalar("metrics/wavelength_nm", wavelength_nm, global_step)
                writer.add_scalar("metrics/wavelength_error_nm", wavelength_error_nm, global_step)
                writer.add_scalar("metrics/wavelength_match_r4", r4, global_step)
            writer.add_scalar("metrics/feasible", float(feasible), global_step)
            if math.isfinite(combined_score):
                writer.add_scalar("metrics/combined_r5", combined_score, global_step)

            logger.info(
                "episode=%d step=%d global=%d action=%d reward=%.6g loss=%s "
                "epsilon=%.5f Q=%.6e lambda=%.6f nm error=%.6f nm "
                "r4=%.8f feasible=%s mode=%d eval_time=%.2fs",
                i_episode + 1,
                step_index + 1,
                global_step,
                int(action.item()),
                reward_value,
                f"{last_loss:.6g}" if math.isfinite(last_loss) else "nan",
                epsilon,
                q_factor,
                wavelength_nm,
                wavelength_error_nm,
                r4,
                feasible,
                mode_index,
                row["elapsed_seconds"],
            )

            # RBF-style periodic snapshot: overwrite cumulative files every N runs.
            if global_step % SAVE_INTERVAL_EVALUATIONS == 0:
                save_checkpoint(i_episode, reason=f"evaluation {global_step}")
                atomic_text_write(generate_summary(), SUMMARY_TXT)

            if done:
                logger.info("Environment ended episode %d after %d steps.", i_episode + 1, step_index + 1)
                break

        completed_episodes = i_episode + 1
        if completed_episodes % TARGET_UPDATE_EPISODES == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logger.info("Target network updated after episode %d.", completed_episodes)

        # Save at each episode boundary as one cumulative file, not per step.
        save_checkpoint(i_episode + 1, reason=f"episode {completed_episodes} completed")
        atomic_text_write(generate_summary(), SUMMARY_TXT)

    logger.info("Training complete: %d episodes, %d evaluations.", completed_episodes, len(history_rows))

finally:
    # Always leave a usable cumulative snapshot, even after Ctrl+C or Slurm signal.
    try:
        save_checkpoint(completed_episodes, reason="finalization")
        summary_text = generate_summary()
        atomic_text_write(summary_text, SUMMARY_TXT)
        logger.info("\n%s", summary_text)
    except Exception:
        logger.exception("Unable to complete final save.")

    writer.flush()
    writer.close()
    try:
        env.close()
    except Exception:
        logger.exception("Environment close failed.")

    logger.info("Total wall time: %.2f minutes", (time.time() - run_start_time) / 60.0)
