"""DQN-compatible COMSOL Gym environment using a Derived Values node (gev1).


Recommended COMSOL setup:
- disable/remove Parametric Sweep from the study
- keep only Eigenfrequency in the study
- ensure gev1 still contains these expressions in this order:
    0: ewfd.freq
    1: ewfd.lambda0
    2: a/ewfd.lambda0
    3: ewfd.Qfactor
    4: a
"""

from __future__ import annotations

from typing import Dict, Tuple
import traceback

import gym
from gym import spaces
import numpy as np
import mph
import os

class ComsolEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        mph_path: str = os.environ.get("MPH_PATH", "2.2u_PCSEL_Si_v1.mph"),
        study_tag: str = "std2",
        numerical_tag: str = "gev1",
        target_lambda_nm: float = 2200.0,
        lambda_window_nm: float = 100.0,
        max_steps: int = 20,
        print_all_modes: bool = False,
        print_selected_mode: bool = False,
    ):
        super().__init__()

        self.mph_path = mph_path
        self.study_tag = study_tag
        self.numerical_tag = numerical_tag
        self.target_lambda_nm = float(target_lambda_nm)
        self.lambda_window_nm = float(lambda_window_nm)
        self.max_steps = int(max_steps)
        self.print_all_modes = bool(print_all_modes)
        self.print_selected_mode = bool(print_selected_mode)

        self.param_names = [
            "a",
            "r",
            "t_active",
            "t_ncladding",
            "t_pcladding",
            "Lm",
            "t_air",
            "t_pcontact",
      	    "t_ge",
        ]

        self.param_units = {
            "a": "nm",
            "t_active": "nm",
            "t_ncladding": "nm",
            "t_pcladding": "nm",
            "Lm": "nm",
            "t_air": "nm",
            "t_pcontact": "nm",
            "t_ge": "nm",
        }

        self.initial_state = np.array(
            [520.0, 0.15, 520.0, 2000.0, 360.0, 2200.0, 1500.0, 100.0, 460.0],
            dtype=np.float64,
        )
        self.state = self.initial_state.copy()

        self.lower = np.array(
            [480.0, 0.10, 420.0, 1200.0, 200.0, 1900.0, 100.0, 50.0, 330.0],
            dtype=np.float64,
        )
        self.upper = np.array(
            [560.0, 0.30, 700.0, 2600.0, 600.0, 2500.0, 3000.0, 200.0, 600.0],
            dtype=np.float64,
        )

        self.action_space = spaces.Discrete(2 * len(self.param_names) + 1)
        self.observation_space = spaces.Box(
            low=self.lower, high=self.upper, dtype=np.float64
        )

        self.step_sizes = np.array(
            [2.0, 0.005, 10.0, 50.0, 10.0, 20.0, 50.0, 5.0, 10.0],
            dtype=np.float64,
        )

        self.Q_goal = 2.0e6
        self.lam_goal = float(target_lambda_nm)

        self.cur_step = 0
        self.best_score = -np.inf

        self.client = mph.start(cores=4)
        self.model = self.client.load(self.mph_path)

        try:
            print("Study tags:", list(self.model.java.study().tags()))
            print("Numerical tags:", list(self.model.java.result().numerical().tags()))
        except Exception:
            pass

        self._gev1_n_expr = 5

    def _clip_state(self) -> None:
        self.state = np.clip(self.state, self.lower, self.upper)

    def _format_param_value(self, name: str, value: float) -> str:
        if name in self.param_units:
            return f"{float(value)}[{self.param_units[name]}]"
        return str(float(value))

    def _set_params(self, x: np.ndarray) -> None:
        for name, val in zip(self.param_names, x):
            self.model.parameter(name, self._format_param_value(name, float(val)))

    def _apply_action(self, action: int) -> None:
        if action == (2 * len(self.param_names)):
            return
        idx = action // 2
        direction = +1.0 if action % 2 == 0 else -1.0
        self.state[idx] += direction * self.step_sizes[idx]
        self._clip_state()

    def _run_study(self) -> None:
        self.model.java.study(self.study_tag).run()

    def _reload_model(self) -> None:
        try:
            self.client.clear()
        except Exception:
            pass
        self.model = self.client.load(self.mph_path)

    def _read_gev1(self):
        num = self.model.java.result().numerical(self.numerical_tag)
        num.run()
        raw = num.getReal()
        if raw is None or len(raw) == 0 or len(list(raw[0])) == 0:
            raise RuntimeError("gev1 returned no eigenmode data   study may not have converged")
        freqs = np.array(list(raw[0]), dtype=np.float64)
        lams  = np.array(list(raw[1]), dtype=np.float64)
        Qs    = np.array(list(raw[3]), dtype=np.float64)
        return freqs, lams, Qs

    def _pick_mode(self, lams: np.ndarray, Qs: np.ndarray) -> int:
        candidates = np.where(np.abs(lams - self.target_lambda_nm) < self.lambda_window_nm)[0]
        if candidates.size > 0:
            idx = candidates[np.argmax(Qs[candidates])]
        else:
            idx = int(np.argmin(np.abs(lams - self.target_lambda_nm)))
        return int(idx)

    def _compute_reward(self, q_factor: float, wavelength_nm: float) -> float:
        q_term = q_factor / self.Q_goal
        lam_term = 1.0 - abs(wavelength_nm - self.lam_goal) / self.lam_goal
        lam_term = max(lam_term, -1.0)
        return float(100.0 * q_term + 40.0 * lam_term)

    def reset(self):
        self.state = self.initial_state.copy()
        self.cur_step = 0
        self.best_score = -np.inf
        self._set_params(self.state)
        return self.state.copy()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.cur_step += 1
        self._apply_action(int(action))

        try:
            self._set_params(self.state)
            self._run_study()

            freqs, lams, Qs = self._read_gev1()

            if self.print_all_modes:
                print("\nAll modes from gev1:")
                for i, (freq_i, lam_i, q_i) in enumerate(zip(freqs, lams, Qs)):
                    print(
                        f"mode {i}: freq={freq_i:.6e}, "
                        f"lambda={lam_i:.3f} nm, Q={q_i:.6e}"
                    )

            idx = self._pick_mode(lams, Qs)

            if self.print_selected_mode:
                print(
                    f"Selected mode={idx}, "
                    f"lambda={float(lams[idx]):.3f} nm, "
                    f"Q={float(Qs[idx]):.6e}"
                )
            
            q_factor = float(Qs[idx])
            wavelength_nm = float(lams[idx])
            freq = float(freqs[idx])

            reward = self._compute_reward(q_factor, wavelength_nm)
            self.best_score = max(self.best_score, reward)
            done = self.cur_step >= self.max_steps

            info: Dict[str, float] = {
                "mode_index": idx,
                "Qfactor": q_factor,
                "wavelength_nm": wavelength_nm,
                "freq": freq,
                "best_score": self.best_score,
            }

            return self.state.copy(), reward, done, info

        except Exception as e:
            print("\n[COMSOL STEP ERROR]")
            print("Exception:", repr(e))
            traceback.print_exc()
            try:
                self._reload_model()
            except Exception:
                pass

            reward = -1.0e6
            done = True
            info = {
                "error": str(e),
                "Qfactor": -1.0,
                "wavelength_nm": -1.0,
                "freq": -1.0,
                "best_score": self.best_score,
            }
            return self.state.copy(), reward, done, info

    def render(self, mode="human"):
        print({name: val for name, val in zip(self.param_names, self.state)})

    def close(self):
        try:
            self.client.clear()
        except Exception:
            pass
