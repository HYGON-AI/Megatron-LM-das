# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, process-local policy for comparing UltraEP with master-only training.

The training bridge supplies globally reduced times and validity. This policy
does not synchronize devices, execute training, or advance the runtime tuner.
"""

from dataclasses import dataclass
import math
from statistics import median


@dataclass(frozen=True)
class AdaptiveConfig:
    samples: int = 3
    max_samples: int = 5
    min_speedup: float = 0.01
    max_calibration_steps: int = 48

    def __post_init__(self):
        if self.samples < 3 or self.samples % 2 != 1:
            raise ValueError("adaptive samples must be odd and >= 3")
        if self.max_samples < self.samples or self.max_samples % 2 != 1:
            raise ValueError("adaptive max samples must be odd and >= samples")
        if not math.isfinite(self.min_speedup) or not 0 <= self.min_speedup < 1:
            raise ValueError("adaptive min speedup must be finite and in [0, 1)")
        if self.max_calibration_steps < 2 * self.samples + 6:
            raise ValueError("adaptive calibration budget must be >= 2 * samples + 6")


class AdaptiveController:
    """Warm up baseline, finish optional tuning, then alternate fixed modes.

    Each mode change gets one unscored settling step. A/B pairs reverse order
    on successive rounds. Consistent paired wins allow early selection;
    disagreement gets bounded additional observations. Final selection uses
    median step times without a duration-noise or paired-win threshold.
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.phase = "warmup"
        self.active = False
        self.steps = 0
        self._warmups = 2
        self._settling = 0
        self._probe_index = 0
        self.times = {False: [], True: []}
        self.result = None
        self._workload = None

    @property
    def complete(self):
        return self.phase == "complete"

    @property
    def needs_timing(self):
        return self.phase == "compare" and self._settling == 0

    def _probe_mode(self):
        # A B | B A | A B ...
        round_index, position = divmod(self._probe_index, 2)
        return bool(position ^ (round_index % 2))

    def _next_mode(self):
        active = self._probe_mode()
        self._settling = int(active != self.active)
        self.active = active

    def _measurements(self):
        # Each A/B or B/A round contributes one entry to each mode's list.
        # Exclude a pending half-round from both scores and win counts when
        # the total budget is reached.
        count = min(map(len, self.times.values()))
        times = {mode: values[:count] for mode, values in self.times.items()}
        baseline = median(times[False]) if count else None
        ultraep = median(times[True]) if count else None
        paired_speedups = [
            (baseline_ms - ultraep_ms) / baseline_ms
            for baseline_ms, ultraep_ms in zip(times[False], times[True])
        ]
        return {
            "baseline_ms": baseline,
            "ultraep_ms": ultraep,
            "relative_mad": {
                name: median(abs(t - center) for t in times[mode]) / center if center else None
                for mode, name, center in ((False, 'baseline', baseline), (True, 'ultraep', ultraep))
            },
            "speedup": (baseline - ultraep) / baseline if baseline and ultraep else None,
            "paired_speedups": paired_speedups,
            "paired_wins": sum(speedup > 0 for speedup in paired_speedups),
            "paired_samples": len(paired_speedups),
            "samples": {"baseline": len(self.times[False]), "ultraep": len(self.times[True])},
        }

    def _has_median_speedup(self, measurements):
        return (
            measurements['paired_samples'] >= self.config.samples
            and measurements['speedup'] > 0
            and measurements['speedup'] >= self.config.min_speedup
        )

    def _finish(self, active, reason):
        self.active = active
        self.phase = "complete"
        self.result = {
            "enabled": active,
            "reason": reason,
            **self._measurements(),
            "calibration_steps": self.steps,
        }

    def step_end(self, elapsed_ms, *, valid=True, tuning=False, workload=None):
        if self.complete:
            return
        self.steps += 1
        if workload is not None:
            if self._workload is not None and workload != self._workload:
                self._finish(False, "workload changed during calibration")
                return
            self._workload = workload
        if valid:
            if self.phase == "warmup":
                self._warmups -= 1
                if self._warmups == 0:
                    self.phase = "tune" if tuning else "compare"
                    if tuning:
                        self.active = True
                    else:
                        self._next_mode()
            elif self.phase == "tune":
                if not tuning:
                    self.phase = "compare"
                    self._next_mode()
            elif self._settling:
                self._settling -= 1
            elif elapsed_ms is not None and math.isfinite(elapsed_ms) and elapsed_ms > 0:
                self.times[self.active].append(elapsed_ms)
                self._probe_index += 1
                count = min(map(len, self.times.values()))
                if count >= self.config.samples and count % 2 == 1 and self._probe_index % 2 == 0:
                    measurements = self._measurements()
                    enabled = self._has_median_speedup(measurements)
                    no_gain = not any(
                        speedup > 0 and speedup >= self.config.min_speedup
                        for speedup in measurements['paired_speedups']
                    )
                    if count >= self.config.max_samples:
                        self._finish(enabled, "measured speedup" if enabled else "insufficient stable speedup")
                    elif enabled and measurements['paired_wins'] == count:
                        self._finish(True, "measured speedup")
                    elif no_gain:
                        self._finish(False, "insufficient stable speedup")
                if not self.complete:
                    self._next_mode()
        if not self.complete and self.steps >= self.config.max_calibration_steps:
            # Do not discard already consistent evidence merely because an
            # extra observation or settling step consumed the remaining budget.
            enabled = self._has_median_speedup(self._measurements())
            self._finish(enabled, "measured speedup" if enabled else "calibration budget exhausted")
