from __future__ import annotations

import logging

from reporelay_engine.models import BlendState, UserProfile

logger = logging.getLogger(__name__)


class ThresholdBlender:
    """
    Blends recommendation strategies with per-user dynamic weights.

    Weight evolution based on user life-cycle:

        Cold (0 interactions)
          content-based:  0.30
          item-based CF:  0.50
          user-based CF:  0.00  (not enough data)
          exploration:     0.20

        Warm (3+ interactions)
          content-based:  0.20
          item-based CF:  0.45
          user-based CF:  0.25
          exploration:     0.10

        Hot (10+ interactions)
          content-based:  0.10
          item-based CF:  0.35  (stays relevant, keeps novelty window open)
          user-based CF:  0.45
          exploration:     0.10
    """

    def __init__(self, user: UserProfile) -> None:
        self.user = user
        self.state = user.blend

    def adapt_weights(self, recent_feedback: list[str]) -> BlendState:
        """Adjust weights based on recent thumbs feedback."""
        state = self.state

        state.total_interactions += len(recent_feedback)

        for feedback in recent_feedback:
            state.feedback_window.append(feedback)
            if feedback == "up":
                state.total_thumbs_up += 1
            elif feedback == "down":
                state.total_thumbs_down += 1

        if len(state.feedback_window) > 10:
            state.feedback_window = state.feedback_window[-10:]

        quality = self._feedback_quality()

        if quality > 0.7:
            self._boost_exploration()
        elif quality < 0.4:
            self._shift_strategy()
        else:
            self._maintain()

        self._update_stage()

        logger.info(
            "blend for %s: stage=%s content=%.2f item=%.2f user=%.2f explore=%.2f quality=%.2f",
            state.user_id,
            state.current_data_stage,
            state.weight_content,
            state.weight_item_cf,
            state.weight_user_cf,
            state.weight_exploration,
            quality,
        )

        return state

    def _feedback_quality(self) -> float:
        window = self.state.feedback_window
        if not window:
            return 0.5
        ups = window.count("up")
        return ups / len(window)

    def _boost_exploration(self) -> None:
        """User likes what we show — raise exploration to avoid filter bubble."""

        self.state.weight_exploration = min(0.25, self.state.weight_exploration + 0.05)
        self.state.weight_user_cf = max(0.10, self.state.weight_user_cf - 0.025)
        self.state.weight_item_cf = max(0.15, self.state.weight_item_cf - 0.025)
        self._normalize()

    def _shift_strategy(self) -> None:
        """User dislikes current mode — shift toward the alternative."""

        if self.state.weight_item_cf > self.state.weight_user_cf:
            self.state.weight_item_cf -= 0.10
            self.state.weight_user_cf += 0.10
        else:
            self.state.weight_user_cf -= 0.10
            self.state.weight_item_cf += 0.10

        self.state.weight_exploration = max(0.05, self.state.weight_exploration + 0.05)
        self._normalize()

    def _maintain(self) -> None:
        """No strong signal — hold weights steady. Let data accumulation drive change."""
        self._update_stage()

    def _normalize(self) -> None:
        total = (
            self.state.weight_content
            + self.state.weight_item_cf
            + self.state.weight_user_cf
            + self.state.weight_exploration
        )
        if total > 0:
            self.state.weight_content /= total
            self.state.weight_item_cf /= total
            self.state.weight_user_cf /= total
            self.state.weight_exploration /= total

    def _update_stage(self) -> None:
        n = self.state.total_interactions

        if n >= self.state.hot_threshold:
            self.state.current_data_stage = "hot"
            self._set_stage_weights("hot")
        elif n >= self.state.warm_threshold:
            self.state.current_data_stage = "warm"
            self._set_stage_weights("warm")
        else:
            self.state.current_data_stage = "cold"
            self._set_stage_weights("cold")

    def _set_stage_weights(self, stage: str) -> None:
        base = {
            "cold": {"content": 0.30, "item_cf": 0.50, "user_cf": 0.00, "explore": 0.20},
            "warm": {"content": 0.20, "item_cf": 0.45, "user_cf": 0.25, "explore": 0.10},
            "hot": {"content": 0.10, "item_cf": 0.35, "user_cf": 0.45, "explore": 0.10},
        }[stage]

        n = self.state.total_interactions

        if stage == "cold":
            self.state.weight_content = base["content"]
            self.state.weight_item_cf = base["item_cf"]
            self.state.weight_user_cf = base["user_cf"]
            self.state.weight_exploration = base["explore"]
        elif stage == "warm":
            progress = min(1.0, (n - self.state.warm_threshold) / (self.state.hot_threshold - self.state.warm_threshold))
            self.state.weight_content = self._lerp(base["content"], 0.10, progress)
            self.state.weight_item_cf = self._lerp(base["item_cf"], 0.35, progress)
            self.state.weight_user_cf = self._lerp(base["user_cf"], 0.45, progress)
            self.state.weight_exploration = 0.10
        else:
            self.state.weight_content = base["content"]
            self.state.weight_item_cf = base["item_cf"]
            self.state.weight_user_cf = base["user_cf"]
            self.state.weight_exploration = base["explore"]

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t
