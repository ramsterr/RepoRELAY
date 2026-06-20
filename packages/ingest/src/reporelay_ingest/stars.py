from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_sample_stars(repo_ids: list[int], users: list[int], events_per_repo: int = 50) -> list[dict]:
    """Generate synthetic star events for testing without BigQuery."""
    import random

    random.seed(42)
    events: list[dict] = []
    now = datetime.now(timezone.utc)

    for repo_id in repo_ids:
        star_users = random.sample(users, min(events_per_repo, len(users)))
        for user_id in star_users:
            days_ago = random.randint(0, 180)
            starred_at = int((now.timestamp() - days_ago * 86400) * 1000)
            events.append({
                "user_id": user_id,
                "repo_id": repo_id,
                "starred_at": starred_at,
            })

    return events


def save_stars_jsonl(events: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    logger.info("Wrote %d star events to %s", len(events), path)


def load_stars_jsonl(path: str) -> list[dict]:
    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
