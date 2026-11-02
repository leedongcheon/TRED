"""Helpers shared by the pipeline scripts."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def analyzer():
    from seed.analysis import SEEDEpisodeAnalyzer

    return SEEDEpisodeAnalyzer(backend="openai", skill_mode="episode_only", render_response="action_only",
                               include_episode_summary=True)

