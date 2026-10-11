from importlib import import_module
from pathlib import Path


def config_dir(config_id: int = 1) -> Path:
    if config_id < 1:
        raise ValueError("config must be at least 1")
    directory = Path(__file__).resolve().parent / str(config_id)
    for filename in ("reward.py", "fitness.py", "actions.json", "neat_config.ini"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"config {config_id}: missing {directory / filename}")
    return directory


def reward_module(config_id: int = 1):
    config_dir(config_id)
    return import_module(f".{config_id}.reward", __name__)


def fitness_module(config_id: int = 1):
    config_dir(config_id)
    return import_module(f".{config_id}.fitness", __name__)


CONFIG_DIR = Path(__file__).resolve().parent / "1"
NEAT_CONFIG = CONFIG_DIR / "neat_config.ini"
