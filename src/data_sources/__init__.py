from .base import DataSource
from .simulator import SimulatorSource
from .espncricinfo import ESPNCricinfoSource
from .cricbuzz import CricbuzzSource

SOURCE_REGISTRY = {
    "simulator": SimulatorSource,
    "espncricinfo": ESPNCricinfoSource,
    "cricbuzz": CricbuzzSource,
}

__all__ = ["DataSource", "SOURCE_REGISTRY"]
