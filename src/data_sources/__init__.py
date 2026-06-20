from .base import DataSource
from .simulator import SimulatorSource
from .espncricinfo import ESPNCricinfoSource
from .cricbuzz import CricbuzzSource
from .wikipedia import WikipediaSource

SOURCE_REGISTRY = {
    "simulator": SimulatorSource,
    "espncricinfo": ESPNCricinfoSource,
    "cricbuzz": CricbuzzSource,
    "wikipedia": WikipediaSource,
}

__all__ = ["DataSource", "SOURCE_REGISTRY"]
