from .base import DataSource
from .simulator import SimulatorSource
from .espncricinfo import ESPNCricinfoSource
from .cricbuzz import CricbuzzSource
from .cricsheet import CricsheetSource
from .wikipedia import WikipediaSource

SOURCE_REGISTRY = {
    "simulator": SimulatorSource,
    "espncricinfo": ESPNCricinfoSource,
    "cricbuzz": CricbuzzSource,
    "cricsheet": CricsheetSource,
    "wikipedia": WikipediaSource,
}

__all__ = ["DataSource", "SOURCE_REGISTRY"]
