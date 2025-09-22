from loguru import logger
from rich.console import Console

console = Console()
logger.remove()


def print(*args, **kwargs):
    console.print(*args, **kwargs, end="")


logger.add(
    print,
    level="TRACE",
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    colorize=True,
)
