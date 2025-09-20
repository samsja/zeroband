import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add with simpler format
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)
