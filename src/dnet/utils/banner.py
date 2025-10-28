"""Startup banner utilities."""

from pathlib import Path
from .logger import logger


def _find_art_file() -> Path | None:
    """Locate the ASCII art file misc/dnet.art relative to common roots."""
    candidates: list[Path] = []
    try:
        candidates.append(Path.cwd() / "misc" / "dnet.art")
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        parents = list(here.parents)
        for idx in (3, 4, 5):
            if idx < len(parents):
                candidates.append(parents[idx] / "misc" / "dnet.art")
    except Exception:
        pass

    for p in candidates:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def print_startup_banner() -> None:
    """Print the ASCII art banner at startup if available."""
    art_path = _find_art_file()
    if art_path is None:
        return
    try:
        art = art_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    if art.strip():
        # Prepend a newline so the art starts on a fresh line in logs
        logger.info("\n%s", art)

