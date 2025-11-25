"""Startup banner utilities."""

from typing import Optional
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


def get_banner_text() -> str | None:
    """Get the ASCII art banner text if available."""
    art_path = _find_art_file()
    if art_path is None:
        return None
    try:
        return art_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def print_startup_banner(tag: Optional[str] = None) -> None:
    """Print the ASCII art banner at startup if available."""
    art = get_banner_text()
    if art and art.strip():
        # Prepend a newline so the art starts on a fresh line in logs
        if tag:
            art += f"\n[=== {tag.upper()} ===]\n"
        logger.info("\n%s", art)
