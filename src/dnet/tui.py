"""Terminal User Interface for dnet using Rich."""

import psutil
import asyncio
import logging
from collections import deque
from typing import Optional, Deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.table import Table

from dnet.utils.banner import get_banner_text
from dnet.utils.logger import logger as dnet_logger


class TUILogHandler(logging.Handler):
    """Custom logging handler that sends logs to the TUI."""

    def __init__(self, log_queue: Deque[str]):
        super().__init__()
        self.log_queue = log_queue
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            style = ""
            if record.levelno >= logging.ERROR:
                style = "[red]"
            elif record.levelno >= logging.WARNING:
                style = "[yellow]"
            elif record.levelno >= logging.INFO:
                style = "[white]"
            else:
                style = "[dim]"

            self.log_queue.append(f"{style}{msg}[/]")
        except Exception:
            self.handleError(record)


class DnetTUI:
    """Manages the Rich TUI layout and updates."""

    def __init__(self, title: str = "DNET"):
        self.console = Console()
        self.layout = Layout()
        self.log_queue: Deque[str] = deque(maxlen=100)
        self.status_message = "Initializing..."
        self.is_running = False
        self.title = title
        self.banner_text = get_banner_text() or title

        # Model Info State
        self.model_name: Optional[str] = None
        self.model_layers: int = 0
        self.model_residency: int = 0
        self.model_loaded: bool = False
        self.show_model_info: bool = True  # Always visible
        self.show_layers_visual: bool = True

        # Calculate header size based on banner lines, defaulting to 3 if no banner
        header_size = len(self.banner_text.splitlines()) + 2 if get_banner_text() else 3

        # Setup layout
        self.layout.split(
            Layout(name="header", size=header_size),
            Layout(name="body", ratio=2),
            Layout(name="model_info", size=3),  # Always visible
            Layout(name="footer", size=3),
        )

        # Setup logging
        self.log_handler = TUILogHandler(self.log_queue)
        dnet_logger.addHandler(self.log_handler)

    def _generate_header(self) -> Panel:
        return Panel(
            Text(self.banner_text, justify="left", style="bold white"),
            style="cyan",
            title=self.title,
        )

    def _generate_logs(self) -> Panel:
        header_height = self.layout["header"].size or 3
        model_info_height = self.layout["model_info"].size or 3
        footer_height = self.layout["footer"].size or 3

        total_height = self.console.size.height
        body_layout_height = total_height - (
            header_height + model_info_height + footer_height
        )

        available_lines = max(1, body_layout_height - 4)

        visible_logs = list(self.log_queue)[-available_lines:]
        log_text = "\n".join(visible_logs)

        text = Text.from_markup(log_text)
        text.no_wrap = True
        text.overflow = "ellipsis"

        return Panel(
            text,
            title="Logs",
            border_style="cyan",
            padding=(0, 1),
        )

    def _generate_model_info(self) -> Panel:
        if not self.model_name:
            return Panel(
                Text("No model loaded", style="dim"),
                title="Model Info",
                border_style="cyan",
            )

        status_style = "bold green" if self.model_loaded else "bold yellow"
        status_text = "LOADED" if self.model_loaded else "LOADING..."

        # Visualize layers as boxes
        # Limit to avoid cluttering if too many layers
        max_boxes = 40
        layers_visual = Text()

        if self.model_layers > 0:
            num_boxes = min(self.model_layers, max_boxes)

            # Determine how many are resident
            # If loaded, we show residency. If loading, maybe just show empty?
            # Let's show residency if loaded.

            for i in range(num_boxes):
                is_resident = i < self.model_residency
                # Filled box for resident, empty for non-resident
                char = "■" if is_resident else "□"
                # Green for resident, dim/blue for non-resident
                style = "bold green" if is_resident else "blue"

                layers_visual.append(f"[{char} ", style=style)

            if self.model_layers > max_boxes:
                layers_visual.append(
                    f"... (+{self.model_layers - max_boxes})", style="dim"
                )

        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")

        grid.add_row(
            f"[bold]{self.model_name}[/bold] ({self.model_layers} layers, {self.model_residency} resident)",
            f"[{status_style}]{status_text}[/]",
        )
        if self.model_layers > 0 and self.show_layers_visual:
            grid.add_row(layers_visual, "")

        return Panel(
            grid,
            title="Model Info",
            border_style="cyan",
        )

    def _generate_footer(self) -> Panel:
        spinner = Spinner("aesthetic", text=f" {self.status_message}", style="cyan")

        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        mem_text = f"[bold]RAM:[/bold] {used_gb:.1f}/{total_gb:.1f} GB (Avail: {avail_gb:.1f} GB)"

        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")

        grid.add_row(
            spinner, Text.from_markup(f"{mem_text}  [dim]Ctrl+C to stop[/dim]")
        )

        return Panel(
            grid,
            title="Current Work",
            border_style="cyan",
        )

    def update_status(self, message: str) -> None:
        self.status_message = message

    def update_model_info(
        self,
        name: str,
        layers: int,
        residency: int = 0,
        loaded: bool = False,
        show_layers_visual: bool = True,
    ) -> None:
        """Update the model information panel."""
        self.model_name = name
        self.model_layers = layers
        self.model_residency = residency
        self.model_loaded = loaded
        self.show_model_info = True
        self.show_layers_visual = show_layers_visual
        # Adjust size if needed, maybe 4 if we have layers visual
        self.layout["model_info"].size = 4 if (layers > 0 and show_layers_visual) else 3

    def _on_log_record(self, record: logging.LogRecord):
        """Callback for new log records."""
        if "Repacking model weights" in record.getMessage():
            self.update_status("Repacking weights...")
        elif "Model loaded successfully" in record.getMessage():
            self.update_status("Running...")

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the TUI loop until stop_event is set."""
        self.is_running = True

        with Live(self.layout, console=self.console, refresh_per_second=4, screen=True):
            while not stop_event.is_set():
                self.layout["header"].update(self._generate_header())
                self.layout["body"].update(self._generate_logs())
                # Always update model info since it's always visible
                self.layout["model_info"].update(self._generate_model_info())
                self.layout["footer"].update(self._generate_footer())
                await asyncio.sleep(0.25)

        self.is_running = False
        dnet_logger.removeHandler(self.log_handler)
