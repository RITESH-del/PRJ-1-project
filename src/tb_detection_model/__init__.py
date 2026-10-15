"""TB Detection Model command-line package."""

from .cli import app


def main() -> None:
    """Run the Typer command application."""
    app()


__all__ = ["app", "main"]
