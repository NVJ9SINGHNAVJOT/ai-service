"""
CLI entry point.

Usage examples:
    python -m app.cli.main --help
    python -m app.cli.main models list
    python -m app.cli.main models download --repo mlx-community/Llama-3.2-3B-Instruct-4bit
    python -m app.cli.main models update --name mlx-community__Llama-3.2-3B-Instruct-4bit
    python -m app.cli.main models delete --name mlx-community__Llama-3.2-3B-Instruct-4bit
    python -m app.cli.main chat --model mlx-community__Llama-3.2-3B-Instruct-4bit
    python -m app.cli.main serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from app.config import settings
from app.core.logging import setup_logging

setup_logging()

# ── Typer app tree ───────────────────────────────────────────────────────────

cli = typer.Typer(
    name="ai-service",
    help="AI Service local model manager and inference server.",
    no_args_is_help=True,
)

models_app = typer.Typer(help="Manage local AI models.", no_args_is_help=True)
cli.add_typer(models_app, name="models")

console = Console()
err_console = Console(stderr=True, style="bold red")


# ── Helper ───────────────────────────────────────────────────────────────────

def _abort(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=code)


# ── models list ──────────────────────────────────────────────────────────────

@models_app.command("list")
def models_list() -> None:
    """List all locally available models (downloaded + custom)."""
    from app.services.model_manager import ModelManager
    from app.core.exceptions import RegistryError

    manager = ModelManager()
    try:
        model_list = manager.list_models()
    except RegistryError as exc:
        _abort(str(exc))

    if not model_list:
        console.print("[yellow]No models found.[/yellow]")
        console.print(
            f"Downloaded models go in: [cyan]{settings.downloaded_models_path}[/cyan]\n"
            f"Custom models go in:     [cyan]{settings.custom_models_path}[/cyan]"
        )
        return

    table = Table(title="Local AI Models", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Source", style="magenta")
    table.add_column("State")
    table.add_column("Loadable", justify="center")
    table.add_column("Size (MB)", justify="right")
    table.add_column("HF Repo", style="dim")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")

    for m in model_list:
        state_style = {
            "ready": "green",
            "downloading": "yellow",
            "running": "bold cyan",
            "incomplete": "red",
        }.get(m.state.value, "white")
        loadable_icon = "✓" if m.loadable else "✗"
        loadable_style = "green" if m.loadable else "red"
        created = m.created_at.strftime("%Y-%m-%d") if m.created_at else "—"
        updated = m.updated_at.strftime("%Y-%m-%d") if m.updated_at else "—"
        table.add_row(
            m.name,
            m.source.value,
            f"[{state_style}]{m.state.value}[/{state_style}]",
            f"[{loadable_style}]{loadable_icon}[/{loadable_style}]",
            str(m.size_mb) if m.size_mb is not None else "—",
            m.repo_id or "—",
            created,
            updated,
        )

    console.print(table)


# ── models download ──────────────────────────────────────────────────────────

@models_app.command("download")
def models_download(
    repo: str = typer.Option(
        ...,
        "--repo",
        "-r",
        help="HuggingFace repo ID, e.g. mlx-community/Llama-3.2-3B-Instruct-4bit",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if the model already exists locally.",
    ),
) -> None:
    """Download an MLX-compatible model from HuggingFace."""
    from app.services.model_manager import ModelManager
    from app.core.exceptions import DownloadError, ModelAlreadyExistsError

    manager = ModelManager()
    console.print(f"[dim]Downloading [cyan]{repo}[/cyan]…[/dim]")
    try:
        info = manager.download(repo, force=force)
        console.print(f"[bold green]✓[/bold green] Model downloaded: [cyan]{info.name}[/cyan]")
        console.print(f"  Path: {info.path}")
    except ModelAlreadyExistsError as exc:
        _abort(f"{exc}\n  Tip: use --force to re-download.")
    except DownloadError as exc:
        _abort(str(exc))


# ── models update ────────────────────────────────────────────────────────────

@models_app.command("update")
def models_update(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Local (sanitised) model name to update.",
    ),
) -> None:
    """Re-download a model to get the latest version (delete + re-download)."""
    from app.services.model_manager import ModelManager
    from app.core.exceptions import DownloadError, ModelNotFoundError, InvalidModelPathError

    manager = ModelManager()
    console.print(f"[dim]Updating model [cyan]{name}[/cyan]…[/dim]")
    try:
        info = manager.update(name)
        console.print(f"[bold green]✓[/bold green] Model updated: [cyan]{info.name}[/cyan]")
    except ModelNotFoundError as exc:
        _abort(str(exc))
    except InvalidModelPathError as exc:
        _abort(str(exc))
    except DownloadError as exc:
        _abort(str(exc))


# ── models delete ────────────────────────────────────────────────────────────

@models_app.command("delete")
def models_delete(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Local (sanitised) model name to delete.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
    allow_custom: bool = typer.Option(
        False,
        "--allow-custom",
        help="Also allow deletion of custom models.",
    ),
) -> None:
    """Delete a local model. Only downloaded models are deleted by default."""
    from app.services.model_manager import ModelManager
    from app.core.exceptions import ModelNotFoundError, InvalidModelPathError

    manager = ModelManager()

    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to permanently delete '{name}'?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            raise typer.Exit()

    try:
        manager.delete(name, allow_custom=allow_custom)
        console.print(f"[bold green]✓[/bold green] Model [cyan]{name}[/cyan] deleted.")
    except ModelNotFoundError as exc:
        _abort(str(exc))
    except InvalidModelPathError as exc:
        _abort(str(exc))


# ── chat ─────────────────────────────────────────────────────────────────────

@cli.command("chat")
def chat(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Local (sanitised) model name to chat with.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt to use for the session.",
    ),
    max_tokens: int = typer.Option(
        settings.default_max_tokens,
        "--max-tokens",
        help="Maximum tokens per response.",
    ),
    temperature: float = typer.Option(
        settings.default_temperature,
        "--temperature",
        help="Sampling temperature.",
    ),
    top_p: float = typer.Option(
        settings.default_top_p,
        "--top-p",
        help="Nucleus sampling probability.",
    ),
    repetition_penalty: float = typer.Option(
        settings.default_repetition_penalty,
        "--repetition-penalty",
        help="Repetition penalty factor.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print token and timing stats after each assistant reply.",
    ),
) -> None:
    """Start an interactive terminal chat session with a local model."""
    from app.services.model_manager import ModelManager
    from app.services.chat_session import ChatSession
    from app.core.exceptions import ModelNotFoundError

    manager = ModelManager()
    try:
        info = manager.get_model(model)
    except ModelNotFoundError:
        _abort(
            f"Model '{model}' not found.\n"
            "  Run `python -m app.cli.main models list` to see available models."
        )

    session = ChatSession(
        model_path=Path(info.path),
        model_name=info.name,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        verbose=verbose,
    )
    session.run()


# ── serve ─────────────────────────────────────────────────────────────────────

@cli.command("serve")
def serve(
    host: str = typer.Option(settings.api_host, "--host", help="Bind host."),
    port: int = typer.Option(settings.api_port, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)."),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level."),
) -> None:
    """Start the FastAPI inference server."""
    try:
        import uvicorn
    except ImportError:
        _abort("uvicorn is not installed. Run: pip install uvicorn[standard]")

    console.print(f"[bold green]Starting AI Service API[/bold green] on [cyan]http://{host}:{port}[/cyan]")
    console.print(f"  Docs: [link]http://{host}:{port}/docs[/link]")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
