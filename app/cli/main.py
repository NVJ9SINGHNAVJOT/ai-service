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

audio_app = typer.Typer(help="Prepare local speech (STT / TTS) models.", no_args_is_help=True)
cli.add_typer(audio_app, name="audio")

console = Console()
err_console = Console(stderr=True, style="bold red")


# ── Helper ───────────────────────────────────────────────────────────────────

def _abort(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=code)


def _style_for_state(state: str) -> str:
    """Return a rich style name for a model lifecycle state."""
    return {
        "ready": "green",
        "downloading": "yellow",
        "running": "bold cyan",
        "unsupported": "red",
        "incomplete": "red",
    }.get(state, "white")


def _style_for_input_kind(kind: str) -> str:
    """Return a rich style name for a model input modality."""
    return {
        "text": "green",
        "image": "bold cyan",
        "audio": "magenta",
    }.get(kind, "white")


def _format_input_modalities(kinds: list[str]) -> str:
    """Render input modalities as colored rich labels."""
    if not kinds:
        return "—"
    return ", ".join(
        f"[{_style_for_input_kind(kind)}]{kind}[/{_style_for_input_kind(kind)}]"
        for kind in kinds
    )


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
    table.add_column("Backend", style="blue")
    table.add_column("Modalities")
    table.add_column("Loadable", justify="center")
    table.add_column("Size (MB)", justify="right")
    table.add_column("HF Repo", style="dim")
    table.add_column("Updated", style="dim")

    for m in model_list:
        state_style = _style_for_state(m.state.value)
        loadable_icon = "✓" if m.loadable else "✗"
        loadable_style = "green" if m.loadable else "red"
        updated = m.updated_at.strftime("%Y-%m-%d") if m.updated_at else "—"
        table.add_row(
            m.name,
            m.source.value,
            f"[{state_style}]{m.state.value}[/{state_style}]",
            m.backend,
            _format_input_modalities(m.input_modalities),
            f"[{loadable_style}]{loadable_icon}[/{loadable_style}]",
            str(m.size_mb) if m.size_mb is not None else "—",
            m.repo_id or "—",
            updated,
        )

    console.print(table)


@models_app.command("doctor")
def models_doctor(
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Inspect one local model. If omitted, inspects all local models.",
    ),
) -> None:
    """Diagnose local model readiness and common runtime issues."""
    from app.core.exceptions import ModelNotFoundError, RegistryError
    from app.services.model_manager import ModelManager

    manager = ModelManager()
    try:
        if name:
            diagnosis = manager.diagnose_model(name)
            table = Table(title=f"Model Doctor: {diagnosis.name}", show_lines=True)
            table.add_column("Field", style="cyan", no_wrap=True)
            table.add_column("Value")
            table.add_row("Name", diagnosis.name)
            table.add_row("Source", diagnosis.source.value)
            table.add_row("State", f"[{_style_for_state(diagnosis.state.value)}]{diagnosis.state.value}[/{_style_for_state(diagnosis.state.value)}]")
            table.add_row("Loadable", "[green]✓[/green]" if diagnosis.loadable else "[red]✗[/red]")
            table.add_row("HF Repo", diagnosis.repo_id or "—")
            table.add_row("Model Type", diagnosis.model_type or "—")
            table.add_row("MLX Architecture", diagnosis.effective_model_type or "—")
            table.add_row("Backend", diagnosis.backend)
            table.add_row("Inputs", _format_input_modalities(diagnosis.input_modalities))
            table.add_row(
                "Supported By MLX",
                "✓" if diagnosis.supported_by_mlx is True else ("✗" if diagnosis.supported_by_mlx is False else "—"),
            )
            table.add_row("Path", diagnosis.path)
            table.add_row("Diagnosis", diagnosis.summary)
            table.add_row("Missing Files", ", ".join(diagnosis.missing_files) if diagnosis.missing_files else "—")
            console.print(table)
            if diagnosis.recommendations:
                console.print("\n[bold]Recommendations[/bold]")
                for item in diagnosis.recommendations:
                    console.print(f"- {item}")
            return

        diagnoses = manager.diagnose_models()
    except ModelNotFoundError as exc:
        _abort(str(exc))
    except RegistryError as exc:
        _abort(str(exc))

    if not diagnoses:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title="Model Doctor", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("State")
    table.add_column("Loadable", justify="center")
    table.add_column("Model Type")
    table.add_column("Backend")
    table.add_column("Inputs")
    table.add_column("MLX Support", justify="center")
    table.add_column("Diagnosis")

    for diagnosis in diagnoses:
        support = "✓" if diagnosis.supported_by_mlx is True else ("✗" if diagnosis.supported_by_mlx is False else "—")
        table.add_row(
            diagnosis.name,
            f"[{_style_for_state(diagnosis.state.value)}]{diagnosis.state.value}[/{_style_for_state(diagnosis.state.value)}]",
            "[green]✓[/green]" if diagnosis.loadable else "[red]✗[/red]",
            diagnosis.model_type or "—",
            diagnosis.backend,
            _format_input_modalities(diagnosis.input_modalities),
            support,
            diagnosis.summary,
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
    """Re-download a model to get the latest version."""
    from app.services.model_manager import ModelManager
    from app.core.exceptions import DownloadError, ModelBusyError, ModelNotFoundError, InvalidModelPathError

    manager = ModelManager()
    console.print(f"[dim]Updating model [cyan]{name}[/cyan]…[/dim]")
    try:
        info = manager.update(name)
        console.print(f"[bold green]✓[/bold green] Model updated: [cyan]{info.name}[/cyan]")
    except ModelNotFoundError as exc:
        _abort(str(exc))
    except InvalidModelPathError as exc:
        _abort(str(exc))
    except ModelBusyError as exc:
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
    from app.core.exceptions import ModelBusyError, ModelNotFoundError, InvalidModelPathError

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
    except ModelBusyError as exc:
        _abort(str(exc))


# ── audio prepare ──────────────────────────────────────────────────────────────

@audio_app.command("prepare")
def audio_prepare() -> None:
    """
    Pre-download the speech model weights into the HuggingFace cache.

    Run once (e.g. from `task audio:setup`) so the first voice request doesn't
    block while a multi-GB Whisper download happens. The repos come from
    settings: STT_MODEL (Whisper) and TTS_MODEL (Kokoro).
    """
    from huggingface_hub import snapshot_download

    targets = [("STT (Whisper)", settings.stt_model), ("TTS (Kokoro)", settings.tts_model)]
    for label, repo in targets:
        console.print(f"[dim]Downloading {label} model [cyan]{repo}[/cyan] …[/dim]")
        try:
            snapshot_download(repo_id=repo, token=settings.hf_token)
        except Exception as exc:
            _abort(f"Failed to download {label} model '{repo}': {exc}")
        console.print(f"[bold green]✓[/bold green] {label} ready: [cyan]{repo}[/cyan]")

    console.print("[bold green]Speech models ready.[/bold green]")


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
    from app.core.exceptions import InvalidModelPathError, ModelNotFoundError, UnsupportedModelError

    from app.services.media_chat_session import MediaChatSession

    manager = ModelManager()
    try:
        raw_info = manager.get_model(model)
    except ModelNotFoundError:
        _abort(
            f"Model '{model}' not found.\n"
            "  Run `python -m app.cli.main models list` to see available models."
        )

    is_vlm = raw_info.backend == "mlx-vlm"

    try:
        if is_vlm:
            info = manager.ensure_model_files_ready(model)
        else:
            info = manager.ensure_model_loadable(model)
    except (InvalidModelPathError, UnsupportedModelError) as exc:
        _abort(str(exc))

    if is_vlm:
        console.print(f"[dim]'{model}' — backend: mlx-vlm | accepts: {', '.join(info.input_modalities)}[/dim]")
        session = MediaChatSession(
            model_path=Path(info.path),
            model_name=info.name,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            allowed_modalities=info.input_modalities,
            verbose=verbose,
        )
        try:
            session.run()
        except Exception as exc:
            _abort(str(exc))
    else:
        console.print(f"[dim]'{model}' — backend: mlx-lm | accepts: {', '.join(info.input_modalities)}[/dim]")
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


@cli.command("chat-media")
def chat_media(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Local (sanitised) model name to use for media prompting.",
    ),
    image: Optional[Path] = typer.Option(
        None,
        "--image",
        "-i",
        exists=False,
        help="Optional path to an image file to preload before chat starts.",
    ),
    audio: Optional[Path] = typer.Option(
        None,
        "--audio",
        "-a",
        exists=False,
        help="Optional path to an audio file to preload before chat starts.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="Optional system prompt to use for the media chat session.",
    ),
    max_tokens: int = typer.Option(
        settings.default_max_tokens,
        "--max-tokens",
        help="Maximum number of tokens to generate.",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        help="Sampling temperature.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print raw streamed text instead of panel output when supported by mlx-vlm.",
    ),
) -> None:
    """Start an interactive media chat session via mlx-vlm."""
    from app.core.exceptions import InvalidModelPathError, MediaChatError, ModelNotFoundError
    from app.services.model_manager import ModelManager
    from app.services.media_chat_session import MediaChatSession

    manager = ModelManager()
    try:
        info = manager.ensure_model_files_ready(model)
    except ModelNotFoundError:
        _abort(
            f"Model '{model}' not found.\n"
            "  Run `python -m app.cli.main models list` to see available models."
        )
    except InvalidModelPathError as exc:
        _abort(str(exc))

    supported_inputs = set(info.input_modalities)
    if image is not None and "image" not in supported_inputs:
        _abort(f"Model '{model}' does not appear to support image input.")
    if audio is not None and "audio" not in supported_inputs:
        _abort(f"Model '{model}' does not appear to support audio input.")
    if not ({"image", "audio"} & supported_inputs):
        _abort(
            f"Model '{model}' does not appear to support image or audio input.\n"
            "  Use `python -m app.cli.main chat --model ...` for text-only chat."
        )

    session = MediaChatSession(
        model_path=Path(info.path),
        model_name=info.name,
        image_path=image,
        audio_path=audio,
        allowed_modalities=info.input_modalities,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        verbose=verbose,
    )
    try:
        session.run()
    except MediaChatError as exc:
        _abort(str(exc))


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
