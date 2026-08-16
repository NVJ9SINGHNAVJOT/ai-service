"""
Interactive terminal chat session.

Manages an in-process conversation loop that reuses a single loaded model.
Streams tokens if mlx_lm supports it; falls back to full-response mode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.core.exceptions import InferenceError, ModelLoadError
from app.core.logging import get_logger
from app.schemas.inference import ChatMessage, Role
from app.services.inference import InferenceService

logger = get_logger(__name__)
console = Console()


class ChatSession:
    """
    Manages an interactive terminal chat loop.

    Usage:
        session = ChatSession(model_path, model_name, system_prompt)
        session.run()
    """

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.verbose = verbose

        self._svc = InferenceService()
        self._history: List[ChatMessage] = []

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive chat loop. Exits on 'exit', 'quit', or Ctrl+C."""
        console.print(
            Panel(
                f"[bold green]AI Core Chat[/bold green]\n"
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                f"Type [bold]exit[/bold] or [bold]quit[/bold] to end the session.",
                expand=False,
            )
        )

        # Load model
        console.print(f"[dim]Loading model…[/dim]")
        try:
            self._svc.load(self.model_path, self.model_name)
        except ModelLoadError as exc:
            logger.error("Failed to load model '%s'", self.model_name, exc_info=exc)
            console.print(f"[bold red]Failed to load model:[/bold red] {exc}")
            sys.exit(1)

        console.print(f"[bold green]✓ Model loaded.[/bold green]\n")

        # Add system prompt to history
        self._history = [ChatMessage(role=Role.system, content=self.system_prompt)]

        try:
            self._loop()
        except KeyboardInterrupt:
            console.print("\n[dim]Session ended.[/dim]")
        finally:
            self._svc.unload()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Main input/response loop."""
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]Goodbye![/dim]")
                break

            self._history.append(ChatMessage(role=Role.user, content=user_input))

            console.print("\n[bold green]Assistant[/bold green]:")
            try:
                response, usage, interrupted = self._stream_response()
            except InferenceError as exc:
                logger.error("Inference failed during chat with '%s'", self.model_name, exc_info=exc)
                console.print(f"[bold red]Error:[/bold red] {exc}")
                # Remove the failed user message so the conversation stays clean
                self._history.pop()
                continue

            console.print()
            if interrupted:
                if response:
                    self._history.append(ChatMessage(role=Role.assistant, content=response))
                else:
                    # Remove the user message if nothing was generated before interruption.
                    self._history.pop()
                console.print("[yellow]Generation stopped. You can keep chatting.[/yellow]\n")
                continue

            if self.verbose:
                self._print_verbose_stats(usage)
            console.print()

            # Append assistant reply to history
            self._history.append(ChatMessage(role=Role.assistant, content=response))

    def _stream_response(self) -> tuple[str, Optional[dict], bool]:
        """
        Stream one assistant reply.

        Returns ``(response_text, usage, interrupted)`` so callers can decide
        whether to keep the partial response when the user presses Ctrl+C.
        """
        chunks: list[str] = []
        usage: Optional[dict] = None
        try:
            for chunk, _usage in self._svc.chat_stream(
                messages=self._history,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            ):
                chunks.append(chunk)
                if _usage is not None:
                    usage = _usage
                if chunk:
                    console.print(chunk, end="", markup=False, highlight=False)
        except KeyboardInterrupt:
            return "".join(chunks), usage, True

        return "".join(chunks), usage, False

    def _print_verbose_stats(self, usage: Optional[dict]) -> None:
        """Print a compact inference summary similar to local LLM CLIs."""
        usage = usage or {}
        metrics = usage.get("metrics") or {}

        lines = [
            f"total duration:       {self._format_duration(metrics.get('total_duration_s'))}",
            f"load duration:        {self._format_duration(self._svc.last_load_duration_s)}",
            f"prompt eval count:    {self._format_token_count(usage.get('prompt_tokens'))}",
            f"prompt eval duration: {self._format_duration(metrics.get('prompt_eval_duration_s'))}",
            f"prompt eval rate:     {self._format_rate(metrics.get('prompt_eval_rate'))}",
            f"eval count:           {self._format_token_count(usage.get('completion_tokens'))}",
            f"eval duration:        {self._format_duration(metrics.get('eval_duration_s'))}",
            f"eval rate:            {self._format_rate(metrics.get('eval_rate'))}",
        ]
        console.print("\n".join(lines), style="dim")

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        if seconds is None:
            return "n/a"
        if seconds < 1:
            return f"{seconds * 1000:.6f}ms"
        return f"{seconds:.9f}s"

    @staticmethod
    def _format_token_count(count: Optional[int]) -> str:
        return f"{count} token(s)" if count is not None else "n/a"

    @staticmethod
    def _format_rate(rate: Optional[float]) -> str:
        return f"{rate:.2f} tokens/s" if rate is not None else "n/a"
