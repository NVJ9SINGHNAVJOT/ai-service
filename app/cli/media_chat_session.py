"""
Interactive media chat via mlx-vlm.

This session mirrors the repo's normal terminal chat UX while delegating the
actual multimodal generation work to mlx-vlm.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.core.exceptions import MediaChatError
from app.core.logging import get_logger
from app.patches import patch_gemma4_shared_kv_load
from app.services.model_runtime_state import ModelRuntimeState

logger = get_logger(__name__)
console = Console()


class MediaChatSession:
    """Run an interactive image/audio + text chat using mlx-vlm."""

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        image_path: Optional[Path] = None,
        audio_path: Optional[Path] = None,
        allowed_modalities: Optional[Sequence[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.image_path = image_path
        self.audio_path = audio_path
        self.allowed_modalities = set(allowed_modalities or [])
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.verbose = verbose
        self._runtime_state = ModelRuntimeState()

        self._model: Any = None
        self._processor: Any = None
        self._apply_chat_template: Any = None
        self._stream_generate: Any = None
        self._load_image: Any = None
        self._prompt_cache_state: Any = None
        self._vision_cache: Any = None
        self._last_load_duration_s: Optional[float] = None

        self._history: List[dict] = []
        self._current_image_path: Optional[str] = str(image_path) if image_path else None
        self._current_audio_path: Optional[str] = str(audio_path) if audio_path else None

    def run(self) -> None:
        """Load the model and start the interactive media chat loop."""
        self._validate()
        marker = self._runtime_state.mark_running(self.model_name)
        try:
            self._load_runtime()
            self._load_model()
            self._print_intro()
            if self.system_prompt:
                self._history.append(self._message("system", self.system_prompt))
            if self.image_path is not None:
                self._set_image(self.image_path)
            if self.audio_path is not None:
                self._set_audio(self.audio_path)
            self._loop()
        finally:
            self._runtime_state.clear_marker(marker)

    def _validate(self) -> None:
        """Check local prerequisites before launching mlx-vlm."""
        if importlib.util.find_spec("mlx_vlm") is None:
            raise MediaChatError(
                "`mlx-vlm` is not installed. Run `python -m pip install -U mlx-vlm` first."
            )
        if not self.model_path.exists():
            raise MediaChatError(f"Model path does not exist: {self.model_path}")
        if self.image_path is not None:
            self._validate_modality_allowed("image")
            if not self.image_path.exists():
                raise MediaChatError(f"Image path does not exist: {self.image_path}")
            if not self.image_path.is_file():
                raise MediaChatError(f"Image path is not a file: {self.image_path}")
        if self.audio_path is not None:
            self._validate_modality_allowed("audio")
            if not self.audio_path.exists():
                raise MediaChatError(f"Audio path does not exist: {self.audio_path}")
            if not self.audio_path.is_file():
                raise MediaChatError(f"Audio path is not a file: {self.audio_path}")

    def _load_runtime(self) -> None:
        """Import mlx-vlm components lazily so startup errors are clearer."""
        try:
            mlx_vlm = importlib.import_module("mlx_vlm")
            generate_mod = importlib.import_module("mlx_vlm.generate")
            prompt_utils_mod = importlib.import_module("mlx_vlm.prompt_utils")
            utils_mod = importlib.import_module("mlx_vlm.utils")
            vision_cache_mod = importlib.import_module("mlx_vlm.vision_cache")
        except Exception as exc:
            raise MediaChatError(f"Failed to import mlx-vlm runtime: {exc}") from exc

        self._mlx_vlm_load = mlx_vlm.load
        self._stream_generate = generate_mod.stream_generate
        self._apply_chat_template = prompt_utils_mod.apply_chat_template
        self._load_image = utils_mod.load_image
        self._prompt_cache_state = generate_mod.PromptCacheState()
        self._vision_cache = vision_cache_mod.VisionFeatureCache()

        patch_gemma4_shared_kv_load()  # see app/patches + PATCHES.md

    def _load_model(self) -> None:
        """Load the selected multimodal model."""
        console.print("[dim]Loading media model…[/dim]")
        try:
            started_at = time.perf_counter()
            self._model, self._processor = self._mlx_vlm_load(str(self.model_path))
            self._last_load_duration_s = time.perf_counter() - started_at
        except Exception as exc:
            raise MediaChatError(f"Failed to load media model: {exc}") from exc
        console.print("[bold green]✓ Media model loaded.[/bold green]\n")

    def _print_intro(self) -> None:
        """Render a compact session header."""
        image_label = str(self.image_path) if self.image_path else "not loaded yet"
        audio_label = str(self.audio_path) if self.audio_path else "not loaded yet"
        console.print(
            Panel(
                f"[bold green]AI Service Media Chat[/bold green]\n"
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                f"Image: [cyan]{image_label}[/cyan]\n"
                f"Audio: [cyan]{audio_label}[/cyan]\n"
                f"Type [bold]exit[/bold], [bold]quit[/bold], or [bold]/exit[/bold] to end.\n"
                f"Use [bold]/image <path>[/bold] or [bold]/audio <path>[/bold] to load media.",
                expand=False,
            )
        )

    def _loop(self) -> None:
        """Main input/response loop."""
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "/exit"}:
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.startswith("/"):
                if not self._handle_command(user_input):
                    break
                continue

            self._history.append(self._message("user", user_input))
            try:
                response, usage = self._generate_response()
            except MediaChatError as exc:
                logger.error("Media generation failed with '%s'", self.model_name, exc_info=exc)
                console.print(f"[bold red]Error:[/bold red] {exc}")
                self._history.pop()
                continue

            self._history.append(self._message("assistant", response))
            if self.verbose:
                console.print()
                self._print_verbose_stats(usage)
            console.print()

    def _handle_command(self, user_input: str) -> bool:
        """Handle slash commands. Returns False when the session should end."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/help":
            self._print_help()
            return True
        if command == "/clear":
            self._history = [self._message("system", self.system_prompt)] if self.system_prompt else []
            self._prompt_cache_state = type(self._prompt_cache_state)()
            console.print("[dim]Conversation cleared.[/dim]")
            return True
        if command == "/image":
            if not arg:
                console.print("[bold red]Error:[/bold red] Please provide an image path.")
                return True
            try:
                self._set_image(Path(arg))
            except MediaChatError as exc:
                logger.warning("Failed to load image '%s': %s", arg, exc, exc_info=exc)
                console.print(f"[bold red]Error:[/bold red] {exc}")
            return True
        if command == "/audio":
            if not arg:
                console.print("[bold red]Error:[/bold red] Please provide an audio path.")
                return True
            try:
                self._set_audio(Path(arg))
            except MediaChatError as exc:
                logger.warning("Failed to load audio '%s': %s", arg, exc, exc_info=exc)
                console.print(f"[bold red]Error:[/bold red] {exc}")
            return True
        if command == "/exit":
            console.print("[dim]Goodbye![/dim]")
            return False

        console.print(f"[bold red]Error:[/bold red] Unknown command: {command}")
        return True

    def _print_help(self) -> None:
        """Show supported slash commands."""
        console.print(
            Panel(
                "Commands:\n"
                "/image <path>  Load or switch the active image\n"
                "/audio <path>  Load or switch the active audio clip\n"
                "/clear         Clear conversation history\n"
                "/help          Show this help\n"
                "/exit          End the session",
                title="Media Chat Help",
                expand=False,
            )
        )

    def _set_image(self, image_path: Path) -> None:
        """Validate and switch the active image."""
        self._validate_modality_allowed("image")
        if not image_path.exists():
            raise MediaChatError(f"Image path does not exist: {image_path}")
        if not image_path.is_file():
            raise MediaChatError(f"Image path is not a file: {image_path}")
        try:
            self._load_image(str(image_path))
        except Exception as exc:
            raise MediaChatError(f"Failed to load image: {exc}") from exc
        self._current_image_path = str(image_path)
        console.print(f"[bold blue]Loaded image:[/bold blue] {image_path}")

    def _set_audio(self, audio_path: Path) -> None:
        """Validate and switch the active audio clip."""
        self._validate_modality_allowed("audio")
        if not audio_path.exists():
            raise MediaChatError(f"Audio path does not exist: {audio_path}")
        if not audio_path.is_file():
            raise MediaChatError(f"Audio path is not a file: {audio_path}")
        self._current_audio_path = str(audio_path)
        console.print(f"[bold magenta]Loaded audio:[/bold magenta] {audio_path}")

    def _validate_modality_allowed(self, modality: str) -> None:
        """Fail early when the selected model does not advertise a media type."""
        if modality not in self.allowed_modalities:
            raise MediaChatError(
                f"Model '{self.model_name}' does not appear to support {modality} input."
            )

    def _generate_response(self) -> tuple[str, Optional[dict]]:
        """Generate one assistant reply while streaming tokens to the terminal."""
        num_images = 1 if self._current_image_path else 0
        num_audios = 1 if self._current_audio_path else 0
        prompt = self._apply_chat_template(
            self._processor,
            self._model.config,
            self._history,
            num_images=num_images,
            num_audios=num_audios,
        )

        console.print("\n[bold green]Assistant[/bold green]:")
        chunks: list[str] = []
        usage: Optional[dict] = None
        started_at = time.perf_counter()
        first_token_at: Optional[float] = None
        last_chunk: Optional[Any] = None
        try:
            for chunk in self._stream_generate(
                self._model,
                self._processor,
                prompt,
                image=[self._current_image_path] if self._current_image_path else None,
                audio=[self._current_audio_path] if self._current_audio_path else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                vision_cache=self._vision_cache,
                prompt_cache_state=self._prompt_cache_state,
            ):
                last_chunk = chunk
                if chunk.text:
                    chunks.append(chunk.text)
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    console.print(chunk.text, end="", markup=False, highlight=False)
                if getattr(chunk, "finish_reason", None) is not None:
                    finished_at = time.perf_counter()
                    prompt_tokens = int(getattr(chunk, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(chunk, "generation_tokens", 0) or 0)
                    prompt_eval_duration = max((first_token_at or finished_at) - started_at, 0.0)
                    eval_duration = max(finished_at - (first_token_at or finished_at), 0.0)
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "finish_reason": getattr(chunk, "finish_reason", "stop"),
                        "metrics": {
                            "total_duration_s": finished_at - started_at,
                            "prompt_eval_duration_s": prompt_eval_duration,
                            "prompt_eval_rate": getattr(chunk, "prompt_tps", None),
                            "eval_duration_s": eval_duration,
                            "eval_rate": getattr(chunk, "generation_tps", None),
                        },
                    }
        except Exception as exc:
            raise MediaChatError(str(exc)) from exc

        if usage is None:
            finished_at = time.perf_counter()
            prompt_tokens = getattr(last_chunk, "prompt_tokens", None) if last_chunk is not None else None
            completion_tokens = getattr(last_chunk, "generation_tokens", None) if last_chunk is not None else None
            prompt_eval_duration = max((first_token_at or finished_at) - started_at, 0.0)
            eval_duration = max(finished_at - (first_token_at or finished_at), 0.0)
            usage = {
                "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
                "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
                "total_tokens": (
                    int(prompt_tokens) + int(completion_tokens)
                    if prompt_tokens is not None and completion_tokens is not None
                    else None
                ),
                "finish_reason": getattr(last_chunk, "finish_reason", None) if last_chunk is not None else None,
                "metrics": {
                    "total_duration_s": finished_at - started_at,
                    "prompt_eval_duration_s": prompt_eval_duration,
                    "prompt_eval_rate": getattr(last_chunk, "prompt_tps", None) if last_chunk is not None else None,
                    "eval_duration_s": eval_duration,
                    "eval_rate": getattr(last_chunk, "generation_tps", None) if last_chunk is not None else None,
                },
            }

        return "".join(chunks).replace("<end_of_utterance>", ""), usage

    @staticmethod
    def _message(role: str, text: str) -> dict:
        """Return one mlx-vlm chat message payload."""
        return {
            "role": role,
            "content": [{"type": "text", "text": text}],
        }

    def _print_verbose_stats(self, usage: Optional[dict]) -> None:
        """Print a compact inference summary similar to the text chat session."""
        usage = usage or {}
        metrics = usage.get("metrics") or {}

        lines = [
            f"total duration:       {self._format_duration(metrics.get('total_duration_s'))}",
            f"load duration:        {self._format_duration(self._last_load_duration_s)}",
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
