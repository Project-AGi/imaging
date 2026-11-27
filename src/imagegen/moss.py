"""AIVA's raster painting: a client for the MossInfer ``moss-server`` sidecar.

``moss-server`` (from the community MossInfer engine for MOSS-Image) speaks one
JSON message per line over stdin/stdout — protocol version 2 — the same role
``llama-server`` plays for llama.cpp. This module owns one child process:

* ``hello`` negotiates the protocol,
* ``load`` loads the MOSS-Image model once (INT8 components, ``min_memory``
  streaming so a small GPU is enough),
* ``generate`` renders one PNG (the 8-step DMD distill bundle resolves the
  step count and CFG),
* ``release`` frees GPU memory after every painting so the graphics card
  stays available to the chat models.

The server trusts its launcher completely, so every path is produced here and
never taken from model output or the network.
"""
from __future__ import annotations

import json
import math
import os
import selectors
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

ENGINE_ROOT = Path.home() / "mossinfer/mossinfer-2.2.1-linux-cu13-x86_64"
WEIGHTS_ROOT = Path.home() / "AGi/models/mossimage"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 5_900_000


class MossPaintError(RuntimeError):
    """A painting could not be generated."""


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def paint_size(width, height) -> tuple[int, int]:
    """Choose the generation size for a canvas placement.

    Keep a layer within 256..704 pixels per side, on a 16-pixel grid.
    Preserve its aspect ratio where the size bounds allow it. At most 0.5 MP
    keeps decoding, transport and GPU workspace modest on a shared GPU.
    """
    try:
        width, height = float(width), float(height)
    except (TypeError, ValueError):
        return 512, 512
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        return 512, 512
    scale = min(704 / max(width, height), max(1, 256 / min(width, height)))
    return tuple(max(256, min(704, round(value * scale / 16) * 16))
                 for value in (width, height))


class MossClient:
    """Own one ``moss-server`` child; generate PNG paintings on demand."""

    def __init__(self) -> None:
        self.binary = _path_env("MOSS_PAINT_SERVER_BIN", Path(
            shutil.which("moss-server") or ENGINE_ROOT / "bin" / "moss-server"))
        weights = _path_env("MOSS_PAINT_WEIGHTS_ROOT", WEIGHTS_ROOT)
        self.model_dir = _path_env("MOSS_PAINT_MODEL_DIR", weights / "sft")
        self.dit_file = _path_env(
            "MOSS_PAINT_DIT_FILE", weights / "comfy" / "diffusion_models" / "moss_image_sft_int8_convrot.safetensors")
        self.text_encoder_file = _path_env(
            "MOSS_PAINT_TE_FILE", weights / "comfy" / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors")
        self.vae_file = _path_env(
            "MOSS_PAINT_VAE_FILE", weights / "comfy" / "vae" / "flux2-vae.safetensors")
        self.distill_dir = _path_env("MOSS_PAINT_DISTILL_DIR", weights / "distill-8step")
        self.device = os.getenv("MOSS_PAINT_DEVICE", "1")
        self.vram_budget_bytes = int(os.getenv("MOSS_PAINT_VRAM_MIB", "4096")) * 1024 * 1024
        self.timeout = float(os.getenv("MOSS_PAINT_TIMEOUT_SECONDS", "300"))
        self.release_after = os.getenv("MOSS_PAINT_RELEASE", "1") != "0"
        self._process: subprocess.Popen | None = None
        self._buffer = b""
        self._stderr: deque[str] = deque(maxlen=6)
        self._lock = threading.Lock()
        self._process_lock = threading.RLock()
        self._closed = False
        self._next_id = 1
        self._loaded = False

    # -- availability ---------------------------------------------------------

    def available(self) -> bool:
        """True when the engine and the weights are all in place."""
        return (self.binary.is_file() and os.access(self.binary, os.X_OK)
                and (self.model_dir / "model_index.json").is_file()
                and self.dit_file.is_file()
                and self.text_encoder_file.is_file()
                and self.vae_file.is_file()
                and (self.distill_dir / "lora.safetensors").is_file()
                and (self.distill_dir / "time_embedding.safetensors").is_file())

    # -- process handling -----------------------------------------------------

    def _stderr_tail(self) -> str:
        return " ".join(self._stderr)[-600:]

    def _spawn(self) -> None:
        with self._process_lock:
            if self._closed:
                raise MossPaintError("The painting engine is shutting down")
            if self._process is not None and self._process.poll() is None:
                return
            self._dispose()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.device)
            try:
                self._process = subprocess.Popen(
                    [str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env=env)
            except OSError as error:
                raise MossPaintError(f"Cannot start moss-server: {error}") from error
            threading.Thread(target=self._drain_stderr, args=(self._process,), daemon=True).start()
        self._hello()

    def _drain_stderr(self, process: subprocess.Popen) -> None:
        try:
            for raw in iter(process.stderr.readline, b""):
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr.append(text)
        except (OSError, ValueError):
            pass  # shutdown closes the pipe

    def _dispose(self) -> None:
        """Stop and reap the child, including a timed-out or still-generating one."""
        with self._process_lock:
            process, self._process = self._process, None
            self._loaded = False
            self._buffer = b""
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()

    def _send(self, method: str, params: dict | None = None) -> int:
        request_id = self._next_id
        self._next_id += 1
        request = {"id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        try:
            assert self._process is not None and self._process.stdin is not None
            self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError, AssertionError):
            raise MossPaintError(f"moss-server is not accepting requests. {self._stderr_tail()}") from None
        return request_id

    def _read_line(self, process: subprocess.Popen, timeout: float) -> str | None:
        """One protocol line with a timeout; ``None`` when it does not arrive."""
        buffer = self._buffer
        deadline = time.monotonic() + max(0.1, timeout)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._buffer = buffer
                    return None
                if not selector.select(remaining):
                    continue
                chunk = process.stdout.read1(65536)
                if not chunk:
                    raise MossPaintError(f"moss-server stopped. {self._stderr_tail()}")
                buffer += chunk
        line, _, self._buffer = buffer.partition(b"\n")
        return line.decode("utf-8", errors="replace").strip()

    def _call(self, method: str, params: dict | None, timeout: float,
              wait_done: bool = False) -> dict:
        """Send one request; return its result, or its ``done`` event."""
        process = self._process
        if process is None:
            raise MossPaintError("The painting engine is not running")
        request_id = self._send(method, params)
        if timeout <= 0:
            raise MossPaintError("Painting time budget reached")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MossPaintError(f"moss-server did not answer {method} within {timeout:g}s")
            line = self._read_line(process, remaining)
            if line is None:
                raise MossPaintError(f"moss-server did not answer {method} within {timeout:g}s")
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            if message.get("event") == "done":
                if message.get("status") == "cancelled":
                    raise MossPaintError("Painting was cancelled")
                if message.get("status") == "error":
                    error = message.get("error") or {}
                    raise MossPaintError(f'{error.get("code", "error")}: {error.get("message", "")}')
                return message
            if "error" in message:
                error = message["error"]
                raise MossPaintError(f'{error.get("code", "error")}: {error.get("message", "")}')
            if "result" in message:
                if not wait_done:
                    return message

    # -- protocol steps -------------------------------------------------------

    def _hello(self) -> None:
        result = self._call("hello", {"client": "imagegen", "protocol_version": 2}, 30)
        protocol = (result.get("result") or {}).get("protocol_version")
        if protocol != 2:
            raise MossPaintError("moss-server speaks an unexpected protocol version")
        probe = (result.get("result") or {}).get("probe", {})
        if probe.get("driver_compatible") is False or probe.get("device_count") == 0:
            raise MossPaintError(probe.get("message") or "No compatible NVIDIA GPU is available")

    def _load(self, timeout=120) -> None:
        if self._loaded:
            return
        params = {
            "model": str(self.model_dir),
            "tier": "high",
            "mode": "min_memory",
            "device": 0,
            "vram_budget_bytes": self.vram_budget_bytes,
            "dit_file": str(self.dit_file),
            "text_encoder_file": str(self.text_encoder_file),
            "vae_file": str(self.vae_file),
            "distill": str(self.distill_dir),
            "te_host_pin": "auto",
        }
        self._call("load", params, min(120, timeout))
        self._loaded = True

    def _release(self) -> None:
        """Free GPU memory after a painting; the next one reloads from disk."""
        if not self.release_after or self._process is None:
            return
        try:
            self._call("release", {"keep_host_copy": False}, 5)
            # Soft release retains the engine and embedding cache. The next
            # generate reloads its GPU weights without recreating the engine.
        except MossPaintError:
            # A failed release must not leave a GPU allocation behind.
            self._dispose()

    # -- public API -----------------------------------------------------------

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        """Render one painting and return its PNG bytes."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise MossPaintError("The painting prompt is empty")
        output = Path(tempfile.gettempdir()) / f"aiva-paint-{uuid.uuid4().hex}.png"
        with self._lock:
            deadline = time.monotonic() + self.timeout
            try:
                self._spawn()
                self._load(deadline - time.monotonic())
                self._call("generate", {
                    "prompt": prompt[:2000],
                    "width": int(width),
                    "height": int(height),
                    "output": str(output),
                    "save_metadata": False,
                    "include_prompts": False,
                    "distill": str(self.distill_dir),
                }, deadline - time.monotonic(), wait_done=True)
                try:
                    if output.stat().st_size > MAX_PNG_BYTES:
                        raise MossPaintError("The painted PNG exceeds the canvas image limit")
                    png = output.read_bytes()
                    if not png.startswith(PNG_SIGNATURE):
                        raise MossPaintError("The painting engine did not produce a PNG")
                    return png
                except OSError as error:
                    raise MossPaintError(f"The painted PNG could not be read: {error}") from error
            except (MossPaintError, OSError, ValueError) as error:
                # A timeout is not a cancellation: stop the child so it cannot
                # keep generating or leave unread protocol events behind.
                self._dispose()
                if isinstance(error, MossPaintError):
                    raise
                raise MossPaintError(f"Painting engine stopped: {error}") from error
            finally:
                try:
                    output.unlink(missing_ok=True)
                except OSError:
                    pass
                self._release()

    def close(self) -> None:
        """Stop the sidecar even while a painting is in progress."""
        with self._process_lock:
            self._closed = True
            self._dispose()
