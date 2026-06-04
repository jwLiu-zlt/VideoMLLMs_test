from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

from .prototype_gate import PrototypeResult, Signal
from .video_utils import VideoClip


@dataclass
class LLMSlowPathRequest:
    clip: VideoClip
    gate_result: PrototypeResult
    prompt: str


@dataclass
class LLMSlowPathResult:
    clip_id: int
    start_time: float
    end_time: float
    decision: str
    success: bool
    prompt: str
    response: str | None
    warning: str | None = None


class MockLLMRunner:
    def __call__(self, request: LLMSlowPathRequest) -> str:
        result = request.gate_result
        return (
            f"clip {result.clip_id} is marked {result.decision.value}. "
            f"deviation={_fmt(result.deviation)}, change={_fmt(result.change)}, "
            f"offset={_fmt(result.offset)}, threshold={_fmt(result.threshold)}. "
            "Use a real LLM backend to replace this diagnostic response."
        )


class HuggingFaceTextLLMRunner:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 128,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def __call__(self, request: LLMSlowPathRequest) -> str:
        pipeline = self._load_pipeline()
        outputs = pipeline(
            request.prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if not outputs:
            return ""
        return str(outputs[0].get("generated_text", "")).strip()

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace LLM backend requires transformers and torch. "
                "Install transformers or use --llm_backend mock."
            ) from exc

        device_arg = self._resolve_device(torch)
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_path,
            device=device_arg,
            torch_dtype="auto",
        )
        return self._pipeline

    def _resolve_device(self, torch_module):
        if self.device == "auto":
            return 0 if torch_module.cuda.is_available() else -1
        if self.device == "cpu":
            return -1
        if self.device.startswith("cuda"):
            if ":" in self.device:
                return int(self.device.split(":", 1)[1])
            return 0
        return self.device


def make_llm_runner(
    backend: str,
    model_path: str | None,
    device: str,
    max_new_tokens: int,
) -> Callable[[LLMSlowPathRequest], str]:
    if backend == "mock":
        return MockLLMRunner()
    if backend == "hf":
        if not model_path:
            raise ValueError("--llm_model_path is required when --llm_backend hf.")
        return HuggingFaceTextLLMRunner(
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


def build_llm_prompt(
    result: PrototypeResult,
    template: str,
    video_label: str,
) -> str:
    values = {
        "video_label": video_label,
        "clip_id": result.clip_id,
        "start_time": _fmt(result.start_time),
        "end_time": _fmt(result.end_time),
        "decision": result.decision.value,
        "change": _fmt(result.change),
        "deviation": _fmt(result.deviation),
        "offset": _fmt(result.offset),
        "threshold": _fmt(result.threshold),
    }
    return template.format(**values)


def run_llm_slow_path(
    clips: Iterable[VideoClip],
    gate_results: Iterable[PrototypeResult],
    runner: Callable[[LLMSlowPathRequest], str],
    prompt_template: str,
    video_label: str,
    trigger_mode: str = "suspicious",
) -> List[LLMSlowPathResult]:
    clip_by_id = {clip.clip_id: clip for clip in clips}
    outputs: List[LLMSlowPathResult] = []

    for result in gate_results:
        if not _should_run_llm(result, trigger_mode):
            continue
        clip = clip_by_id.get(result.clip_id)
        if clip is None:
            outputs.append(_failed_result(result, "", "clip_not_found"))
            continue

        prompt = build_llm_prompt(result, prompt_template, video_label)
        request = LLMSlowPathRequest(clip=clip, gate_result=result, prompt=prompt)
        try:
            response = runner(request)
            outputs.append(
                LLMSlowPathResult(
                    clip_id=result.clip_id,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    decision=result.decision.value,
                    success=True,
                    prompt=prompt,
                    response=response,
                )
            )
        except Exception as exc:
            outputs.append(_failed_result(result, prompt, f"llm_failed: {exc}"))

    return outputs


def _should_run_llm(result: PrototypeResult, trigger_mode: str) -> bool:
    if trigger_mode == "all":
        return True
    if trigger_mode == "suspicious":
        return result.decision == Signal.SUSPICIOUS
    raise ValueError(f"Unsupported LLM trigger mode: {trigger_mode}")


def _failed_result(result: PrototypeResult, prompt: str, warning: str) -> LLMSlowPathResult:
    return LLMSlowPathResult(
        clip_id=result.clip_id,
        start_time=result.start_time,
        end_time=result.end_time,
        decision=result.decision.value,
        success=False,
        prompt=prompt,
        response=None,
        warning=warning,
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.6f}"
