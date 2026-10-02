from __future__ import annotations

from typing import Any

from ego2act.vidgen.ego2act_backend import (
    Ego2ActGenerationRunner,
    env_positive_float,
    env_positive_int,
)


class Cosmos3GenerationRunner(Ego2ActGenerationRunner):
    """Cosmos 3 adapter for the pinned Ego2Act inference backend."""

    runner_id = "cosmos3"
    fps = 24
    inference_steps = 35

    def __init__(
        self,
        alias: str,
        model_id: str,
        backend_url: str | None = None,
    ) -> None:
        super().__init__(alias, model_id, backend_url)
        self.width = env_positive_int("COSMOS3_WIDTH")
        self.height = env_positive_int("COSMOS3_HEIGHT")
        if (self.width is None) != (self.height is None):
            raise ValueError("COSMOS3_WIDTH and COSMOS3_HEIGHT must be set together")
        self.fps = env_positive_int("COSMOS3_FPS", self.fps)
        self.inference_steps = env_positive_int(
            "COSMOS3_STEPS", self.inference_steps
        )
        self.guidance_scale = env_positive_float("COSMOS3_GUIDANCE_SCALE", 4.0)
        self.flow_shift = env_positive_float("COSMOS3_FLOW_SHIFT", 10.0)

    def build_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = super().build_request(payload)
        if self.width is not None and self.height is not None:
            request["options"]["width"] = self.width
            request["options"]["height"] = self.height
        return request

    def extra_options(self) -> dict[str, Any]:
        return {"guidance_scale": self.guidance_scale}

    def provider_options(self) -> dict[str, Any]:
        return {
            "request_fields": {"flow_shift": self.flow_shift},
            "extra_params": {
                "guardrails": False,
                "use_resolution_template": False,
                "use_duration_template": False,
            },
        }

