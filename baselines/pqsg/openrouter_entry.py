from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from openai import OpenAI

from baselines.video import standardize_video_for_inference
from baselines.common import UsageRecorder, write_json


def video_data_url(path: str, cache: Path) -> str:
    source = Path(path).resolve()
    key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    proxy, _ = standardize_video_for_inference(source, cache / key)
    suffix = proxy.suffix.lower()
    mime = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
    }.get(suffix, "video/mp4")
    encoded = base64.b64encode(proxy.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--media-cache", type=Path, required=True)
    parser.add_argument("--usage-output", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    import pqsg
    from pqsg.qa import (
        _build_answers_dict,
        _flatten_questions,
        _parse_initial,
        _parse_yesno,
    )
    from pqsg.qg import _extract_json, _load_prompt_template

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=180,
        max_retries=2,
    )
    usage = UsageRecorder()

    def call(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            extra_body={"provider": {"require_parameters": True}},
        )
        usage.add(response)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenRouter returned an empty response")
        return text

    def generate_psg(prompt: str, model: str = "", api_key: str | None = None) -> dict:
        template = _load_prompt_template()
        full_prompt = (
            f"{template}\n\nNow produce a similar JSON for this new prompt "
            f"and reference video:\n\nPrompt: {prompt}"
        )
        text = call([{"role": "user", "content": full_prompt}])
        return json.loads(_extract_json(text))

    def answer_psg_gemini(
        video_path: str,
        psg: dict,
        model: str = "",
        fps: int = 24,
        api_key: str | None = None,
    ) -> dict:
        all_questions, question_map = _flatten_questions(psg)
        if not all_questions:
            return {"answers": {}, "score": 0.0}
        questions_text = "".join(
            f"{index + 1}. [{question_id}] "
            f"{question_map[question_id]['text']}\n"
            for index, question_id in enumerate(all_questions)
        )
        step_1_prompt = (
            "You are a verifier for AI-generated videos. Given a list of "
            "questions and a video, verify whether the video satisfies each "
            "question.\n\n"
            f"QUESTIONS:\n{questions_text}\n"
            "Provide your answer for each question in the format:\n"
            "[Question ID]: Your detailed reasoning here\n\n"
            "Answer each question completely."
        )
        video = {
            "type": "video_url",
            "video_url": {"url": video_data_url(video_path, args.media_cache)},
        }
        initial_text = call([{
            "role": "user",
            "content": [video, {"type": "text", "text": step_1_prompt}],
        }])
        initial = _parse_initial(initial_text, all_questions)
        followup = (
            "Based on your previous answers, give a yes/no for each question.\n"
            "Respond ONE per line:\n"
            "[Question ID]: yes  or  [Question ID]: no\n"
            f"Provide answers for all {len(all_questions)} questions."
        )
        final_text = call([{
            "role": "user",
            "content": [
                video,
                {"type": "text", "text": step_1_prompt},
                {"type": "text", "text": f"Assistant response:\n{initial_text}"},
                {"type": "text", "text": followup},
            ],
        }])
        yes_no = _parse_yesno(final_text, all_questions)
        answers, score = _build_answers_dict(
            all_questions,
            question_map,
            initial,
            yes_no,
        )
        return {"answers": answers, "score": score}

    pqsg.generate_psg = generate_psg
    pqsg.answer_psg_gemini = answer_psg_gemini
    sys.argv = [
        str(upstream / "scripts" / "run.py"),
        "--input",
        args.input,
        "--output",
        args.output,
        "--qg",
        args.model,
        "--qa",
        "gemini",
        "--qa-model",
        args.model,
    ]
    runpy.run_path(str(upstream / "scripts" / "run.py"), run_name="__main__")
    write_json(args.usage_output, usage.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
