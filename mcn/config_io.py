from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".json", ".jsonc"}:
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError("PyYAML is required to load non-JSON YAML config files") from exc
    payload = yaml.safe_load(text)
    return payload or {}


def save_config(path: str | Path, payload: dict[str, Any]) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.suffix.lower() in {".json", ".jsonc"}:
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    try:
        import yaml
    except ImportError:
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def flatten_cli_args(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for section in ("model", "training", "loss", "data", "logging"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif value is not None:
                args.extend([flag, str(value)])
    return args
