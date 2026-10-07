import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался YAML-объект: {path}")
    return data


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not slug:
        raise ValueError("Имя эксперимента не содержит допустимых символов")
    return slug


def resolve_from_project(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_single_chunk_dataset(path: Path) -> tuple[int, str]:
    """Проверяет схему и возвращает число примеров и хеш содержимого."""
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("golden dataset должен быть непустым JSON-массивом")

    errors = []
    for row_number, item in enumerate(data, start=1):
        contexts = item.get("contexts") if isinstance(item, dict) else None
        source = item.get("source") if isinstance(item, dict) else None
        if (
            not isinstance(contexts, list)
            or len(contexts) != 1
            or not isinstance(contexts[0], str)
            or not contexts[0].strip()
        ):
            errors.append(f"строка {row_number}: нужен ровно один contexts")
        if not isinstance(source, dict) or source.get("file_name") is None or source.get("chunk_index") is None:
            errors.append(f"строка {row_number}: нет source.file_name/chunk_index")
    if errors:
        raise ValueError(
            "Датасет не соответствует схеме одного золотого чанка: "
            + "; ".join(errors[:5])
        )
    return len(data), hashlib.sha256(raw).hexdigest()


def run_signature(dataset_sha256: str, run_config: dict) -> str:
    payload = json.dumps(
        {"dataset_sha256": dataset_sha256, "config": run_config},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_service(url: str, name: str) -> None:
    try:
        with urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except (URLError, OSError, RuntimeError) as error:
        raise RuntimeError(f"{name} недоступен по адресу {url}: {error}") from error


def build_run_config(base: dict, matrix_name: str, run: dict) -> dict:
    allowed = {
        "name",
        "search_mode",
        "top_n_retrieval",
        "top_k",
        "use_reranker",
        "reranker_model",
    }
    unknown = set(run) - allowed
    if unknown:
        raise ValueError(
            f"Запуск '{run.get('name', '?')}' содержит параметры, которые могут "
            f"потребовать переиндексации или не поддерживаются: {sorted(unknown)}"
        )

    config = deepcopy(base)
    config["experiment_name"] = matrix_name
    config["top_k"] = int(run["top_k"])
    config["vector_store"]["search_mode"] = run["search_mode"]
    config["reranker"]["top_n_retrieval"] = int(run["top_n_retrieval"])
    config["reranker"]["use_reranker"] = bool(run["use_reranker"])
    if "reranker_model" in run:
        config["reranker"]["model_name"] = run["reranker_model"]
    config["vector_store"]["recreate_on_start"] = False
    return config


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    name: str,
    config_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> tuple[int, float]:
    metrics_path = output_dir / "metrics" / f"{name}.json"
    log_path = output_dir / "logs" / f"{name}.log"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_rag.py"),
        "--config", str(config_path),
        "--dataset", str(dataset_path),
        "--run-name", name,
        "--output-dir", str(output_dir / "reports"),
        "--metrics-output", str(metrics_path),
    ]

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Последовательный запуск retrieval-экспериментов"
    )
    parser.add_argument("--matrix", default="experiments.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Пропускать запуски, для которых уже существует metrics JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать матрицу без запуска экспериментов",
    )
    args = parser.parse_args()

    matrix_path = resolve_from_project(args.matrix)
    matrix = load_yaml(matrix_path)
    base_config_path = resolve_from_project(matrix["base_config"])
    dataset_path = resolve_from_project(matrix["dataset"])
    output_dir = resolve_from_project(matrix.get("output_dir", "experiment_results"))
    runs = matrix.get("runs", [])
    if not runs:
        parser.error("В матрице нет запусков")
    if not dataset_path.exists():
        parser.error(f"Датасет не найден: {dataset_path}")

    names = [run.get("name") for run in runs]
    if any(not name for name in names) or len(names) != len(set(names)):
        parser.error("Каждый запуск должен иметь уникальное непустое имя")

    if args.list:
        for index, run in enumerate(runs, start=1):
            print(
                f"{index:02d}. {run['name']}: mode={run['search_mode']}, "
                f"retrieval={run['top_n_retrieval']}, top_k={run['top_k']}, "
                f"reranker={run['use_reranker']}, "
                f"model={run.get('reranker_model', 'base config')}"
            )
        return 0

    try:
        dataset_size, dataset_sha256 = validate_single_chunk_dataset(dataset_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(
        f"📚 Golden dataset: {dataset_size} примеров, один эталонный чанк, "
        f"sha256={dataset_sha256[:12]}"
    )

    base_config = load_yaml(base_config_path)
    qdrant = base_config["vector_store"]
    qdrant_url = qdrant.get("url") or f"http://{qdrant['host']}:{qdrant['port']}"
    ollama_url = base_config["llm"]["base_url"].rstrip("/")
    check_service(f"{qdrant_url.rstrip('/')}/collections", "Qdrant")
    check_service(f"{ollama_url}/api/tags", "Ollama")

    config_dir = output_dir / "configs"
    metrics_dir = output_dir / "metrics"
    config_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for index, run in enumerate(runs, start=1):
        name = slugify(run["name"])
        metrics_path = metrics_dir / f"{name}.json"
        metadata_path = metrics_dir / f"{name}.meta.json"
        run_config = build_run_config(base_config, matrix["experiment_name"], run)
        signature = run_signature(dataset_sha256, run_config)
        print(f"\n{'=' * 72}\n[{index}/{len(runs)}] {name}\n{'=' * 72}")

        saved_metadata = {}
        if metadata_path.exists():
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            args.resume
            and metrics_path.exists()
            and saved_metadata.get("run_signature") == signature
        ):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_rows.append({
                "run_name": name,
                "status": "skipped",
                "dataset_sha256": dataset_sha256,
                **run,
                **metrics,
            })
            write_summary(summary_rows, output_dir)
            print("⏭️ Результат уже существует, запуск пропущен.")
            continue

        metrics_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

        config_path = config_dir / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        return_code, duration = run_one(
            name, config_path, dataset_path, output_dir
        )
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        status = "success" if return_code == 0 and metrics else "failed"
        if status == "success":
            metadata_path.write_text(
                json.dumps({
                    "run_signature": signature,
                    "dataset_sha256": dataset_sha256,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary_rows.append({
            "run_name": name,
            "status": status,
            "duration_seconds": round(duration, 3),
            "dataset_sha256": dataset_sha256,
            **{key: value for key, value in run.items() if key != "name"},
            **metrics,
        })
        write_summary(summary_rows, output_dir)

    successes = sum(row["status"] in {"success", "skipped"} for row in summary_rows)
    print(
        f"\nЗавершено: {successes}/{len(summary_rows)}. "
        f"Сводка: {output_dir / 'summary.csv'}"
    )
    return 0 if successes == len(summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
