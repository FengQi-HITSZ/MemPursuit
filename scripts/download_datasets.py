"""Download LoCoMo, LongMemEval-S and BEAM datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from benchmarks.pipeline.io import atomic_write_json, sha256_file

BEAM_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
URLS = {
    "locomo": (
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "locomo/locomo10.json",
    ),
    "longmemeval": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "longmemeval/longmemeval_s_cleaned.json",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", metavar="BENCHMARK", help="locomo, longmemeval, beam; all three by default")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--beam-size", choices=["500K", "1M"], default="500K")
    args = parser.parse_args()
    if any(name not in {"locomo", "longmemeval", "beam"} for name in args.benchmarks):
        parser.error("benchmarks must be locomo, longmemeval and/or beam")
    for benchmark in args.benchmarks or ["locomo", "longmemeval", "beam"]:
        if benchmark in URLS:
            url, filename = URLS[benchmark]
            target = args.output_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix(".download")
                request = urllib.request.Request(url, headers={"User-Agent": "mempursuit/1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                value = json.loads(temporary.read_text())
                expected = 10 if benchmark == "locomo" else 500
                if not isinstance(value, list) or len(value) != expected:
                    raise ValueError(f"Unexpected {benchmark} dataset cardinality")
                temporary.replace(target)
            metadata = {"source_url": url, "sha256": sha256_file(target)}
        else:
            from datasets import load_dataset
            import pyarrow as pa

            target = args.output_dir / "beam" / f"beam-{args.beam_size}.arrow"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                data = load_dataset("Mohammadta/BEAM", split=args.beam_size, revision=BEAM_REVISION)
                if len(data) != 35:
                    raise ValueError("Expected 35 BEAM conversations in the selected split")
                temporary = target.with_suffix(".download")
                with pa.OSFile(str(temporary), "wb") as sink:
                    with pa.ipc.new_stream(sink, data.data.table.schema) as writer:
                        writer.write_table(data.data.table)
                temporary.replace(target)
            metadata = {
                "repository": "Mohammadta/BEAM",
                "revision": BEAM_REVISION,
                "split": args.beam_size,
                "sha256": sha256_file(target),
            }
        atomic_write_json(target.with_suffix(target.suffix + ".source.json"), metadata)
        print(f"{benchmark}: {target} ({metadata['sha256']})")


if __name__ == "__main__":
    main()
