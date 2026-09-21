"""Sample per-process NVIDIA GPU memory and write a compact peak summary."""

import argparse
import csv
import json
import signal
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


class StopSampling:
    requested = False

    def __call__(self, _signum, _frame):
        self.requested = True


def process_info(pid: int) -> tuple[int | None, str]:
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8", errors="replace")
        ppid = int(stat[stat.rfind(")") + 2 :].split()[1])
    except (OSError, ValueError, IndexError):
        ppid = None
    try:
        raw = (proc / "cmdline").read_bytes().replace(b"\0", b" ").strip()
        command = raw.decode(errors="replace")
    except OSError:
        command = ""
    return ppid, command


def gpu_processes(gpu_id: str) -> list[tuple[int, float]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[1] in {"N/A", "[N/A]"}:
            continue
        processes.append((int(fields[0]), float(fields[1])))
    return processes


def write_summary(path: Path, samples: list[dict], error: str | None) -> None:
    peaks = defaultdict(float)
    commands = {}
    ppids = {}
    totals = defaultdict(float)
    for sample in samples:
        pid = sample["pid"]
        peaks[pid] = max(peaks[pid], sample["used_memory_mb"])
        if sample["command"] or pid not in commands:
            commands[pid] = sample["command"]
        if sample["ppid"] is not None or pid not in ppids:
            ppids[pid] = sample["ppid"]
        totals[sample["sample"]] += sample["used_memory_mb"]
    report = {
        "sample_count": len(totals),
        "peak_total_used_memory_mb": round(max(totals.values(), default=0.0), 2),
        "processes": [
            {
                "pid": pid,
                "ppid": ppids[pid],
                "peak_used_memory_mb": round(value, 2),
                "command": commands[pid],
            }
            for pid, value in sorted(peaks.items(), key=lambda item: -item[1])
        ],
        "error": error,
    }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    stop = StopSampling()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    samples = []
    error = None
    started = time.monotonic()
    try:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample",
                    "timestamp",
                    "elapsed_s",
                    "pid",
                    "ppid",
                    "used_memory_mb",
                    "command",
                ),
            )
            writer.writeheader()
            sample_index = 0
            while not stop.requested:
                for pid, used_mb in gpu_processes(args.gpu_id):
                    ppid, command = process_info(pid)
                    sample = {
                        "sample": sample_index,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "pid": pid,
                        "ppid": ppid,
                        "used_memory_mb": used_mb,
                        "command": command,
                    }
                    samples.append(sample)
                    writer.writerow(sample)
                handle.flush()
                sample_index += 1
                time.sleep(args.interval)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        write_summary(args.summary, samples, error)


if __name__ == "__main__":
    main()
