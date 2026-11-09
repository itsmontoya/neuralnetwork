#!/usr/bin/env python3
"""Observe payload copies/lifetimes in isolated, exactly pinned source archives.

The Go workspace resolves ONLY the instrumented source copies. No production
checkout/module cache is patched. Do not use these instrumented builds for time
or allocation measurements. Use the ordinary runtime benchmark for those.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(arguments, **kwargs):
    try:
        return subprocess.run(arguments, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as failure:
        if failure.stdout:
            print(failure.stdout)
        if failure.stderr:
            print(failure.stderr)
        raise


def archive(repository, revision, destination):
    destination.mkdir()
    source = subprocess.Popen(["git", "-C", str(repository), "archive", revision], stdout=subprocess.PIPE)
    try:
        run(["tar", "-x", "-C", str(destination)], stdin=source.stdout)
    finally:
        source.stdout.close()
    if source.wait():
        raise RuntimeError("git archive failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-revision", default="4d5a000c20ffae3e053190a4cd04322529851800")
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    tooling = Path(__file__).resolve().parent
    environment = dict(os.environ, GOMAXPROCS="1", CGO_ENABLED="0", GOWORK="off")
    with tempfile.TemporaryDirectory(prefix="nn13-metrics-") as temporary:
        root = Path(temporary).resolve()
        runtime, backend = root / "onnx", root / "backend"
        archive(arguments.runtime, arguments.runtime_revision, runtime)
        archive(arguments.backend, arguments.backend_revision, backend)
        shutil.copytree(tooling, backend / "tools/nn13", dirs_exist_ok=True)
        binary = root / "instrument"
        run(["go", "build", "-o", str(binary), "./tools/nn13/instrument"], cwd=backend, env=environment)
        run([str(binary), str(backend)])
        run([str(binary), str(runtime)])
        shape = runtime / "shape_call.go"
        source = shape.read_text()
        marker = "c.shape, c.shapeValues = []int64{int64(end - start)}, left[start:end]"
        if source.count(marker) != 1:
            raise RuntimeError("pinned Shape allocation marker changed")
        shape.write_text(source.replace(marker, marker + "\n nn13trace.Track(c.shapeValues)"))
        shutil.copyfile(tooling / "runtime_metrics_test.go.txt", runtime / "nn13_metrics_test.go")
        run(["go", "work", "init", "./onnx", "./backend"], cwd=root, env=environment)
        environment["GOWORK"] = str(root / "go.work")
        output = run(["go", "test", "-run", "^TestNN13PayloadMetrics$", "-count=1", "-v", "."], cwd=runtime, env=environment, capture_output=True)
        observations = [json.loads(line[len("NN13METRIC "):]) for line in output.stdout.splitlines() if line.startswith("NN13METRIC ")]
        if len(observations) != 12:
            raise RuntimeError(f"expected 12 observations; got {len(observations)}\n{output.stdout}")
        observations.sort(key=lambda item: (item["case"], item["phase"]))
        report = {"runtime_revision": arguments.runtime_revision, "backend_revision": arguments.backend_revision, "metrics": observations}
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote 12 payload observations to {arguments.output}")


if __name__ == "__main__":
    main()
