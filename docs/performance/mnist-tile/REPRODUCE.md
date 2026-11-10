# Reproduction

Use the final checkout with Go 1.27.2. These commands keep baseline source,
binaries, alternate module files and caches in a fresh temporary directory.
The optional ONNX commands assume the sibling checkout and its existing cached
MNIST archive/model. They never change its source or dependency files.

```sh
conv_repo="$PWD"
conv_onnx="$(dirname "$conv_repo")/onnx"
conv_artifacts="$(mktemp -d /tmp/neuralnetwork-tile.XXXXXX)"
export GOWORK=off CGO_ENABLED=0 GOMAXPROCS=1 GOARM64=v8.0
export GOCACHE="$conv_artifacts/cache"
export CONV_REPO="$conv_repo" CONV_ONNX="$conv_onnx" CONV_ARTIFACTS="$conv_artifacts"

git rev-parse HEAD 'v0.10.0^{commit}'
go version
go env GOOS GOARCH GOARM64

git show v0.10.0:tensor/conv2d.go > "$conv_artifacts/conv2d-before.go"
python3 - <<'PY'
from pathlib import Path
import json, os, shutil
repo = Path(os.environ['CONV_REPO'])
onnx = Path(os.environ['CONV_ONNX'])
artifacts = Path(os.environ['CONV_ARTIFACTS'])
(artifacts/'baseline-overlay.json').write_text(json.dumps({'Replace': {
    str(repo/'tensor/conv2d.go'): str(artifacts/'conv2d-before.go')}}))
for variant in ['before', 'after']:
    source = (onnx/'go.mod').read_text()
    if variant == 'after':
        source += '\nreplace github.com/itsmontoya/neuralnetwork => ' + str(repo) + '\n'
    (artifacts/f'onnx-{variant}.mod').write_text(source)
    shutil.copyfile(onnx/'go.sum', artifacts/f'onnx-{variant}.sum')
PY

go test -overlay "$conv_artifacts/baseline-overlay.json" -c -o "$conv_artifacts/kernel-before.test" ./tensor
go test -c -o "$conv_artifacts/kernel-after.test" ./tensor
go test -tags purego -overlay "$conv_artifacts/baseline-overlay.json" -c -o "$conv_artifacts/kernel-purego-before.test" ./tensor
go test -tags purego -c -o "$conv_artifacts/kernel-purego-after.test" ./tensor

# Original baseline protocol; tensor construction remains outside timing.
"$conv_artifacts/kernel-before.test" -test.run '^$' -test.bench '^BenchmarkConv2DInto$/^MNIST' -test.benchtime=700ms -test.benchmem -test.count=7
"$conv_artifacts/kernel-before.test" -test.run '^$' -test.bench '^BenchmarkConv2DInto$' -test.benchtime=200ms -test.benchmem -test.count=5

# Run builds and tests before measurements, sequentially.
(cd "$conv_onnx" && go test -modfile "$conv_artifacts/onnx-before.mod" -c -o "$conv_artifacts/onnx-before.test" .)
(cd "$conv_onnx" && go test -modfile "$conv_artifacts/onnx-after.mod" -c -o "$conv_artifacts/onnx-after.test" .)
(cd "$conv_onnx" && go test -tags purego -modfile "$conv_artifacts/onnx-before.mod" -c -o "$conv_artifacts/onnx-purego-before.test" .)
(cd "$conv_onnx" && go test -tags purego -modfile "$conv_artifacts/onnx-after.mod" -c -o "$conv_artifacts/onnx-purego-after.test" .)

go test ./...
go test -tags purego ./...
CGO_ENABLED=1 go test -race ./...
CGO_ENABLED=1 go test -race -tags purego ./tensor
go vet ./...
go test -overlay "$conv_artifacts/baseline-overlay.json" -run Conv2D ./tensor
(cd "$conv_onnx" && go test -modfile "$conv_artifacts/onnx-after.mod" ./...)
(cd "$conv_onnx" && go test -tags purego -modfile "$conv_artifacts/onnx-after.mod" ./...)
GOOS=linux GOARCH=arm64 go test -c -o "$conv_artifacts/linux-arm64.test" ./tensor
GOOS=linux GOARCH=arm64 go test -tags purego -c -o "$conv_artifacts/linux-arm64-purego.test" ./tensor
GOOS=linux GOARCH=amd64 go test -c -o "$conv_artifacts/linux-amd64.test" ./tensor
GOOS=linux GOARCH=386 go test -c -o "$conv_artifacts/linux-386.test" ./tensor

# Each pair reverses order on the next repetition. No concurrent work.
python3 docs/performance/mnist-tile/measure.py "$conv_artifacts" mnist-kernel mnist kernels purego-kernels nodes purego
benchstat "$conv_artifacts/mnist-kernel-before.txt" "$conv_artifacts/mnist-kernel-after.txt"
benchstat "$conv_artifacts/kernels-before.txt" "$conv_artifacts/kernels-after.txt"
```

The existing optional-bias benchmark is unchanged. `BenchmarkConv2DIntoNoBias`
adds just the two nil-bias MNIST cases. Each scenario emits raw logs, executed
commands/order, samples, min/median/max and allocation sets into the artifact
directory. Use a fresh directory when repeating the harness.

Run the unchanged 19-case comparison with the existing pinned Python environment
(or one with NumPy 1.26.4, ONNX 1.16.2 and ORT 1.19.2). The measurements used
`/private/tmp/neuralnetwork-onnx-reference/bin/python`, Python 3.9.6. The original
driver fixes `GOWORK=off` but inherits `GOFLAGS`, so no driver copy is required.

```sh
conv_python=/private/tmp/neuralnetwork-onnx-reference/bin/python
for conv_variant in before after; do
    GOFLAGS="-modfile=$conv_artifacts/onnx-$conv_variant.mod" "$conv_python" "$conv_onnx/tools/compare_runtime.py" \
        --archive "$conv_onnx/.cache/compare-runtime/mnist-12.tar.gz" \
        --cache-dir "$conv_artifacts/model-cache" \
        --report "$conv_artifacts/ort-$conv_variant.json"
done
```

Keep defaults: 19 inputs, seed 0, ten warmups and 100 calls/input, `atol=1e-4`,
`rtol=1e-5`, one thread and disabled ORT graph optimization. Compare `inputs`,
`model.sha256`, `comparison` and every `results[].go_output` field between reports;
all checks and top-1 comparisons must pass and Go output bits must match.

Collect profiles separately from benchmarks:

```sh
for conv_variant in before after; do
    (cd "$conv_onnx" && "$conv_artifacts/onnx-$conv_variant.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' \
        -test.benchtime=12s -test.cpuprofile="$conv_artifacts/cpu-$conv_variant.pprof")
    pprof -top -focus='onnx.\(\*Session\).Run' -relative_percentages \
        "$conv_artifacts/cpu-$conv_variant.pprof" > "$conv_artifacts/cpu-$conv_variant.txt"
done

go test -gcflags=github.com/itsmontoya/neuralnetwork/tensor=-d=ssa/check_bce/debug=1 -c -o "$conv_artifacts/bce.test" ./tensor 2> "$conv_artifacts/bce.txt"
go test -gcflags=github.com/itsmontoya/neuralnetwork/tensor=-m -c -o "$conv_artifacts/escape.test" ./tensor 2> "$conv_artifacts/escape.txt"
```

On this host Go 1.27.2 does not include `go tool objdump`; LLVM 19 was used:

```sh
conv_objdump=/opt/homebrew/opt/llvm@19/bin/llvm-objdump
"$conv_objdump" --disassemble-symbols=_github.com/itsmontoya/neuralnetwork/tensor.conv2DFourValues "$conv_artifacts/kernel-before.test"
"$conv_objdump" --disassemble-symbols=_github.com/itsmontoya/neuralnetwork/tensor.conv2DInteriorTile.abi0 "$conv_artifacts/kernel-after.test"
```

To reproduce the rejected checked-Go candidate, combine the checked wrapper and
candidate implementation in an overlay for the portable entry file. On ARM64
this preserves all geometry and selects the Go tile with `purego` enabled:

```sh
python3 - <<'PY'
from pathlib import Path
import json, os
repo = Path(os.environ['CONV_REPO'])
artifacts = Path(os.environ['CONV_ARTIFACTS'])
wrapper = (repo/'tensor/conv2d_tile.go').read_text().split('package tensor\n', 1)[1]
candidate = (repo/'docs/performance/mnist-tile/checked-go-candidate.go.txt').read_text().split('package tensor\n', 1)[1]
(artifacts/'checked-go.go').write_text('package tensor\nconst conv2DHasInteriorTile = true\n' + wrapper + candidate)
(artifacts/'checked-go-overlay.json').write_text(json.dumps({'Replace': {
    str(repo/'tensor/conv2d_tile_portable.go'): str(artifacts/'checked-go.go')}}))
PY
go test -tags purego -overlay "$conv_artifacts/checked-go-overlay.json" -run Test_Conv2D_TileOutputBits -v ./tensor
# The NaN payload cases fail; this candidate is retained only as evidence.
```
