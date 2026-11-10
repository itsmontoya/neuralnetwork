# Reproduce the measurements

From the backend checkout, create a workspace without changing ONNX's pin:

```sh
backend_root=$PWD
onnx_root=../onnx
measurement_dir=$(mktemp -d /tmp/conv2d-review.XXXXXX)
(cd "$measurement_dir" && go work init "$backend_root" "$backend_root/../onnx")
GOWORK="$measurement_dir/go.work" go work edit -go=1.26.1 -toolchain=go1.27.2
GOWORK=off CGO_ENABLED=0 go test ./...
GOWORK=off CGO_ENABLED=0 go test -tags=purego ./tensor
GOWORK=off CGO_ENABLED=1 go test -race ./...
GOWORK=off CGO_ENABLED=0 go vet ./...
(cd "$onnx_root" && GOWORK="$measurement_dir/go.work" CGO_ENABLED=0 go test ./...)
(cd "$onnx_root" && GOWORK="$measurement_dir/go.work" CGO_ENABLED=0 go test -tags=purego ./...)
(cd "$onnx_root" && GOWORK=off CGO_ENABLED=0 go test -c -o "$measurement_dir/base.test" .)
(cd "$onnx_root" && GOWORK="$measurement_dir/go.work" CGO_ENABLED=0 go test -c -o "$measurement_dir/change.test" .)
```

Alternate those binaries from the ONNX root, seven repetitions, using
`GOMAXPROCS=1`, `-test.run '^$' -test.bench '^BenchmarkMNIST$'
-test.benchtime=700ms -test.benchmem`. Prepare the optional model/Python cache
with ONNX's `tools/compare_runtime/README.md`. Use the original driver for the
baseline and a temporary copy with the two substitutions above for the workspace
run, passing the exact comparison settings above. The local `bench.py`,
`comparisons.py`, `profile.py`, overlays and `go.work` retain the executed commands.
