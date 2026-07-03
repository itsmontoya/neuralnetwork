# Testing Baseline

Use this command as the repository baseline:

```sh
go test ./...
```

Every package added after the initial scaffold should include focused unit tests
with the package implementation. Prefer table-driven tests when behavior is
deterministic and the expected values can be stated explicitly.

Floating-point assertions should use helpers from
`github.com/itsmontoya/neuralnetwork/internal/testutil` so tolerance choices and
failure messages stay consistent across packages.

Benchmarks should wait until the implementation under test has correctness
coverage. Do not tune performance before the expected behavior is covered by
unit tests.

The v1 numeric type is `float64`. This keeps the initial implementation simple
and gives gradient checks more stable precision. `float32` support is deferred
until there is a measured need.
