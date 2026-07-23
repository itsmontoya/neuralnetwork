package device

import "testing"

func Test_MatMulEligibleForRuntime(t *testing.T) {
	type testcase struct {
		name         string
		operations   uint64
		runtimeReady bool
		want         bool
	}

	tests := []testcase{
		{
			name:         "cold below threshold",
			operations:   metalMatMulColdMinOperations - 1,
			runtimeReady: false,
			want:         false,
		},
		{
			name:         "cold at threshold",
			operations:   metalMatMulColdMinOperations,
			runtimeReady: false,
			want:         true,
		},
		{
			name:         "warm below threshold",
			operations:   metalMatMulWarmMinOperations - 1,
			runtimeReady: true,
			want:         false,
		},
		{
			name:         "warm at threshold",
			operations:   metalMatMulWarmMinOperations,
			runtimeReady: true,
			want:         true,
		},
	}

	var test testcase
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var got bool

			got = matMulEligibleForRuntime(test.operations, test.runtimeReady)
			if got != test.want {
				t.Fatalf(
					"matMulEligibleForRuntime(%d, %t) = %t, want %t",
					test.operations,
					test.runtimeReady,
					got,
					test.want,
				)
			}
		})
	}
}
