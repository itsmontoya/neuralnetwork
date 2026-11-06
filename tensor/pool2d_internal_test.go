package tensor

import "testing"

func Test_PoolOutputDimension(t *testing.T) {
	type testcase struct {
		name                                              string
		input, kernel, stride, dilation, begin, end, want int64
		ceil, invalid                                     bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "floor", input: 5, kernel: 2, stride: 2, dilation: 1, want: 2},
			{name: "ceil partial", input: 5, kernel: 2, stride: 2, dilation: 1, want: 3, ceil: true},
			{name: "ceil negative numerator", input: 2, kernel: 3, stride: 2, dilation: 1, want: 1, ceil: true},
			{name: "ceil zero extent", input: 1, kernel: 3, stride: 2, dilation: 1, ceil: true},
			{name: "ceil empty input", kernel: 1, stride: 2, dilation: 1, ceil: true},
			{name: "ceil removes pad starts", input: 3, kernel: 3, stride: 3, dilation: 1, end: 2, want: 1, ceil: true},
			{name: "ceil caps huge end padding", input: 1, kernel: 1, stride: 1, dilation: 1, end: limit - 1, want: 1, ceil: true},
			{name: "ceil coordinate overflow", input: limit - 2, kernel: limit - 1, stride: limit - 1, dilation: 1, begin: 2, ceil: true, invalid: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got, err = poolOutputDimension(tt.input, tt.kernel, tt.stride, tt.dilation, tt.begin, tt.end, tt.ceil)
			if (err != nil) != tt.invalid || got != tt.want {
				t.Fatalf("output=%d error=%v, want %d invalid=%v", got, err, tt.want, tt.invalid)
			}
		})
	}
}

func Test_PoolKernelRange(t *testing.T) {
	type testcase struct {
		name                           string
		start, input, kernel, dilation int64
		first, end                     int
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "within input", input: 4, kernel: 2, dilation: 1, end: 2},
			{name: "begin pad", start: -2, input: 3, kernel: 3, dilation: 2, first: 1, end: 3},
			{name: "end pad", start: 2, input: 3, kernel: 3, dilation: 1, end: 1},
			{name: "no input", kernel: 2, dilation: 1},
			{name: "starts after input", start: 3, input: 3, kernel: 2, dilation: 1},
			{name: "dilation misses input", start: -1, input: 1, kernel: 2, dilation: 2},
			{name: "extreme checked padding", start: -limit + 1, input: 1, kernel: limit, dilation: 1, first: int(limit - 1), end: int(limit)},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var first, end = poolKernelRange(tt.start, tt.input, tt.kernel, tt.dilation)
			if first != tt.first || end != tt.end {
				t.Fatalf("range=[%d,%d), want [%d,%d)", first, end, tt.first, tt.end)
			}
		})
	}
}
