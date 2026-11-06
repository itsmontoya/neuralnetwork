package tensor

import "testing"

func Test_SpatialOutputDimension(t *testing.T) {
	type testcase struct {
		name                                        string
		input, kernel, stride, dilation, begin, end int64
		want                                        int64
		invalid                                     bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "valid", input: 5, kernel: 3, stride: 2, dilation: 1, want: 2},
			{name: "dilated asymmetric", input: 4, kernel: 2, stride: 2, dilation: 2, begin: 1, end: 2, want: 3},
			{name: "empty", kernel: 1, stride: 1, dilation: 1},
			{name: "padded empty", kernel: 1, stride: 1, dilation: 1, begin: 1, end: 2, want: 3},
			{name: "negative numerator floor", input: 2, kernel: 3, stride: 2, dilation: 1},
			{name: "zero stride", input: 2, kernel: 1, dilation: 1, invalid: true},
			{name: "negative end", input: 2, kernel: 1, stride: 1, dilation: 1, end: -1, invalid: true},
			{name: "kernel overflow", input: 2, kernel: 2, stride: 1, dilation: limit, invalid: true},
			{name: "padding overflow", input: limit, kernel: 1, stride: 1, dilation: 1, end: 1, invalid: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got, err = spatialOutputDimension(tt.input, tt.kernel, tt.stride, tt.dilation, tt.begin, tt.end)
			if (err != nil) != tt.invalid || got != tt.want {
				t.Fatalf("output=%d error=%v, want %d invalid=%v", got, err, tt.want, tt.invalid)
			}
		})
	}
	if err := validateSpatialCoordinates(2, 2, limit, 1); err == nil {
		t.Fatal("sample-coordinate sum overflow accepted")
	}
}
