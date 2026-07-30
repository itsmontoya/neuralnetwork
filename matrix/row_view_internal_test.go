package matrix

import "testing"

func Test_MatrixRowViewSharesOnlySelectedBackingStorage(t *testing.T) {
	var (
		source   *Matrix
		retained *Matrix
		err      error
	)

	source, err = FromSlice(4, 2, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	err = source.WithRowView(1, 3, func(view *Matrix) (viewErr error) {
		retained = view
		if len(view.data) != 4 {
			t.Fatalf("view storage length = %d, want 4", len(view.data))
		}
		if &view.data[0] != &source.data[2] {
			t.Fatal("view storage does not begin at selected owner row")
		}
		if view.residency != nil {
			t.Fatal("new row view shares parent residency")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WithRowView returned error: %v", err)
	}
	if retained.rows != 0 || retained.cols != 0 || retained.data != nil || retained.residency != nil {
		t.Fatal("retained row view fields were not cleared")
	}
}

func Test_MatrixRowViewRejectsInvalidPrivateStorageBeforeCallback(t *testing.T) {
	type testcase struct {
		name   string
		source Matrix
	}

	var (
		tests  []testcase
		maxInt int
	)

	maxInt = int(^uint(0) >> 1)
	tests = []testcase{
		{
			name: "storage length mismatch",
			source: Matrix{
				rows: 2,
				cols: 2,
				data: []float32{1, 2, 3},
			},
		},
		{
			name: "dimension overflow",
			source: Matrix{
				rows: maxInt,
				cols: 2,
				data: []float32{1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called bool
				err    error
			)

			err = tt.source.WithRowView(0, 1, func(*Matrix) (viewErr error) {
				called = true
				return nil
			})
			if err == nil {
				t.Fatal("WithRowView error = nil, want invalid parent error")
			}
			if called {
				t.Fatal("WithRowView invoked callback for invalid parent")
			}
		})
	}
}
