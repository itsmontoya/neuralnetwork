//go:build !darwin || !arm64 || !cgo || !metal || purego

package gpu

import "testing"

func Test_OpenDefaultUnavailable(t *testing.T) {
	var d, err = OpenDefault()
	requireError(t, err, ErrUnavailable)
	if d != nil {
		t.Fatal("unavailable Metal backend returned a device")
	}
}
