//go:build darwin && cgo && metal && !purego

package device

import (
	"slices"
	"testing"
)

func Test_MetalByteBufferRoundTripAndCopy(t *testing.T) {
	var (
		runtime  = requireMetalRuntime(t)
		baseline = runtime.ResourceSnapshot()
		counts   = []uint64{1, 3, 257}
	)

	for _, count := range counts {
		var (
			bytes  = count * 2
			source *Buffer
			target *Buffer
			scope  *Scope
			err    error
		)
		if source, err = runtime.NewByteBuffer(bytes); err != nil {
			t.Fatalf("NewByteBuffer(%d) source returned error: %v", bytes, err)
		}
		if target, err = runtime.NewByteBuffer(bytes); err != nil {
			source.Release()
			t.Fatalf("NewByteBuffer(%d) target returned error: %v", bytes, err)
		}

		var input = make([]uint16, count)
		for i := range input {
			input[i] = uint16((i*257 + 0x1357) & 0xffff)
		}
		if err = source.UploadUint16(input); err != nil {
			source.Release()
			target.Release()
			t.Fatalf("UploadUint16(%d) returned error: %v", count, err)
		}

		if scope, err = runtime.NewScope(); err != nil {
			source.Release()
			target.Release()
			t.Fatalf("NewScope returned error: %v", err)
		}
		if err = scope.EncodeCopy(source, target); err != nil {
			scope.Release()
			source.Release()
			target.Release()
			t.Fatalf("EncodeCopy(%d bytes) returned error: %v", bytes, err)
		}
		if err = scope.Commit(); err != nil {
			scope.Release()
			source.Release()
			target.Release()
			t.Fatalf("Commit returned error: %v", err)
		}
		if err = scope.Wait(); err != nil {
			scope.Release()
			source.Release()
			target.Release()
			t.Fatalf("Wait returned error: %v", err)
		}
		if err = scope.Release(); err != nil {
			source.Release()
			target.Release()
			t.Fatalf("Release scope returned error: %v", err)
		}

		var got = make([]uint16, count)
		if err = target.DownloadUint16(got); err != nil {
			source.Release()
			target.Release()
			t.Fatalf("DownloadUint16(%d) returned error: %v", count, err)
		}
		if !slices.Equal(got, input) {
			source.Release()
			target.Release()
			t.Fatalf("round-trip mismatch for %d uint16 elements", count)
		}

		source.Release()
		target.Release()
	}

	var after = runtime.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes ||
		after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("native resource leak: before=%+v after=%+v", baseline, after)
	}
}
