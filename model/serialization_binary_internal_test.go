package model

import (
	"bytes"
	"crypto/sha256"
	"strings"
	"testing"
)

func TestDecodeBinaryDocumentRejectsOversizedCounts(t *testing.T) {
	t.Run("layers", func(t *testing.T) {
		var (
			document sequentialDocument
			encoded  bytes.Buffer
			err      error
		)

		if err = writeBinaryUint(&encoded, binarySerializationMaxLayers+1); err != nil {
			t.Fatalf("writeBinaryUint returned error: %v", err)
		}
		if document, err = decodeBinaryDocument(&encoded); err == nil {
			t.Fatal("decodeBinaryDocument error = nil, want layer limit error")
		}
		if len(document.Layers) != 0 {
			t.Fatal("decodeBinaryDocument allocated layers on limit error")
		}
		if !strings.Contains(err.Error(), "layer count") || !strings.Contains(err.Error(), "exceeds limit") {
			t.Fatalf("decodeBinaryDocument error = %q, want layer limit context", err)
		}
	})

	t.Run("activation widths", func(t *testing.T) {
		var (
			document sequentialDocument
			encoded  bytes.Buffer
			err      error
			index    int
		)

		if err = writeBinaryUint(&encoded, 1); err != nil {
			t.Fatalf("write layer count returned error: %v", err)
		}
		if err = writeBinaryString(&encoded, serializationLayerActivation); err != nil {
			t.Fatalf("write type returned error: %v", err)
		}
		for index = 0; index < 5; index++ {
			if err = writeBinaryUint(&encoded, 0); err != nil {
				t.Fatalf("write integer returned error: %v", err)
			}
		}
		if err = writeBinaryString(&encoded, "segmented_softmax"); err != nil {
			t.Fatalf("write activation returned error: %v", err)
		}
		if err = writeBinaryUint(&encoded, binarySerializationMaxWidths+1); err != nil {
			t.Fatalf("write width count returned error: %v", err)
		}
		if document, err = decodeBinaryDocument(&encoded); err == nil {
			t.Fatal("decodeBinaryDocument error = nil, want width limit error")
		}
		if len(document.Layers) != 0 {
			t.Fatal("decodeBinaryDocument returned partial document on error")
		}
		if !strings.Contains(err.Error(), "activation width count") || !strings.Contains(err.Error(), "exceeds limit") {
			t.Fatalf("decodeBinaryDocument error = %q, want width limit context", err)
		}
	})
}

func TestDecodeBinaryMatrixRejectsOversizedAndMalformedRecords(t *testing.T) {
	type testcase struct {
		name      string
		presence  byte
		rows      uint64
		cols      uint64
		wantError string
	}

	var tests []testcase
	tests = []testcase{
		{
			name:      "invalid presence",
			presence:  2,
			wantError: "presence marker",
		},
		{
			name:      "zero rows",
			presence:  1,
			cols:      1,
			wantError: "dimensions must be positive",
		},
		{
			name:      "value limit",
			presence:  1,
			rows:      binarySerializationMaxValues + 1,
			cols:      1,
			wantError: "values exceed limit",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				buffer      [binarySerializationBuffer]byte
				encoded     bytes.Buffer
				totalValues uint64
				current     *serializedMatrix
				err         error
			)

			encoded.WriteByte(tt.presence)
			if tt.presence == 1 {
				if err = writeBinaryUint(&encoded, tt.rows); err != nil {
					t.Fatalf("write rows returned error: %v", err)
				}
				if err = writeBinaryUint(&encoded, tt.cols); err != nil {
					t.Fatalf("write cols returned error: %v", err)
				}
			}
			if current, err = decodeBinaryMatrix(&encoded, buffer[:], &totalValues); err == nil {
				t.Fatal("decodeBinaryMatrix error = nil, want error")
			}
			if current != nil {
				t.Fatal("decodeBinaryMatrix returned matrix on error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("decodeBinaryMatrix error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func TestDecodeSequentialBinaryRejectsUnknownLayerWithValidChecksum(t *testing.T) {
	var (
		document sequentialDocument
		artifact bytes.Buffer
		payload  bytes.Buffer
		digest   [sha256.Size]byte
		loaded   *Sequential
		checksum ModelChecksum
		err      error
	)

	document.Format = serializationFormatSequential
	document.Version = serializationVersion
	document.Layers = []serializedLayer{{Type: "future_layer"}}
	if err = encodeBinaryDocument(&payload, document); err != nil {
		t.Fatalf("encodeBinaryDocument returned error: %v", err)
	}
	if err = writeBinaryBytes(&artifact, []byte(binarySerializationMagic)); err != nil {
		t.Fatalf("write magic returned error: %v", err)
	}
	if err = writeBinaryUint16(&artifact, binarySerializationVersion); err != nil {
		t.Fatalf("write version returned error: %v", err)
	}
	if err = writeBinaryBytes(&artifact, payload.Bytes()); err != nil {
		t.Fatalf("write payload returned error: %v", err)
	}
	digest = sha256.Sum256(payload.Bytes())
	if err = writeBinaryBytes(&artifact, digest[:]); err != nil {
		t.Fatalf("write checksum returned error: %v", err)
	}

	loaded, checksum, err = decodeSequentialBinary(&artifact)
	if err == nil {
		t.Fatal("decodeSequentialBinary error = nil, want unknown layer error")
	}
	if loaded != nil || checksum != (ModelChecksum{}) {
		t.Fatal("decodeSequentialBinary returned output on error")
	}
	if !strings.Contains(err.Error(), "unknown layer type") {
		t.Fatalf("decodeSequentialBinary error = %q, want unknown layer type", err)
	}
}
