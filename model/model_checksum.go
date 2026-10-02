package model

import "encoding/hex"

// ModelChecksum is the SHA-256 digest of a binary model's architecture and
// evaluation parameters.
type ModelChecksum [32]byte

// String returns the lowercase hexadecimal checksum representation.
func (c ModelChecksum) String() (checksum string) {
	checksum = hex.EncodeToString(c[:])
	return checksum
}
