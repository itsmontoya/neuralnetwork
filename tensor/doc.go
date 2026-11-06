// Package tensor provides owned, contiguous row-major float32 and int64 values.
// Shapes contain concrete nonnegative int64 dimensions. A rank-zero shape holds
// one scalar; any zero dimension makes the value empty. Constructors, shape
// accessors, and value accessors copy their arguments or results.
//
// Published inputs may be read concurrently. Callers must synchronize writes
// and keep shared inputs separate from output destinations. Copy a Tensor with
// Clone for independent storage; a shallow Go copy shares its storage identity.
package tensor
