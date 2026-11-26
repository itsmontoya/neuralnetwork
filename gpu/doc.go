// Package gpu provides experimental, explicit Metal execution for contiguous
// row-major tensors. Supported storage types are Float32, Float16, BFloat16,
// Q8_0, Q4_0, and Q4_1. On macOS arm64, build with cgo and the metal tag.
// Default builds, purego builds, unsupported hosts, and machines without a
// default Metal device return ErrUnavailable from OpenDefault.
//
// Each Device owns a logical namespace for its tensors and scopes. Use one
// handle to share read-only weights between scopes and give each scope its own
// mutable destinations. Upload and DownloadInto copy full slices during the
// call. Tensor metadata is immutable, and Shape copies the dimensions. A
// tensor needs a full upload or successful GPU write before it can be read.
// An earlier write in the same scope can initialize a subsequent operation's
// input.
//
// Scope operations record commands and acquire whole-buffer access leases.
// Readers can coexist. Writers need exclusive access, and host transfers
// require no outstanding scope lease. Transfers and conflicting operations
// return ErrBusy so the caller can choose when to synchronize. Submit commits
// the recording once. Wait publishes finished writes and releases leases.
// When a submitted command fails, its write destinations become unreadable
// until a successful full write. An operation validation error after earlier operations,
// or an error from the native encoder, requires closing and discarding the
// scope. Discarding preserves the previous tensor values.
//
// A KVCache owns private, fixed-capacity key and value history. AppendKV
// stages consecutive rows in a disjoint slot. After a successful Wait the
// caller can publish or discard the staged rows as part of a larger step.
//
// If a context is canceled at the submission boundary, Submit discards the
// recording. Canceling Wait leaves submitted work and its leases outstanding.
// Wait again or close the scope to drain it. Close every scope and tensor.
// A tensor can outlive its scope. Tensor.View creates zero-copy reshaped views
// that share the underlying allocation, initialization, and lease tracking.
// Closing a tensor or view prevents new uses on that handle and releases the
// underlying allocation after all sibling views close and the last existing
// lease or transfer ends. Device.Close returns ErrBusy while its children remain. The underlying
// default Metal device, queue and runtime pipelines are shared with the ANN
// path for the life of the process.
package gpu
