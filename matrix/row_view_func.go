package matrix

// RowViewFunc consumes a temporary contiguous row window owned by another
// Matrix. The view aliases owner storage and is valid only during the
// synchronous call. It must not be retained, mutated, or used concurrently.
type RowViewFunc func(view *Matrix) (err error)
