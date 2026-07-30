# Opt-In Data Views

Status: accepted implementation contract.

This document freezes the additive public and behavioral contract for
[ROADMAP Item 3](../ROADMAP.md#3-add-opt-in-zero-copy-data-views). It is the
implementation contract for the remaining milestone sections. The declarations
are not implemented yet; the status means that implementation must use these
exact names and semantics rather than reopening them while optimizing.

The accepted boundary is a scoped, read-only-intent callback. Ordered
contiguous rows share the data owner's host storage for the callback. The
callback receives ordinary `*matrix.Matrix` values because every existing
layer consumes that type. Go cannot prevent the callback from retaining or
mutating those pointers, so the contract forbids both actions and invalidates
the temporary matrices when the callback returns. It does not claim language-
enforced immutability.

Non-contiguous rows cannot be represented by the current dense matrix layout.
Callers select either strict `data.ViewOnly` rejection or the explicit
`data.ViewOrCopy` fallback. Every published data view reports whether that
fallback copied its selected values.

## Decision Summary

The accepted path has four layers:

```text
matrix.Matrix.WithRowView
    -> data DatasetView or SequenceDatasetView
    -> explicit view batching, selection, and splitting
    -> model Sequential.FitWithViews or FitWithLengthViews
```

`matrix.Matrix.WithRowView` is the only matrix-level addition. It publishes a
temporary `*matrix.Matrix` whose rows are one contiguous window of its parent.
It does not add a permanent submatrix, arbitrary strides, indirect row
indexes, borrowed constructor, public device handle, or general immutable
matrix.

`data.DatasetView` keeps ordinary inputs and targets paired.
`data.SequenceDatasetView` keeps sequence inputs, targets, steps, and logical
lengths in one selection. There is no independent `SequenceLengths` view.

`data.ViewPolicy` makes non-contiguous behavior explicit:

* `ViewOnly` accepts only whole or contiguous ordered rows and otherwise fails
  before callback publication or random-source consumption.
* `ViewOrCopy` still views contiguous rows, but it may prepare owned selected
  copies for shuffled, repeated, or arbitrary rows. `Copied` then reports
  `true`.

The model additions wrap the existing fit configurations rather than adding
fields to them. Ordered view fitting copies no dataset input, target, or
length storage. Shuffled view fitting requires `ViewOrCopy`: its training
batches retain the current selected-row copies, while complete training and
validation evaluation use views.

All existing constructors, accessors, batches, splits, fit operations, layer
formulas, parameter behavior, random behavior, and sequential persistence stay
unchanged.

## Verified Current Contract

The following facts were rechecked against the repository before accepting
this design.

### Matrix

`matrix.Matrix` has private positive dimensions, one contiguous row-major
`[]float32`, and optional private `device.Residency`. `New`, `FromSlice`, and
the random constructors allocate independent host storage. `Values`, `Clone`,
and `SelectRows` copy. `ValuesInto` and `SelectRowsInto` copy into
caller-owned destinations.

Shape methods do not synchronize. Host observations synchronize a
device-newer matrix. Host writes publish a new host revision and make device
content stale. `Clone` and `CopyFrom` preserve independent ownership even when
their implementation uses a device-to-device copy. The complete residency and
fallback contract is recorded in [metal.md](metal.md).

No existing matrix method publishes a slice, permanent row window, submatrix,
stride, or view. Matrix mutation methods operate on the receiver.

### Ordinary data

`NewDataset` validates equal input and target row counts and clones both
matrices. `LoadCSV` builds package-owned matrices. `Dataset.Inputs`,
`Dataset.Targets`, `Batch.Inputs`, and `Batch.Targets` clone again. Their
`Into` methods and `Dataset.SelectRowsInto` copy without retaining caller
destinations.

`Dataset.Batches` uses one ordered or caller-shuffled row index list and
creates independently owned batch matrices. `Dataset.Split` uses the same
selection rule and returns independently owned datasets.

### Sequence data

`NewSequenceLengths` requires positive steps, a non-empty slice, and each
value in `[1, steps]`, then copies the slice. `Values` copies again, while
`ValuesInto` and `SelectRowsInto` copy without retaining destinations.
Private storage and the absence of an aliasing accessor make the current type
immutable.

`NewSequenceDataset` validates the complete association and independently
copies inputs, targets, and lengths. `SequenceDataset`, `SequenceBatch`,
batching, splitting, and selection apply one row order to all three values.
Their allocating accessors return owned copies and their destination accessors
retain nothing. This implemented contract is recorded in
[sequence-lengths-design.md](sequence-lengths-design.md).

### Model fitting

`Sequential.Fit` and `FitWithLengths` build one index slice per epoch. They
copy selected rows into reusable `fitScratch` matrices instead of calling the
allocating data batching APIs. The sequence path copies aligned lengths into
reusable `[]int` scratch.

Training and validation evaluation copy each complete dataset into reusable
evaluation matrices before prediction, loss, and accuracy. Warmed epoch
helpers already allocate zero; their pressure is the repeated byte copy, not
only allocation count.

Fit validation, schedule timing, batch traversal, prediction, loss, backward,
optimizer update, history, callback, early stopping, training-mode restoration,
and cleanup are implemented in `model/sequential.go`. Gradient clipping
composes at the unchanged optimizer boundary described in
[gradient-clipping-design.md](gradient-clipping-design.md).

### Stable layouts and APIs

One matrix row remains one sample. Ordinary physical shapes are
`[N, inputSize]` and `[N, targetSize]`. Sequence inputs remain
`[N, steps*featureSize]` in flattened time-major `TF` order, targets remain
`[N, targetSize]`, and lengths remain one positive `int` per row.

The accepted v1 inventory is in [v1-api-review.md](v1-api-review.md).
Flattened `CHW` CNN rows and layer ownership are in
[cnn-design.md](cnn-design.md). Flattened `TF` RNN rows, stateless recurrence,
and scratch ownership are in [rnn-design.md](rnn-design.md). Views change none
of those layouts or formulas.

## Public API

The following declarations are exact. All new exported symbols require concise
GoDoc using the ownership vocabulary in this document.

### Matrix row view

Package `matrix` adds:

```go
type RowViewFunc func(view *Matrix) (err error)

func (m *Matrix) WithRowView(
	start,
	end int,
	use RowViewFunc,
) (err error)
```

`start` is inclusive and `end` is exclusive. A valid call requires
`0 <= start < end <= m.Rows()`. The published shape is
`[end-start, m.Cols()]`, and its row zero begins at parent row `start`.
Columns remain in their original row-major order.

The callback is synchronous. The temporary matrix is valid only until `use`
returns. `WithRowView` expires it on success, returned error, and panic. The
zero value of `RowViewFunc` is nil and invalid. A nil or zero-value `Matrix`
remains invalid.

The callback may use shape and observation methods, pass the view as a
read-only layer or model input, and clone values it must retain. It must not:

* retain the temporary matrix;
* mutate it through `Set`, `Fill`, `CopyFrom`, an in-place method, or use as a
  destination;
* publish it to another goroutine;
* mutate the parent or another overlapping alias; or
* start work that can outlive the callback.

These restrictions are contractual, not enforced by Go's type system.
Compliant uses receive a read-only alias. Behavior after a prohibited mutation
or escape is unsupported and may include visible owner mutation or broken
host/device coherence.

### View policy

Package `data` adds:

```go
type ViewPolicy uint8

const (
	ViewOnly ViewPolicy = iota
	ViewOrCopy
)
```

`ViewOnly` is the zero value and the strict policy. It never falls back to a
bulk value copy. `ViewOrCopy` permits a copied selection only where this
document says so. Other numeric values are invalid.

Policy controls row-selection preparation, not accessor ownership. An existing
copying accessor still copies regardless of policy. A view accessor still
aliases the view's current backing storage even when that backing storage was
prepared by `ViewOrCopy`.

### Ordinary view

Package `data` adds:

```go
type DatasetView struct { /* unexported fields */ }

func (v *DatasetView) Validate() (err error)
func (v *DatasetView) Inputs() (inputs *matrix.Matrix, err error)
func (v *DatasetView) Targets() (targets *matrix.Matrix, err error)
func (v *DatasetView) SampleCount() (samples int)
func (v *DatasetView) InputSize() (features int)
func (v *DatasetView) TargetSize() (values int)
func (v *DatasetView) Copied() (copied bool)

type DatasetViewFunc func(view *DatasetView) (err error)

type DatasetSplitViewFunc func(
	train,
	test *DatasetView,
) (err error)
```

`Inputs` and `Targets` return the paired temporary matrices without copying.
They succeed only while the view callback is active. A nil, zero, or expired
`DatasetView` has scalar accessors returning zero values; `Validate`, `Inputs`,
and `Targets` return contextual errors when it is unusable.

`Copied` is false for a whole owner or contiguous owner window. It is true
when `ViewOrCopy` had to materialize an owned non-contiguous selection before
the callback. It returns false for a nil, zero, or expired view. `Copied` does
not weaken the callback lifetime or mutation rules.

Package `data` adds these owner methods:

```go
func (d *Dataset) WithView(use DatasetViewFunc) (err error)
func (d *Dataset) WithRowView(
	start,
	end int,
	use DatasetViewFunc,
) (err error)
func (d *Dataset) WithSelectedRows(
	indexes []int,
	policy ViewPolicy,
	use DatasetViewFunc,
) (err error)
func (d *Dataset) ViewBatches(
	batchSize int,
	random *rand.Rand,
	policy ViewPolicy,
	use DatasetViewFunc,
) (err error)
func (d *Dataset) ViewSplit(
	testFraction float32,
	random *rand.Rand,
	policy ViewPolicy,
	use DatasetSplitViewFunc,
) (err error)

func (b *Batch) WithView(use DatasetViewFunc) (err error)
func (b *Batch) WithRowView(
	start,
	end int,
	use DatasetViewFunc,
) (err error)
func (b *Batch) WithSelectedRows(
	indexes []int,
	policy ViewPolicy,
	use DatasetViewFunc,
) (err error)
```

`WithView` is equivalent to the complete row window. `WithRowView` uses
inclusive/exclusive bounds. `WithSelectedRows` considers indexes contiguous
only when they are non-empty, in range, strictly increasing by one, and
therefore contain no repeats. A contiguous index list uses a view under either
policy. An arbitrary, reordered, skipped, or repeated list fails under
`ViewOnly` and is explicitly copied under `ViewOrCopy`.

`ViewBatches` calls `use` once per non-empty batch. A nil random source
preserves order and uses full or partial contiguous views. A non-nil source
means shuffled batching. Shuffled batching fails under `ViewOnly` before
consuming the source. Under `ViewOrCopy`, it uses the current identity-list and
`rand.Rand.Shuffle` algorithm, then copies each selected batch and publishes it
with `Copied() == true`.

`ViewSplit` uses the current floored test count and requires non-empty train and
test results. With nil randomness it publishes the leading train window and
trailing test window with both `Copied() == false`. With randomness it fails
under `ViewOnly` before consuming the source. Under `ViewOrCopy`, it performs
the current shuffle and prepares both selected copies before calling `use`;
both views report copied.

### Aligned sequence view

Package `data` adds:

```go
type SequenceDatasetView struct { /* unexported fields */ }

func (v *SequenceDatasetView) Validate() (err error)
func (v *SequenceDatasetView) Inputs() (inputs *matrix.Matrix, err error)
func (v *SequenceDatasetView) Targets() (targets *matrix.Matrix, err error)
func (v *SequenceDatasetView) Lengths() (lengths []int, err error)
func (v *SequenceDatasetView) SampleCount() (samples int)
func (v *SequenceDatasetView) InputSize() (features int)
func (v *SequenceDatasetView) TargetSize() (values int)
func (v *SequenceDatasetView) Steps() (steps int)
func (v *SequenceDatasetView) Copied() (copied bool)

type SequenceDatasetViewFunc func(
	view *SequenceDatasetView,
) (err error)

type SequenceDatasetSplitViewFunc func(
	train,
	test *SequenceDatasetView,
) (err error)
```

`Inputs`, `Targets`, and `Lengths` are one indivisible row selection.
`Lengths` returns the selected private slice window without copying. Like the
temporary matrices, the returned slice must not be mutated, retained, or sent
to another goroutine. Go slice headers cannot be revoked after escape, so
expiry can invalidate the view object but cannot make a retained slice header
inaccessible. The API and GoDoc must state that limitation rather than claim
enforced borrowing.

The complete association is validated before callback publication. The input
width remains divisible by `Steps`, every length remains in `[1, Steps]`, and
all row counts match. The zero value and nil or expired receivers are invalid.

Package `data` adds:

```go
func (d *SequenceDataset) WithView(
	use SequenceDatasetViewFunc,
) (err error)
func (d *SequenceDataset) WithRowView(
	start,
	end int,
	use SequenceDatasetViewFunc,
) (err error)
func (d *SequenceDataset) WithSelectedRows(
	indexes []int,
	policy ViewPolicy,
	use SequenceDatasetViewFunc,
) (err error)
func (d *SequenceDataset) ViewBatches(
	batchSize int,
	random *rand.Rand,
	policy ViewPolicy,
	use SequenceDatasetViewFunc,
) (err error)
func (d *SequenceDataset) ViewSplit(
	testFraction float32,
	random *rand.Rand,
	policy ViewPolicy,
	use SequenceDatasetSplitViewFunc,
) (err error)

func (b *SequenceBatch) WithView(
	use SequenceDatasetViewFunc,
) (err error)
func (b *SequenceBatch) WithRowView(
	start,
	end int,
	use SequenceDatasetViewFunc,
) (err error)
func (b *SequenceBatch) WithSelectedRows(
	indexes []int,
	policy ViewPolicy,
	use SequenceDatasetViewFunc,
) (err error)
```

These methods use the exact ordinary selection, batching, splitting, copied
flag, callback, lifetime, and randomness rules while applying one selection
to inputs, targets, and lengths. They never publish one component before the
complete association is ready.

`SequenceLengths` gains no view method. Independent length aliases would let a
caller detach metadata from the matrix row selection and would expose mutable
storage without the aligned context. Safe independent access remains
`Values`, `ValuesInto`, and `SelectRowsInto`.

### Opt-in model fitting

Package `model` adds:

```go
type ViewFitConfig struct {
	FitConfig FitConfig
	Policy    data.ViewPolicy
}

type SequenceViewFitConfig struct {
	SequenceFitConfig SequenceFitConfig
	Policy            data.ViewPolicy
}

func (s *Sequential) FitWithViews(
	trainingData *data.Dataset,
	config ViewFitConfig,
) (history TrainingHistory, err error)

func (s *Sequential) FitWithLengthViews(
	trainingData *data.SequenceDataset,
	config SequenceViewFitConfig,
) (history TrainingHistory, err error)
```

The wrapper configurations are new public value types. Their zero values are
invalid because their nested fit configuration is invalid; `Policy` itself
defaults to strict `ViewOnly`. No additional fields or setters are approved.

`FitWithViews` has the same semantic controls as `Fit` in its nested
configuration. `FitWithLengthViews` has the same controls as
`FitWithLengths`. Validation data remains in the nested existing
configuration.

For `Shuffle == false`, both policies use contiguous batch views and complete
dataset views for training and validation evaluation. They do not copy dataset
inputs, targets, or lengths.

For `Shuffle == true`:

* `ViewOnly` returns an incompatible-policy error during preflight, before
  consuming `Random`, changing model training mode, applying a schedule,
  traversing a layer, or updating a parameter.
* `ViewOrCopy` uses the same identity indexes and caller-provided shuffle as
  the existing fit. Selected training batches are copied into reusable fit
  scratch and remain row-order equivalent to the current path. Complete
  training and validation evaluation use views.

The policy never changes numeric formulas or silently chooses an indirect
matrix representation. The opt-in name means that the caller accepts scoped
aliasing; it does not promise that an explicitly permitted non-contiguous
fallback is a view.

## Why These Package Boundaries

`matrix` owns the minimum primitive because only it can safely derive a
contiguous `data[start*cols:end*cols]` window, synchronize the parent, create
and expire a valid temporary matrix, and clean up temporary residency without
exposing private storage.

`data` owns paired and aligned views, row-selection classification, batching,
splitting, and random permutation. A matrix alone cannot prove that ordinary
targets or sequence lengths use the same rows.

`model` owns the opt-in fit because training-mode state, schedules, callbacks,
early stopping, scratch, layer traversal, optimizer updates, and cleanup are
already `Sequential` responsibilities. Reusing nested existing configurations
preserves their meanings and avoids breaking unkeyed literals.

This boundary does not decide:

* ROADMAP Item 4's long-term sequence/container representation;
* Items 5 and 15's mask, ragged, packed, or loss-reduction semantics; or
* Item 31's general tensor, public device tensor, automatic differentiation,
  or `layer.Layer` contract.

The view remains a dense two-dimensional matrix plus an optional aligned
length slice. It adds no dimension vocabulary, strides, mask propagation, or
general metadata carrier.

## Logical and Physical Layout

For ordinary data, view row `r` maps to owner row `start+r` and preserves:

```text
inputs  [end-start, inputSize]
targets [end-start, targetSize]
```

For sequence data it preserves:

```text
inputs  [end-start, steps*featureSize]
targets [end-start, targetSize]
lengths [end-start]
```

Within a sequence row, `(step, feature)` remains column
`step*featureSize+feature`. A view never moves a column, changes `steps`,
changes the padded suffix, interprets a padded value, or consumes a feature as
metadata.

Positive-row matrix validation means an empty view is never published. The
final mini-batch may be smaller than `batchSize` but must contain at least one
row. Bounds and the already validated parent size prove all slice offsets fit
in `int`; implementation still uses checked multiplication before slicing so
an invalid private parent cannot panic.

## Ownership, Aliasing, and Lifetime

The dataset or batch remains the owner of whole and contiguous-window backing
storage. A copied fallback owns temporary selected storage for the duration of
the callback. The view never owns its original owner's storage.

Before publication, the matrix primitive makes the parent's host values
current. It creates a separate temporary matrix header over the exact host
slice. It does not share the parent's `device.Residency`.

The callback may retain independent results it creates, such as `Clone`,
`Values`, or an owned prediction after honoring that result's existing
lifetime. It may not retain the view, its matrices, or its raw lengths.

On every callback exit:

1. any temporary matrix execution must already have completed under the
   existing synchronous public model/matrix boundary;
2. temporary matrix residency is synchronized and released if it was created;
3. temporary matrix fields and view state are expired; and
4. callback and cleanup errors are returned with both causes when needed.

A callback panic performs the same expiry and resource cleanup and then
re-panics. There is no public `Close`.

View values may be copied as Go values, but all copies share private active
state and expire together. Retained temporary matrix pointers are invalidated
because the implementation clears the pointed-to matrix value. A retained raw
length slice cannot be revoked; retaining it is a contract violation.

The receiver and temporary storage keep the original backing array reachable
while the callback is active. Parent garbage collection therefore cannot
shorten a valid callback lifetime. After expiry, no owner-lifetime guarantee
is made to escaped aliases.

Nested `Matrix.WithRowView` calls on an active row view are supported and map
relative bounds into the immediate parent window. They share the original
backing array, have their own temporary matrix state, and must finish before
the outer callback. Sequential overlapping read-only windows are supported.
Data view APIs do not add a nested view method; callers use one owner operation
at a time.

Later non-overlapping owner calls do not invalidate completed independent
copies. No live data view survives long enough for a later normal call to
invalidate it.

## Mutation and Concurrency

Every view is read-only by contract. The API does not expose a setter on a view
type, but its matrix accessor necessarily returns the existing mutable
`*matrix.Matrix`, and its sequence accessor necessarily returns `[]int` for
the current length-aware consumer. The library cannot enforce immutability.

The following are supported:

* any number of sequential observations during one callback;
* passing temporary matrices as read-only inputs to synchronous layers,
  losses, metrics, and model operations;
* nested sequential read-only matrix windows;
* independent view operations on distinct, non-aliasing datasets, batches,
  matrices, models, and optimizers in separate goroutines.

The following are forbidden:

* concurrent view operations on the same owner;
* concurrent read/read use of the same view from multiple goroutines;
* every read/write or write/write combination involving an owner, view,
  overlapping view, length slice, matrix, model, layer, or optimizer;
* reentrant view publication from the same data owner;
* retaining or asynchronously consuming any view component.

The repository makes no same-owner concurrent-read promise because host
synchronization, temporary residency, layer caches, and existing dataset
objects were not designed as concurrent objects. `go test -race` must prove
the supported distinct-object pattern and ordinary synchronous callbacks. It
must not intentionally execute a forbidden data race merely to demonstrate
that the race detector can find one.

Constructor arguments and results from safe copying accessors remain
independent. Mutating those independent objects remains supported and cannot
change an owner or view.

## Matrix Residency and CPU/Metal Behavior

A row view always begins from current host storage:

1. validate the complete parent;
2. complete any relevant pending producer;
3. download the parent once if its device revision is newer;
4. derive the host row window; and
5. publish a temporary matrix with no residency.

This is a synchronization or transfer when necessary, but it is not a bulk
host-to-host value copy. Benchmarks report any Metal transfer separately from
logical data copy volume.

The parent retains its complete residency record and full-size device buffer.
The row view never points a shortened matrix at that full buffer and never
shares revision counters.

If a synchronous supported model operation consumes a row view, the temporary
matrix may lazily create its own residency sized exactly to the view. It may
upload that window, bind to one execution, and finish host-current or
synchronized. Callback cleanup synchronizes if needed, releases the
temporary device buffer, and clears the temporary matrix. The parent residency
is unchanged because compliant use never writes the shared host slice.

CPU and Metal results follow their existing reference and tolerance contracts.
Unsupported builds and unavailable or ineligible devices remain CPU/SIMD
fallbacks. Device initialization or operational failure after eligibility is
returned with existing context; a view path does not silently retry.

`Values`, `ValuesInto`, `At`, `Clone`, `CopyFrom`, and every operation on an
active temporary matrix retain their current synchronization semantics.
`Clone` and copies made from a view own independent storage. There is no public
view `Release`; callback cleanup owns release.

A prohibited mutation through a view is not made coherent with the owner's
independent residency. That is why the design describes read-only intent and
forbids mutation instead of promising mutable aliases.

## Batching, Selection, Splitting, and Randomness

The complete case table is:

| Operation | `ViewOnly` | `ViewOrCopy` |
| --- | --- | --- |
| Whole dataset or batch | Zero-copy contiguous view. | Zero-copy contiguous view. |
| Ordered contiguous row window | Zero-copy contiguous view. | Zero-copy contiguous view. |
| Ordered full mini-batch | Zero-copy contiguous view. | Zero-copy contiguous view. |
| Ordered partial final mini-batch | Zero-copy contiguous view. | Zero-copy contiguous view. |
| Strictly contiguous explicit indexes | Zero-copy contiguous view. | Zero-copy contiguous view. |
| Skipped, reversed, or arbitrary indexes | Reject. | Explicit selected copy; `Copied == true`. |
| Repeated indexes | Reject. | Explicit selected copy with duplicates; `Copied == true`. |
| Shuffled batches | Reject before random consumption. | Current seeded permutation and copied batches. |
| Ordered split | Two zero-copy contiguous views. | Two zero-copy contiguous views. |
| Shuffled split | Reject before random consumption. | Current seeded permutation and two selected copies. |

There is no indirect matrix, gather matrix, hidden copy in `ViewOnly`, or
automatic fallback based on size.

For equivalent seeds and initial random states, `ViewOrCopy` calls
`rand.Rand.Shuffle` once with the same identity-list length and swap callback
as the current copied operation. It therefore consumes the same random values
and produces the same row order.

Complete receiver, scalar, policy, callback, count, fraction, index, and
capability validation occurs before random consumption. Once traversal starts,
a callback error stops later batches. Randomness already consumed for that
valid traversal is not rolled back.

The random source is never retained. Nil means ordered behavior and consumes
no random values.

## Validation and Failure Behavior

Error prefixes are:

```text
matrix: row view ...
data: dataset view ...
data: batch view ...
data: sequence dataset view ...
data: sequence batch view ...
model: view fit ...
model: length-view fit ...
```

The validation order is:

### Matrix

1. receiver and structural matrix validity;
2. row bounds and checked offsets;
3. non-nil callback;
4. host synchronization;
5. temporary state construction;
6. callback invocation.

No callback is invoked and no view is published on failure before step six.

### Data whole and row views

1. owner and complete paired or aligned association;
2. row bounds where applicable;
3. non-nil callback;
4. complete temporary input, target, and length state;
5. callback invocation.

### Selected rows, batches, and splits

1. owner association;
2. scalar sizes or fraction and non-empty derived counts;
3. every index where applicable;
4. policy validity;
5. non-nil callback;
6. policy/capability compatibility;
7. random permutation when applicable;
8. preparation of every component for the next publication;
9. callback invocation.

One view publication is all-or-nothing. A split callback is not invoked until
both train and test views are ready. A sequence callback is not invoked until
both matrices and the complete length selection are ready. Temporary
allocations from a failed preparation are discarded.

Batch traversal may already have completed earlier callbacks when a later
callback or operational synchronization fails. The returned error identifies
the batch number, and no later callback runs. This is traversal progress, not
partial publication of the failing batch.

### Views and model operations

View accessors validate non-nil active state, unexpired matrices, complete row
alignment, steps, and length range before returning aliases.

Model preflight validates:

1. receiver, ready graph, and ordinary versus length-aware graph;
2. training and validation datasets;
3. nested existing fit configuration;
4. `ViewPolicy`;
5. incompatibility of `ViewOnly` with `Shuffle`;
6. model/dataset input, target, and step compatibility.

Preflight finishes before random consumption, learning-rate schedule
invocation, view publication, training-mode change, layer traversal, gradient
work, or parameter updates.

Callback-returned errors are wrapped with owner or operation context.
Cleanup errors are joined rather than replacing the callback or operation
error. Panic cleanup expires all views, restores model state where the
existing model boundary promises it, and re-panics.

## Training and Numeric Behavior

Data views add no forward or backward formula. An active matrix view has the
same shape and values as `SelectRows` for the same contiguous rows. Existing
layers may read it through unchanged `Forward` methods. Existing layer input
caches keep the values required for later backward work.

The two opt-in fits preserve:

* epoch and row order;
* full and partial batch boundaries;
* prediction, loss, and accuracy values;
* forward and backward formulas and cache rules;
* summed gradient accumulation and loss-controlled scaling;
* parameter discovery and order;
* gradient clipping observation and wrapper order;
* optimizer calls, state, updates, and successful gradient reset;
* schedule timing;
* training and validation evaluation order;
* history, callback timing, partial history, and callback errors;
* early-stopping monitoring and decisions;
* training-mode transitions and restoration;
* length-aware association invalidation and `GatherLastValid` snapshots; and
* execution finish, abort, resource release, and panic cleanup.

`GatherLastValid` still snapshots logical lengths needed by backward. That
existing layer cache is not a hidden data-view fallback. Benchmarks distinguish
zero-copy data acquisition from downstream copies that both workflows require.

Equivalent CPU copy and view workflows are expected to be bit-identical when
they execute the same operations in the same order. CPU/Metal comparisons use
the tolerances in [metal.md](metal.md), including `2e-4` absolute and relative
tolerance for a complete supported training step.

No dataset value is mutated by a compliant fit. A requested strict operation
that cannot remain a view returns its contextual incompatibility error instead
of silently copying.

## Parameters, Persistence, and Determinism

The feature adds no `optimizer.Parameter`. It changes no layer or within-layer
parameter order, gradient scale, gradient accumulation, clipping phase,
optimizer ownership, or reset behavior.

`Sequential.Save` remains format `neuralnetwork.sequential`, version `1`.
Views, callbacks, policies, dataset references, copied selections, random
sources, row indexes, fit scratch, raw lengths, temporary matrix residency,
and view lifetime state are runtime values and are never serialized.

Existing ANN, CNN, fixed-length RNN, explicit-length RNN, and
`gather_last_valid` documents retain their encoding and load behavior.
Loading creates no data view or fit configuration.

The view feature creates no random source, seed, map iteration, goroutine, or
implicit schedule. Equivalent parameters, data, configuration, policy, and
caller random state produce the deterministic parity described above.

## Benchmark Contract

Section 2 records the copying baseline before production implementation.
Sections 3 through 6 add matching view cases without rewriting or deleting the
before results.

### Fixed shapes

Use deterministic finite fixtures with these public shapes:

| Case | Samples | Inputs | Targets | Steps | Features | Batch sizes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ordinary board | `4096` | `256` | `16` | n/a | n/a | `256` |
| Ordinary partial board | `4100` | `256` | `16` | n/a | n/a | `256` |
| Wide sequence board | `512` | `4096` | `8` | `128` | `32` | `256`, `192` |
| Wide sequence validation | `256` | `4096` | `8` | `128` | `32` | whole board |

The sequence length fixture cycles deterministically through
`1..128`. Targets derive from the same row number as inputs and lengths so an
alignment error changes the numeric sink.

The ordinary fit graph is a deterministic `Dense(256, 32) -> Linear ->
Dense(32, 16)` model with mean-squared error and SGD. The length-aware fit
graph is a deterministic `SimpleRNN([128, 32], 8) -> GatherLastValid ->
Dense(8, 8)` model with mean-squared error and SGD. Initializers use explicit
constant or caller-seeded finite values.

Fit cases use one epoch. Ordinary batches contain `256` rows. Sequence batches
contain `256` rows for the representative hundreds-by-thousands case and
`192` rows for the ordered partial-final case (`192`, `192`, `128`).
Training and validation evaluation are measured both separately and as part of
the complete epoch.

### Required operation pairs

Data benchmarks record ordinary and sequence constructor copies as baseline-only
compatibility controls because borrowed constructors are rejected. They pair
current copied and future viewed behavior for:

* whole-dataset inputs, targets, and aligned lengths;
* whole-batch access;
* ordered full and partial batching;
* seeded shuffled batching;
* ordered and seeded shuffled splitting;
* contiguous selected rows;
* arbitrary and repeated selected rows under the explicit fallback; and
* strict non-contiguous rejection outside timed setup.

Model benchmarks pair:

* full training evaluation;
* full validation evaluation;
* one complete ordered ordinary fit epoch;
* one complete ordered length-aware fit epoch;
* one `ViewOrCopy` shuffled ordinary fit epoch; and
* one `ViewOrCopy` shuffled length-aware fit epoch.

Cold cases include owned result or scratch construction. Warm cases perform
one untimed warm-up and reuse all operation-owned capacity that the public
contract permits.

### Accounting

Every benchmark calls `b.ReportAllocs`. Setup, deterministic fixture
construction, model construction, and untimed numeric verification stay
outside the timed region. Failures call `b.Fatalf`.

Package-level sinks retain a scalar checksum, returned history, copied flag,
or owned result as appropriate. A view sink observes selected first and last
values with `At`; it must not call a copying accessor in order to defeat
compiler elimination.

Report:

* `ns/op`, `B/op`, and `allocs/op`;
* every matrix shape and batch size;
* logical bytes read;
* calculated logical bytes copied at the data boundary;
* matrix bytes as `elementCount*4`;
* length bytes as `lengthCount*(strconv.IntSize/8)`; and
* copied versus viewed or rejected policy.

Logical copy volume counts value bytes copied between distinct data-owned or
fit-scratch storage. It excludes index-slice initialization, row permutation,
matrix headers, callbacks, allocator metadata, unchanged layer caches,
predictions, gradients, and parameter work. Metal uploads and downloads are
reported separately when a Metal-tagged comparison is made.

A `ViewOrCopy` shuffled case counts its selected batch copies honestly. It does
not label those bytes zero-copy merely because evaluation uses views.

### Commands

Record `go version`, `go env GOOS GOARCH CGO_ENABLED`, hardware, OS version,
power mode, and the clean commit. Use:

```sh
GOMAXPROCS=1 go test ./data -run '^$' \
  -bench '^Benchmark_(DataViews|SequenceDataViews)' \
  -benchmem -count=10 -benchtime=500ms

GOMAXPROCS=1 go test ./model -run '^$' \
  -bench '^Benchmark_(FitDataViews|FitSequenceDataViews)' \
  -benchmem -count=10 -benchtime=3x
```

Save exact raw output and summarized medians in
`Benchmarks_data_views.md`. Do not use elapsed-time assertions in unit tests.
Timing is evidence, not a correctness gate.

### Material improvement

Acceptance requires unchanged numeric results plus:

* whole, contiguous-batch, ordered-split, and ordered-evaluation views copy
  zero logical input, target, and length bytes at the data boundary;
* those view cases reduce calculated logical copied bytes by at least `95%`
  and measured `B/op` by at least `90%` versus the matched allocating copied
  operation;
* ordered complete fit reduces data-boundary copy volume by at least `95%`;
* `ViewOrCopy` shuffled complete fit reduces total data-boundary copy volume by
  at least `40%` by removing complete training and validation evaluation
  copies while retaining one selected training copy;
* the existing default warmed epoch allocation tests remain at zero;
  opt-in-fit metadata allocations, if any, stay bounded independently of row
  width and sample count and allocate no input-, target-, or length-sized
  storage; and
* no timing case regresses by both more than `10%` and more than `100us`
  without a recorded explanation and maintainer acceptance.

The byte and allocation gates are primary. A noisy timing result cannot make a
correct byte-copy reduction fail or hide a copy-volume failure.

## Alternatives Considered

### Direct internal matrix pointers

Returning the dataset's stored `*matrix.Matrix` permanently would be simple,
but it would create an unbounded mutable alias, let callers retain it, and make
device revision ownership ambiguous. Rejected.

### Permanent contiguous matrix windows

A long-lived `*matrix.Matrix` sharing storage would need shared host/device
revisions, buffer offsets, detachment, mutation publication, release, and
owner lifetime. That is a general matrix-view architecture larger than the
measured data-consumption need. Rejected in favor of a scoped row view with
separate temporary residency.

### Separate matrix and length accessors

Independent input, target, or length view calls could use different windows or
permutations. That is especially unsafe for logical lengths. Rejected in favor
of paired and aligned composite view values.

### Independent SequenceLengths view

It would expose mutable private integer storage without carrying the selected
sequence matrices. Rejected. Length aliases exist only inside
`SequenceDatasetView`.

### Read-only matrix interface

A small interface could enforce observation-only methods, but existing
`layer.Layer`, losses, metrics, and model operations require concrete
`*matrix.Matrix` arguments. Adapters would either copy or create a second
repository-wide numeric contract. Rejected for this milestone.

### Callback-free view return

A returned long-lived view would need public close/invalidation and still
could not prevent use after close or slice escape. Scoped callbacks bound
normal usage and permit deterministic cleanup. The callback limitation is
documented honestly.

### Indirect or gather matrix

Representing arbitrary row indexes without copying would require every matrix
operation and device kernel to understand indirection. It would decide a
broader container/tensor question and could make random access slower than the
existing copy. Rejected.

### Hidden copying fallback

An API named as zero-copy must not quietly select rows into new matrices.
Rejected. `ViewOrCopy` is explicit before the operation, and `Copied` reports
what occurred.

### Reject every shuffled operation

Strict rejection is available as `ViewOnly`, but making it the only option
would prevent an opt-in fit from retaining deterministic shuffled training
while still eliminating complete evaluation copies. The explicit fallback
supports that use without mislabeling it.

### Borrowed constructors

Borrowed dataset constructors would weaken construction-time source isolation
and create ownership that survives callbacks. Rejected. All constructors keep
copying.

### Automatic default fit optimization

Changing `Fit` or `FitWithLengths` to alias data would silently change their
ownership and mutation-risk profile. Rejected. Callers choose the additive
methods explicitly.

### Fields on existing fit configurations

Adding fields can break unkeyed composite literals and would make existing fit
calls appear view-aware. Rejected in favor of wrapper configurations.

### General immutable matrices, copy-on-write, and tensors

Each is a repository-wide abstraction without evidence that the data-view
milestone needs it. Rejected and left to separate roadmap decisions.

## Compatibility and Non-Goals

The implementation must preserve:

* copying constructors `NewDataset`, `NewSequenceLengths`,
  `NewSequenceDataset`, and `LoadCSV`;
* every current allocating and destination accessor;
* owned-copy `Batches` and `Split` behavior and signatures;
* `Matrix.Values`, `Clone`, `CopyFrom`, `SelectRows`, and `SelectRowsInto`;
* `Predict`, `Backward`, `TrainBatch`, `Fit`, `PredictWithLengths`,
  `BackwardWithLengths`, `TrainBatchWithLengths`, and `FitWithLengths`;
* `FitConfig` and `SequenceFitConfig` field sets;
* ANN, CNN, fixed-length RNN, explicit-length RNN, and clipping behavior;
* caller-controlled randomness and deterministic row order; and
* sequential format `neuralnetwork.sequential`, version `1`.

This milestone does not add:

* general tensor, ragged, packed, strided, masked, or device-tensor values;
* zero-length sequences, left padding, interior holes, or masked losses;
* general mutable or immutable matrix aliases;
* arbitrary zero-copy gathers or indirect device buffers;
* memory mapping, streaming, external storage, prefetch, or data loaders;
* borrowed constructors or changed safe defaults;
* transactions, copy-on-write, mutation tracking, or locks around datasets;
* new layer formulas, parameters, gradient behavior, optimizers, or kernels;
* public device controls, asynchronous execution, or new Metal kernels; or
* serialization of runtime data, views, policies, callbacks, or scratch.

Implementation status is updated in this document section by section. It must
not be described as implemented and verified until every declaration and
behavior above has production code, focused tests, benchmarks, caller
documentation, and repository-wide verification.
