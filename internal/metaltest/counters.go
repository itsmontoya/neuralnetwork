//go:build darwin && cgo && metal && !purego

// Package metaltest provides repository-private Metal diagnostics for tests and benchmarks.
package metaltest

import (
	"sync"
	"sync/atomic"
)

var (
	enabled            atomic.Bool
	bufferCreations    atomic.Uint64
	inputUploads       atomic.Uint64
	resultDownloads    atomic.Uint64
	commandSubmissions atomic.Uint64
	waits              atomic.Uint64
	errorMutex         sync.Mutex
	lastError          string
)

// Counters captures synchronous Metal bridge activity.
type Counters struct {
	BufferCreations    uint64
	InputUploads       uint64
	ResultDownloads    uint64
	CommandSubmissions uint64
	Waits              uint64
	LastError          string
}

// Enable starts recording Metal bridge activity.
func Enable() {
	Reset()
	enabled.Store(true)
}

// Disable stops recording Metal bridge activity.
func Disable() {
	enabled.Store(false)
}

// Enabled reports whether Metal bridge activity is being recorded.
func Enabled() (ok bool) {
	ok = enabled.Load()
	return ok
}

// Reset clears all recorded Metal bridge activity.
func Reset() {
	bufferCreations.Store(0)
	inputUploads.Store(0)
	resultDownloads.Store(0)
	commandSubmissions.Store(0)
	waits.Store(0)

	errorMutex.Lock()
	lastError = ""
	errorMutex.Unlock()
}

// RecordBridgeActivity adds observed synchronous bridge events.
func RecordBridgeActivity(
	createdBuffers,
	uploadedInputs,
	downloadedResults,
	submittedCommands,
	waitedCommands uint64,
) {
	if !enabled.Load() {
		return
	}

	bufferCreations.Add(createdBuffers)
	inputUploads.Add(uploadedInputs)
	resultDownloads.Add(downloadedResults)
	commandSubmissions.Add(submittedCommands)
	waits.Add(waitedCommands)
}

// RecordFailure records the latest Metal bridge failure.
func RecordFailure(message string) {
	if !enabled.Load() {
		return
	}

	errorMutex.Lock()
	lastError = message
	errorMutex.Unlock()
}

// Snapshot returns the current Metal bridge counters.
func Snapshot() (counters Counters) {
	counters.BufferCreations = bufferCreations.Load()
	counters.InputUploads = inputUploads.Load()
	counters.ResultDownloads = resultDownloads.Load()
	counters.CommandSubmissions = commandSubmissions.Load()
	counters.Waits = waits.Load()

	errorMutex.Lock()
	counters.LastError = lastError
	errorMutex.Unlock()
	return counters
}
