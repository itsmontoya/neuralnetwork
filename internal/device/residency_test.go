package device

import (
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"
)

func Test_ResidencyFinalizerReleasesUnreachableStates(t *testing.T) {
	var tests []struct {
		name  string
		state string
	}
	tests = []struct {
		name  string
		state string
	}{
		{name: "host newer", state: "host-newer"},
		{name: "device newer", state: "device-newer"},
		{name: "failed", state: "failed"},
	}

	var test struct {
		name  string
		state string
	}
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				backendValue *testBackend
				runtimeValue *Runtime
				snapshot     ResourceSnapshot
				deadline     time.Time
			)

			backendValue = newTestBackend()
			runtimeValue = newRuntime(backendValue)
			createUnreachableResidency(t, runtimeValue, test.state)
			deadline = time.Now().Add(2 * time.Second)
			for {
				runtime.GC()
				runtime.Gosched()
				snapshot = runtimeValue.ResourceSnapshot()
				if snapshot.LiveBuffers == 0 {
					break
				}
				if time.Now().After(deadline) {
					t.Fatalf("unreachable %s resources = %+v", test.state, snapshot)
				}
				time.Sleep(time.Millisecond)
			}
			if snapshot.CreatedBuffers != snapshot.ReleasedBuffers {
				t.Fatalf("unreachable %s resources are unbalanced: %+v", test.state, snapshot)
			}
		})
	}
}

func Test_ResidencyStateTransitions(t *testing.T) {
	var (
		backendValue *testBackend
		runtimeValue *Runtime
		residency    *Residency
		buffer       *Buffer
		snapshot     ResidencySnapshot
		resources    ResourceSnapshot
		host         []float32
		allocated    bool
		uploaded     bool
		downloaded   bool
		err          error
	)

	backendValue = newTestBackend()
	runtimeValue = newRuntime(backendValue)
	if residency, err = NewResidency(runtimeValue, 3); err != nil {
		t.Fatalf("NewResidency returned error: %v", err)
	}
	host = []float32{1, 2, 3}
	requireResidencyState(t, residency, "new", 1, 1, 0)

	buffer, allocated, uploaded, err = residency.EnsureDevice(host)
	if err != nil {
		t.Fatalf("EnsureDevice returned error: %v", err)
	}
	if !allocated || !uploaded {
		t.Fatalf("EnsureDevice allocated=%t uploaded=%t, want both true", allocated, uploaded)
	}
	requireResidencyState(t, residency, "synchronized", 1, 1, 1)

	if _, allocated, uploaded, err = residency.EnsureDevice(host); err != nil {
		t.Fatalf("second EnsureDevice returned error: %v", err)
	}
	if allocated || uploaded {
		t.Fatalf("second EnsureDevice allocated=%t uploaded=%t, want both false", allocated, uploaded)
	}

	host[0] = 4
	if err = residency.MarkHostWrite(); err != nil {
		t.Fatalf("MarkHostWrite returned error: %v", err)
	}
	requireResidencyState(t, residency, "host-newer", 2, 2, 1)
	if _, allocated, uploaded, err = residency.EnsureDevice(host); err != nil {
		t.Fatalf("EnsureDevice after host write returned error: %v", err)
	}
	if allocated || !uploaded {
		t.Fatalf("EnsureDevice after host write allocated=%t uploaded=%t, want false/true", allocated, uploaded)
	}
	requireResidencyState(t, residency, "synchronized", 2, 2, 2)

	if buffer, allocated, err = residency.BeginDeviceWrite(); err != nil {
		t.Fatalf("BeginDeviceWrite returned error: %v", err)
	}
	if !allocated {
		t.Fatal("BeginDeviceWrite did not allocate staging")
	}
	if err = buffer.Upload([]float32{7, 8, 9}); err != nil {
		t.Fatalf("Upload staging returned error: %v", err)
	}
	requireResidencyState(t, residency, "pending", 2, 2, 2)
	if err = residency.PublishDeviceWrite(buffer); err != nil {
		t.Fatalf("PublishDeviceWrite returned error: %v", err)
	}
	requireResidencyState(t, residency, "device-newer", 3, 2, 3)

	downloaded, err = residency.EnsureHost(host)
	if err != nil {
		t.Fatalf("EnsureHost returned error: %v", err)
	}
	if !downloaded {
		t.Fatal("EnsureHost did not download a device-newer value")
	}
	requireFloat32Values(t, host, []float32{7, 8, 9})
	requireResidencyState(t, residency, "synchronized", 3, 3, 3)

	if err = residency.MarkPooled(); err != nil {
		t.Fatalf("MarkPooled returned error: %v", err)
	}
	requireResidencyState(t, residency, "pooled", 3, 3, 3)
	if err = residency.ReusePooled(); err != nil {
		t.Fatalf("ReusePooled returned error: %v", err)
	}
	requireResidencyState(t, residency, "synchronized", 3, 3, 3)

	snapshot = residency.Snapshot()
	if snapshot.Uploads != 2 || snapshot.Downloads != 1 || snapshot.AvoidedUploads != 1 {
		t.Fatalf("transfer snapshot = %+v, want two uploads, one download, one avoided upload", snapshot)
	}
	if snapshot.ProposedRevisions != 1 || snapshot.Publications != 1 || snapshot.DiscardedPublications != 0 {
		t.Fatalf("publication snapshot = %+v, want one successful publication", snapshot)
	}
	if err = residency.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}
	requireResidencyState(t, residency, "released", 3, 3, 0)
	if resources = runtimeValue.ResourceSnapshot(); resources.LiveBuffers != 0 {
		t.Fatalf("live buffers after release = %d, want 0", resources.LiveBuffers)
	}
}

func Test_ResidencyRepeatedHostDeviceAlternation(t *testing.T) {
	type transition struct {
		name        string
		action      string
		state       string
		logical     uint64
		host        uint64
		device      uint64
		hostValues  []float32
		deviceValue []float32
	}

	var (
		runtimeValue *Runtime
		residency    *Residency
		buffer       *Buffer
		hostValues   []float32
		transitions  []transition
		current      transition
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	if residency, err = NewResidency(runtimeValue, 2); err != nil {
		t.Fatalf("NewResidency returned error: %v", err)
	}
	hostValues = []float32{1, 2}
	transitions = []transition{
		{name: "initial upload", action: "upload", state: "synchronized", logical: 1, host: 1, device: 1},
		{name: "first host write", action: "host-write", state: "host-newer", logical: 2, host: 2, device: 1, hostValues: []float32{3, 4}},
		{name: "first re-upload", action: "upload", state: "synchronized", logical: 2, host: 2, device: 2},
		{name: "first device write", action: "device-write", state: "device-newer", logical: 3, host: 2, device: 3, deviceValue: []float32{5, 6}},
		{name: "first download", action: "download", state: "synchronized", logical: 3, host: 3, device: 3},
		{name: "second host write", action: "host-write", state: "host-newer", logical: 4, host: 4, device: 3, hostValues: []float32{7, 8}},
		{name: "second re-upload", action: "upload", state: "synchronized", logical: 4, host: 4, device: 4},
		{name: "second device write", action: "device-write", state: "device-newer", logical: 5, host: 4, device: 5, deviceValue: []float32{9, 10}},
		{name: "second download", action: "download", state: "synchronized", logical: 5, host: 5, device: 5},
	}

	for _, current = range transitions {
		t.Run(current.name, func(t *testing.T) {
			switch current.action {
			case "host-write":
				copy(hostValues, current.hostValues)
				if err = residency.MarkHostWrite(); err != nil {
					t.Fatalf("MarkHostWrite returned error: %v", err)
				}
			case "device-write":
				if buffer, _, err = residency.BeginDeviceWrite(); err != nil {
					t.Fatalf("BeginDeviceWrite returned error: %v", err)
				}
				if err = buffer.Upload(current.deviceValue); err != nil {
					t.Fatalf("Upload staging returned error: %v", err)
				}
				if err = residency.PublishDeviceWrite(buffer); err != nil {
					t.Fatalf("PublishDeviceWrite returned error: %v", err)
				}
			case "download":
				if _, err = residency.EnsureHost(hostValues); err != nil {
					t.Fatalf("EnsureHost returned error: %v", err)
				}
			case "upload":
				if _, _, _, err = residency.EnsureDevice(hostValues); err != nil {
					t.Fatalf("EnsureDevice returned error: %v", err)
				}
			default:
				t.Fatalf("unsupported transition action %q", current.action)
			}
			requireResidencyState(
				t,
				residency,
				current.state,
				current.logical,
				current.host,
				current.device,
			)
		})
	}
	requireFloat32Values(t, hostValues, []float32{9, 10})
	if err = residency.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}
}

func Test_ResidencyFailedWritePreservesCommittedValue(t *testing.T) {
	var (
		runtimeValue *Runtime
		residency    *Residency
		buffer       *Buffer
		host         []float32
		snapshot     ResidencySnapshot
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	if residency, err = NewResidency(runtimeValue, 2); err != nil {
		t.Fatalf("NewResidency returned error: %v", err)
	}
	host = []float32{2, 4}
	if _, _, _, err = residency.EnsureDevice(host); err != nil {
		t.Fatalf("EnsureDevice returned error: %v", err)
	}
	if buffer, _, err = residency.BeginDeviceWrite(); err != nil {
		t.Fatalf("BeginDeviceWrite returned error: %v", err)
	}
	if err = buffer.Upload([]float32{8, 16}); err != nil {
		t.Fatalf("Upload staging returned error: %v", err)
	}
	if err = residency.FailDeviceWrite(buffer, errors.New("injected execution failure")); err != nil {
		t.Fatalf("FailDeviceWrite returned error: %v", err)
	}
	snapshot = residency.Snapshot()
	if snapshot.State != "failed" || snapshot.DiscardedPublications != 1 ||
		!strings.Contains(snapshot.LastError, "injected execution failure") {
		t.Fatalf("failed snapshot = %+v", snapshot)
	}
	if _, err = residency.EnsureHost(host); err == nil || !strings.Contains(err.Error(), "injected execution failure") {
		t.Fatalf("EnsureHost failed-state error = %v", err)
	}
	if err = residency.RestoreCommitted(); err != nil {
		t.Fatalf("RestoreCommitted returned error: %v", err)
	}
	requireResidencyState(t, residency, "synchronized", 1, 1, 1)
	if _, err = residency.EnsureHost(host); err != nil {
		t.Fatalf("EnsureHost after restore returned error: %v", err)
	}
	requireFloat32Values(t, host, []float32{2, 4})
	if err = residency.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}
}

func Test_ResidencyReleaseRequiresCurrentHost(t *testing.T) {
	var (
		runtimeValue *Runtime
		residency    *Residency
		buffer       *Buffer
		host         []float32
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	if residency, err = NewResidency(runtimeValue, 1); err != nil {
		t.Fatalf("NewResidency returned error: %v", err)
	}
	host = []float32{3}
	if _, _, _, err = residency.EnsureDevice(host); err != nil {
		t.Fatalf("EnsureDevice returned error: %v", err)
	}
	if buffer, _, err = residency.BeginDeviceWrite(); err != nil {
		t.Fatalf("BeginDeviceWrite returned error: %v", err)
	}
	if err = buffer.Upload([]float32{6}); err != nil {
		t.Fatalf("Upload staging returned error: %v", err)
	}
	if err = residency.PublishDeviceWrite(buffer); err != nil {
		t.Fatalf("PublishDeviceWrite returned error: %v", err)
	}
	if err = residency.Release(); err == nil || !strings.Contains(err.Error(), "host value is stale") {
		t.Fatalf("Release device-newer error = %v", err)
	}
	if _, err = residency.EnsureHost(host); err != nil {
		t.Fatalf("EnsureHost returned error: %v", err)
	}
	if err = residency.Release(); err != nil {
		t.Fatalf("Release after download returned error: %v", err)
	}
}

func Test_ResidencyRevisionOverflowRebases(t *testing.T) {
	var (
		runtimeValue *Runtime
		residency    *Residency
		snapshot     ResidencySnapshot
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	if residency, err = NewResidency(runtimeValue, 1); err != nil {
		t.Fatalf("NewResidency returned error: %v", err)
	}
	residency.logicalRevision = ^uint64(0)
	residency.hostRevision = ^uint64(0)
	if err = residency.MarkHostWrite(); err != nil {
		t.Fatalf("MarkHostWrite returned error: %v", err)
	}
	snapshot = residency.Snapshot()
	if snapshot.LogicalRevision != 2 || snapshot.HostRevision != 2 || snapshot.DeviceRevision != 0 {
		t.Fatalf("rebased snapshot = %+v, want current host revision 2", snapshot)
	}
}

func requireResidencyState(
	tb testing.TB,
	residency *Residency,
	wantState string,
	wantLogical,
	wantHost,
	wantDevice uint64,
) {
	tb.Helper()

	var snapshot ResidencySnapshot
	snapshot = residency.Snapshot()
	if snapshot.State != wantState ||
		snapshot.LogicalRevision != wantLogical ||
		snapshot.HostRevision != wantHost ||
		snapshot.DeviceRevision != wantDevice {
		tb.Fatalf(
			"snapshot = %+v, want state=%s logical=%d host=%d device=%d",
			snapshot,
			wantState,
			wantLogical,
			wantHost,
			wantDevice,
		)
	}
}

func createUnreachableResidency(tb testing.TB, runtimeValue *Runtime, state string) {
	tb.Helper()

	var (
		residency *Residency
		buffer    *Buffer
		host      []float32
		err       error
	)

	host = []float32{1, 2}
	if residency, err = NewResidency(runtimeValue, 2); err != nil {
		tb.Fatalf("NewResidency returned error: %v", err)
	}
	if _, _, _, err = residency.EnsureDevice(host); err != nil {
		tb.Fatalf("EnsureDevice returned error: %v", err)
	}
	switch state {
	case "host-newer":
		if err = residency.MarkHostWrite(); err != nil {
			tb.Fatalf("MarkHostWrite returned error: %v", err)
		}
	case "device-newer":
		if buffer, _, err = residency.BeginDeviceWrite(); err != nil {
			tb.Fatalf("BeginDeviceWrite returned error: %v", err)
		}
		if err = buffer.Upload([]float32{3, 4}); err != nil {
			tb.Fatalf("Upload staging returned error: %v", err)
		}
		if err = residency.PublishDeviceWrite(buffer); err != nil {
			tb.Fatalf("PublishDeviceWrite returned error: %v", err)
		}
	case "failed":
		if buffer, _, err = residency.BeginDeviceWrite(); err != nil {
			tb.Fatalf("BeginDeviceWrite returned error: %v", err)
		}
		if err = residency.FailDeviceWrite(buffer, errors.New("injected failure")); err != nil {
			tb.Fatalf("FailDeviceWrite returned error: %v", err)
		}
	default:
		tb.Fatalf("unsupported residency state %q", state)
	}
	runtime.KeepAlive(residency)
}
