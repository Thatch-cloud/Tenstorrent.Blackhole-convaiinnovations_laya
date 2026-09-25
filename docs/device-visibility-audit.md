# Pinned UMD device discovery audit

Read-only source inspection, 2026-09-25. No device initialization was performed.
This supplements the earlier bounded compiler audit; it does not provide a lease
or qualify a shared-host hardware run.

The physical-stack source chain inspected is TT-XLA
`3bf6e4201f008fdf5a0ce5d244bc83ad155ee328`, TT-MLIR
`c282803f8f628881e805b2bbf6dc5bf53a0d230d`, TT-Metal
`d04395ed862b4c65eb6877000c40200f456cb74e`, and UMD
`8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404`. The offline compiler wheel is a
different revision and does not inherit this audit.

## Visibility is applied before device initialization

In pinned UMD `device/pcie/pci_device.cpp:236`, `enumerate_devices()` reads
`TT_VISIBLE_DEVICES`. An absent variable returns all enumerated devices; an empty
variable returns none. A full PCI BDF is matched against discovered BDF strings.
Integer selectors refer to the BDF-sorted logical inventory, not necessarily the
numeric character-device suffix.

`device/topology/topology_discovery.cpp:156` gets this filtered PCI list before
calling `TTDevice::create` and `init_device` for each selected device. This closes
the earlier uncertainty about the filter's position relative to initialization
in this source revision.

However, filtering builds a BDF map first. `get_bdf_to_device_id_map()` at
`device/pcie/pci_device.cpp:1143` visits every numeric device node and calls
`read_device_info()`. That helper opens each node with `O_APPEND`, reads driver
attributes, and closes it. Thus the source does not support a claim that unselected
device nodes are never opened. Failed metadata reads are caught and omitted;
that behavior alone does not validate restricted container configurations.

For Blackhole remote discovery, the branch at
`device/topology/topology_discovery.cpp:357` records an unselected remote ASIC
as an external connection and continues before remote device creation.
`TopologyDiscoveryBlackhole::create_remote_device` also returns null in this
revision. These observations concern this discovery path, not every TT runtime
operation or reset utility.

## Remaining execution requirements

- Verify the installed runtime matches the audited source chain and that its
  driver metadata-open behavior leaves other workloads unaffected.
- Resolve the pinned TT-XLA documentation's all-device-container requirement
  against the intended device cgroup configuration; do not infer support from
  the UMD filter alone.
- Establish enforceable exclusive board allocation across runners, containers,
  multi-card allocations and reset jobs. A one-runner queue or empty holder
  snapshot is not that allocation.
- Capture actual initialization isolation and stable board identity before
  authorizing the harness's physical mode. No visibility-proof JSON has been
  generated from this source inspection.

Sources: [PCI enumeration and metadata reads](https://github.com/tenstorrent/tt-umd/blob/8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404/device/pcie/pci_device.cpp),
[topology initialization](https://github.com/tenstorrent/tt-umd/blob/8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404/device/topology/topology_discovery.cpp),
[Blackhole remote discovery](https://github.com/tenstorrent/tt-umd/blob/8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404/device/topology/topology_discovery_blackhole.cpp).
