# Future Added Heat Otsu Design

## Goal

Generate the actual hot/cold threshold from the same future-20-second added-heat
quantity used to label completed EQ samples, while keeping histogram memory and
runtime bounded independently of object count.

## Histogram

- Transform each non-negative `future_window_added_heat` with `log1p`.
- Use exactly 10000 bins with score width `0.01`; clamp values above the final
  bin into that bin.
- Store aggregate counts in a heap-allocated fixed `uint64_t[10000]`.
- Compute Otsu with three contiguous passes and no per-update candidate vector.
- Report non-empty bin count and retained sample count. A retained entry is an
  evaluated I/O sample, not an object.

## Time Window

- Retain samples for 120 seconds in 120 one-second slots.
- Store each slot as a fixed `uint32_t[10000]`; use absolute monotonic-second
  tags, not timer callback counts.
- Place a sample by its evaluation deadline, not delayed worker execution time.
- Before insertion or threshold computation, advance to the current monotonic
  second. Clear every crossed slot; a missing second is an empty slot.
- If 120 or more seconds were skipped, clear the complete ring and aggregate.
- Do not insert a sample that is already 120 seconds old when processed.

## Label And Threshold Order

1. At the EQ deadline, compute `future_window_added_heat`.
2. Label the sample with the effective threshold that existed before this
   sample was observed.
3. Insert the completed sample into the timed histogram.
4. Recompute the Otsu candidate after every 100 inserted samples or after the
   existing one-second maximum interval.
5. Keep H0 threshold EMA gain at `0.10` per one-second reference interval and
   P0 prediction threshold fixed at `0.50`.

The added-heat threshold does not decay while idle. Heat decay remains part of
calculating the sample's future added heat, but an already completed sample is
a fixed measurement.

## Existing Object Heat State

The object-keyed order-statistics state remains available for the optional
heat-percentile feature. It no longer feeds Otsu. The old object-total-heat
Otsu with lower bound 10 remains a documented control experiment, not the
default implementation.

## Verification

- Probe fixed bin mapping, Otsu separation, one-second gaps, partial expiry,
  full expiry, late insertion and label-before-observe ordering.
- Verify reset clears aggregate and all time slots.
- Compile and run `hp_algorithm_probe` and the affected Ceph target.
- Rename exported retained-count statistics from object count to sample count
  in OSD and MGR output.
