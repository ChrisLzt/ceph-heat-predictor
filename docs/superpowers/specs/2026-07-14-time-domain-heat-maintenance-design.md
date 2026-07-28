# Time-Domain Heat And Maintenance Design

## Goal

Make object heat, recency features, Otsu threshold tracking, access-window
cleanup, and prediction snapshot freshness independent of per-OSD I/O rate,
while preserving the existing 20-second label task and synchronous prediction
contract.

## Time Model

- Use `steady_clock` nanoseconds for all elapsed-time calculations.
- Heat retains `1/10` of its value after 20 seconds without access.
- The short and long access windows remain strict 5-second and 20-second
  windows and exclude the current I/O from captured features.
- Replace I/O distance since the previous access with elapsed monotonic time;
  expose `log2(1 + elapsed_seconds)` to the model.
- Keep capacities, training batches, counters, and evaluation admission limits
  count-based.

## Event-Driven Maintenance

Reuse the existing expiry worker. Its next wakeup is the minimum of the next
EQ label deadline, short-window event expiry, and long-window event expiry.
The worker removes every due access event under `eq_mutex`, even when no new
I/O arrives. Foreground feature preparation retains the same cleanup guard so
an I/O at a deadline observes strict window contents.

## Heat And Otsu

Store each object's last-access monotonic timestamp and decay directly from
elapsed nanoseconds. Otsu score conversion uses the same time-domain decay
factor. The effective threshold stores a heat value and its monotonic timestamp,
so reads can decay it to prediction, labeling, or reporting time without
mutating state. Otsu updates interpolate the effective threshold in score
space and then store the resulting heat at the update timestamp.

The fixed H0 EMA remains `0.10` per one-second reference interval. For elapsed
time `dt`, use `1 - pow(1 - 0.10, dt / 1s)`. Recompute Otsu after 100 object
observations or after one second with a new observation, whichever comes
first. H1 remains disabled and documented as a later experiment.

## Prediction Snapshot Freshness

Publish a cloned prediction snapshot after 500 trained samples or, when at
least one new sample was trained, after one second since the previous publish.
Training remains asynchronous and batching remains count-based.

## O(1) Prediction Completion

`PredictionReservation` carries a stable `std::list` iterator instead of a raw
node pointer. A node cannot be erased until both prediction and label complete,
and `reset_mutex` protects its lifetime across synchronous prediction. Both
normal and late completion therefore erase directly in O(1), without scanning
the list. No timeout or independent cap is added for `awaiting_prediction`.

## Verification

- Probe exact time decay, time recency, idle access cleanup, time-normalized
  Otsu EMA, dual-trigger snapshot publication, and O(1) late completion.
- Run normal, ASan/UBSan, TSan, and performance probes because core containers,
  timing, and concurrency scheduling change.
- Complete the L2 build/install/restart/reset/status procedure and wait for all
  PGs to return to `active+clean`.
