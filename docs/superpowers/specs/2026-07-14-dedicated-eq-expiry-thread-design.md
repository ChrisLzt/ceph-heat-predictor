# Dedicated EQ Expiry Thread Design

## Goal

Evaluate admitted I/O samples after the existing 20-second monotonic window
without coupling expiry scheduling to model training, while preventing an I/O
that linearizes after a deadline from contaminating the expired sample's
future-heat label.

## Architecture

Keep one FIFO `pending_queue`. Add a dedicated expiry worker that sleeps until
the current head deadline with `condition_variable::wait_until()`. Keep a
constant-time foreground expiry guard in the same `eq_mutex` critical section
as current-I/O feature preparation: whichever thread acquires `eq_mutex` first
drains every ready, due head before the current I/O updates `heat_map`.

The training worker consumes only `train_queue`; it no longer polls EQ every
50 ms. Random-forest snapshot prediction and model training remain outside
`eq_mutex`.

## Synchronization

- Foreground and expiry processing use the lock order
  `reset_mutex(shared) -> eq_mutex`.
- Reset uses `reset_mutex(unique) -> eq_mutex`.
- The expiry worker never holds `reset_mutex` or `eq_mutex` while sleeping.
- A separate wait mutex plus an atomic wake sequence prevents lost wakeups
  without reversing the reset/EQ lock order.
- The worker is notified when an empty queue gains its first item, when the
  current head prediction becomes ready, and on reset, enable/disable, or
  shutdown.

## Foreground Flow

1. Acquire `reset_mutex` shared.
2. Acquire `eq_mutex` once.
3. Capture monotonic time and drain all ready heads whose deadline passed.
4. Increment the I/O index, update heat/features, capture the prediction
   snapshot and threshold, and reserve the new pending slot using the same
   timestamp.
5. Release `eq_mutex` and enqueue any expired training samples.
6. Predict from the immutable snapshot outside `eq_mutex`.
7. Reacquire `eq_mutex` briefly to mark the slot ready; notify the expiry
   worker only if queue scheduling may have changed.

## Expiry Worker Flow

1. Inspect the current queue head under `reset_mutex(shared) -> eq_mutex`.
2. If empty, wait for a wake-sequence change.
3. If not due, wait until its exact deadline or a wake-sequence change.
4. If due but not ready, wait for the head-ready notification.
5. If due and ready, drain all ready due heads, record their evaluation
   results, release `eq_mutex`, and enqueue training samples.
6. Recompute the next state from the current queue; never retain an EQ pointer
   across a wait.

## Lifecycle

`ensure_started()` starts both background workers. Reset wakes the expiry
worker after replacing EQ state. Destruction stops and notifies both workers,
then joins both threads. Disabled state leaves the expiry worker asleep.

## Verification

- A unit/probe test demonstrates the foreground guard expires a due item
  before the current access updates its label state.
- A scheduler test demonstrates empty, deadline, due-not-ready, and reset
  wake-state transitions without sleeping for 20 seconds.
- Existing time-window, bounded-admission, concurrency, reset, and snapshot
  tests remain green.
- TSan is attempted because synchronization changes; an unsupported runtime is
  reported separately from code failures.
- Ceph integration verifies idle expiry, reset/disable wakeup, `active+clean`,
  and unchanged conservation counters.
