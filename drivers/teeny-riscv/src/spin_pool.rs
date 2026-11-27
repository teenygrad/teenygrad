/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! A thread pool for back-to-back short jobs: workers spin between jobs, and sleep
//! only after a spell with none.
//!
//! A model step is thousands of kernel launches, many of them a few microseconds of
//! work on a tensor that fits in cache. Handing each to threads that sleep between
//! jobs costs an OS wake-up per thread per launch, tens of microseconds, which is
//! more than the work. Here a job is published by bumping an epoch counter that idle
//! workers are already watching, so a launch reaches them in well under a
//! microsecond while launches keep coming. A worker that has seen no job for
//! [`SpinPool::new`]'s `spin` sleeps on a condvar, so an idle process does not keep
//! its cores busy.
//!
//! The calling thread runs the job too, then waits for the workers to finish it, so
//! a pool of `n` threads has `n - 1` workers.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The job in flight: a type-erased `&(dyn Fn() + Sync)`, valid while it runs.
///
/// [`SpinPool::run`] does not return until every worker has finished with it, which
/// is what makes handing a borrowed closure to other threads sound.
#[derive(Clone, Copy)]
struct JobPtr(*const (dyn Fn() + Sync));

// Safety: the pointee is `Sync`, and `SpinPool::run` keeps it alive until every
// worker is done with it.
unsafe impl Send for JobPtr {}

struct Shared {
    /// Bumped once per job; a worker runs the job when it sees a new value.
    epoch: AtomicU64,
    /// The current job. Written before `epoch` is bumped and read after a worker
    /// sees the bump.
    job: Mutex<Option<JobPtr>>,
    /// Workers yet to finish the current job.
    pending: AtomicUsize,
    /// Workers asleep on `wake`, so a job knows whether to notify.
    sleepers: AtomicUsize,
    /// Guards the sleep/wake handshake.
    lock: Mutex<()>,
    wake: Condvar,
    /// How long a worker spins without a job before sleeping.
    spin: Duration,
    /// Set when the pool is dropped.
    shutdown: std::sync::atomic::AtomicBool,
}

/// See the module docs.
pub struct SpinPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// One job at a time: the epoch and the pending count describe a single job.
    running: Mutex<()>,
}

impl SpinPool {
    /// A pool of `threads` threads counting the caller, so `threads - 1` workers.
    /// `start` runs first on each worker, with its index.
    pub fn new(threads: usize, spin: Duration, start: impl Fn(usize) + Send + Sync + 'static) -> Self {
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: Mutex::new(None),
            pending: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            spin,
            shutdown: std::sync::atomic::AtomicBool::new(false),
        });
        let start = Arc::new(start);
        let workers = (0..threads.saturating_sub(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                let start = Arc::clone(&start);
                std::thread::Builder::new()
                    .name(format!("teeny-riscv-{i}"))
                    .spawn(move || {
                        start(i);
                        worker(&shared);
                    })
                    .expect("spawning a launcher thread")
            })
            .collect();
        Self {
            shared,
            workers,
            running: Mutex::new(()),
        }
    }

    /// Threads a job runs on, counting the caller.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `job` on every thread of the pool, the caller included, and returns once
    /// all of them have finished it.
    pub fn run(&self, job: &(dyn Fn() + Sync)) {
        let _one_job = self.running.lock().unwrap_or_else(PoisonError::into_inner);
        let shared = &self.shared;
        if self.workers.is_empty() {
            job();
            return;
        }
        // Safety: erases the borrow's lifetime. The pointer is only dereferenced by
        // workers that count themselves out through `pending`, and this function
        // waits for `pending` to reach zero before `job`'s borrow ends.
        let ptr: *const (dyn Fn() + Sync + 'static) =
            unsafe { std::mem::transmute(job as *const (dyn Fn() + Sync)) };
        *shared.job.lock().unwrap_or_else(PoisonError::into_inner) = Some(JobPtr(ptr));
        shared.pending.store(self.workers.len(), Ordering::SeqCst);
        shared.epoch.fetch_add(1, Ordering::SeqCst);
        if shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = shared.lock.lock().unwrap_or_else(PoisonError::into_inner);
            shared.wake.notify_all();
        }

        job();

        while shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    /// [`Self::run`] on the workers alone: the caller publishes `job` and waits.
    ///
    /// For a pool whose workers sit on other cores than the caller -- the K3's A100
    /// cores, whose matrix instructions the X100 core launching the job cannot run.
    pub fn run_on_workers(&self, job: &(dyn Fn() + Sync)) {
        assert!(!self.workers.is_empty(), "a pool with no workers cannot run a job on them");
        let _one_job = self.running.lock().unwrap_or_else(PoisonError::into_inner);
        let shared = &self.shared;
        let ptr: *const (dyn Fn() + Sync + 'static) =
            unsafe { std::mem::transmute(job as *const (dyn Fn() + Sync)) };
        *shared.job.lock().unwrap_or_else(PoisonError::into_inner) = Some(JobPtr(ptr));
        shared.pending.store(self.workers.len(), Ordering::SeqCst);
        shared.epoch.fetch_add(1, Ordering::SeqCst);
        if shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = shared.lock.lock().unwrap_or_else(PoisonError::into_inner);
            shared.wake.notify_all();
        }
        while shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    /// Worker threads, not counting the caller.
    pub fn workers(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.epoch.fetch_add(1, Ordering::SeqCst);
        {
            let _guard = self.shared.lock.lock().unwrap_or_else(PoisonError::into_inner);
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(shared: &Shared) {
    // The epoch the pool was created with, not whatever it reads now: a job published
    // before this thread got here would otherwise look already seen, and its caller
    // would wait for this worker forever.
    let mut seen = 0;
    loop {
        // Spin for a new epoch, then sleep until one arrives.
        let spun_from = Instant::now();
        let mut polls = 0u32;
        loop {
            if shared.epoch.load(Ordering::Acquire) != seen {
                break;
            }
            std::hint::spin_loop();
            polls = polls.wrapping_add(1);
            // Reading the clock every poll would cost more than the poll.
            if polls % 1024 == 0 && spun_from.elapsed() >= shared.spin {
                let guard = shared.lock.lock().unwrap_or_else(PoisonError::into_inner);
                shared.sleepers.fetch_add(1, Ordering::SeqCst);
                // The epoch is re-read under the lock, after counting this worker as
                // a sleeper: a job bumped before the count is seen here, and one
                // bumped after it sees the count and notifies under the same lock.
                let guard = shared
                    .wake
                    .wait_while(guard, |_| shared.epoch.load(Ordering::SeqCst) == seen)
                    .unwrap_or_else(PoisonError::into_inner);
                shared.sleepers.fetch_sub(1, Ordering::SeqCst);
                drop(guard);
                break;
            }
        }
        seen = shared.epoch.load(Ordering::Acquire);
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let job = *shared.job.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(JobPtr(ptr)) = job {
            // Safety: see `SpinPool::run`.
            unsafe { (*ptr)() };
        }
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_thread_runs_every_job() {
        let pool = SpinPool::new(4, Duration::from_micros(50), |_| {});
        let count = AtomicU64::new(0);
        for _ in 0..1000 {
            pool.run(&|| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(count.load(Ordering::Relaxed), 4 * 1000);
    }

    #[test]
    fn run_on_workers_runs_once_per_worker_and_not_on_the_caller() {
        let pool = SpinPool::new(4, Duration::from_micros(50), |_| {});
        let caller = std::thread::current().id();
        let count = AtomicU64::new(0);
        let on_caller = AtomicU64::new(0);
        for _ in 0..500 {
            pool.run_on_workers(&|| {
                count.fetch_add(1, Ordering::Relaxed);
                if std::thread::current().id() == caller {
                    on_caller.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        assert_eq!(count.load(Ordering::Relaxed), 3 * 500);
        assert_eq!(on_caller.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn workers_wake_after_sleeping() {
        // A spin budget far shorter than the pause, so the workers are asleep when
        // the second job arrives.
        let pool = SpinPool::new(3, Duration::from_micros(1), |_| {});
        let count = AtomicU64::new(0);
        pool.run(&|| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        std::thread::sleep(Duration::from_millis(20));
        pool.run(&|| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn a_job_sees_writes_made_before_it_and_the_caller_sees_its_writes() {
        let pool = SpinPool::new(4, Duration::from_micros(50), |_| {});
        let slots: Vec<AtomicU64> = (0..4).map(|_| AtomicU64::new(0)).collect();
        let next = AtomicUsize::new(0);
        for round in 1..=200u64 {
            next.store(0, Ordering::Relaxed);
            pool.run(&|| {
                let i = next.fetch_add(1, Ordering::Relaxed);
                slots[i].store(round, Ordering::Relaxed);
            });
            assert!(slots.iter().all(|s| s.load(Ordering::Relaxed) == round));
        }
    }

    #[test]
    fn a_one_thread_pool_runs_on_the_caller() {
        let pool = SpinPool::new(1, Duration::from_micros(50), |_| {});
        let count = AtomicU64::new(0);
        pool.run(&|| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
