#pragma once

#include "sovereign/types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sovereign {
namespace server {

/**
 * Job lifecycle.
 *
 * These are deliberately distinct from SolverStatus. A job is about the
 * *request* (has it been dequeued, is it still running, was it cancelled); a
 * SolverStatus is about the *mathematics* (is the answer proven optimal). They
 * compose: a job can be SOLVING and later report INFEASIBLE, and conflating
 * them is how a cancelled run ends up looking like a proven answer.
 */
enum class JobState {
  Queued,
  Solving,
  Completed,  // ran to a terminal SolverStatus
  Failed,     // could not run at all (bad model file, internal error)
  Cancelled,  // cancelled before completion
};

std::string to_string(JobState state);

/** One progress event, as delivered over SSE. */
struct JobProgress {
  JobState state = JobState::Queued;
  double runtime_seconds = 0.0;

  // LP / MILP counters. Mirrors what the engine reports, so the UI never has to
  // invent a number the solver did not produce.
  std::int64_t lp_iterations = 0;
  std::int64_t nodes = 0;
  bool has_incumbent = false;
  double incumbent = 0.0;
  bool has_bound = false;
  double best_bound = 0.0;
  double mip_gap = 0.0;

  // Presolve reductions, once known.
  bool has_presolve = false;
  std::int64_t rows_removed = 0;
  std::int64_t cols_removed = 0;
  std::int64_t vars_fixed = 0;
  std::int64_t bounds_tightened = 0;

  std::string device = "cpu";
  std::int64_t threads = 1;
  std::int64_t peak_memory_bytes = 0;

  /**
   * The solution vector, as JSON, once a solve finishes.
   *
   * Emitted only on success. It is the actual point the engine computed, not a
   * reconstruction, so the verifier can be re-run against it independently.
   */
  std::string solution_json;

  // Final fields, only meaningful once Completed.
  std::string solver_status;
  std::string message;
  bool has_objective = false;
  double objective = 0.0;
  std::string verification_json;
};

struct SolveOptions {
  /** Path to a model on disk, resolved inside the model root. */
  std::string model_path;

  /**
   * Inline model JSON, used when the user pastes or uploads a problem instead
   * of choosing a bundled one. Takes precedence over model_path when non-empty.
   */
  std::string model_json;
  std::string model_format;  // json (default) or mps

  std::string algorithm;  // "", "simplex", "ipm", "auto"
  int threads = 1;
  double time_limit_seconds = 0.0;  // 0 = none
  double mip_gap = 0.0;
  bool presolve = true;
  bool device = false;  // false = CPU. GPU is not implemented; see ROADMAP M11.
};

class JobManager;
struct SolveOptions;

/** Handle to a running job, used to observe and cancel it. */
class Job {
 public:
  Job(std::string id, JobManager* owner) : id_(std::move(id)), owner_(owner) {}

  const std::string& id() const { return id_; }

  /** Latest progress snapshot. Safe from any thread. */
  JobProgress snapshot() const;

  /** Request cancellation. Cooperative: the solver checks a flag between pivots. */
  void request_cancel();

  /** True once a worker has finished with this job. */
  bool finished() const;

  /**
   * Block until the job reaches a terminal state. Used by tests and by the
   * one-shot CLI path; the HTTP layer must never call this.
   */
  void wait() const;

 private:
  friend class JobManager;
  std::string id_;
  JobManager* owner_;

  // Set once, by JobManager::submit, before the job is queued. Read by exactly
  // one worker afterwards, so no extra synchronisation is needed beyond the
  // queue push itself.
  SolveOptions options_;

  mutable std::mutex mutex_;
  JobProgress progress_;
  std::atomic<bool> cancel_requested_{false};
  // `mutable` so wait() can be const: waiting does not change the job's logical
  // state, it only blocks the caller until a worker finishes with it.
  mutable std::condition_variable_any finished_cv_;
  bool done_ = false;
};

using JobProgressCallback = std::function<void(const JobProgress&)>;

/**
 * Asynchronous solve queue.
 *
 * The reason this exists: an industrial MILP can run for hours. If the HTTP
 * request blocked on it, the browser would sit on a pending fetch with no
 * progress, no cancellation, and one exhausted worker. So the route handler
 * enqueues and returns a job id immediately, and progress is streamed.
 *
 * Threading: one worker per hardware thread by default. `sovereign_core` is
 * internally single-threaded per solve, so the parallelism here is across jobs,
 * not within one. That is deliberate and it keeps `--threads 1` deterministic.
 */
class JobManager {
 public:
  explicit JobManager(unsigned worker_threads = 0);
  ~JobManager();

  JobManager(const JobManager&) = delete;
  JobManager& operator=(const JobManager&) = delete;

  /** Queue a solve. Returns immediately. */
  std::shared_ptr<Job> submit(const SolveOptions& options,
                               JobProgressCallback on_progress = nullptr);

  std::shared_ptr<Job> find(const std::string& id) const;

  /** Jobs in submission order, for a dashboard listing. */
  std::vector<JobProgress> list() const;

  /** Block until every queued and running job has finished. Test/shutdown aid. */
  void drain();

  /** Stop accepting work; running jobs finish. */
  void shutdown();

  unsigned worker_count() const { return static_cast<unsigned>(workers_.size()); }

  /**
   * Execute a solve on the calling thread. Exposed so the CLI and the tests can
   * use the exact same path the server uses, rather than a second implementation
   * that can drift.
   */
  static JobProgress run_solve(const SolveOptions& options,
                               const std::atomic<bool>& cancel_flag,
                               const JobProgressCallback& on_progress);

 private:
  friend class Job;

  void worker_loop();
  void publish(Job* job, const JobProgress& progress);

  mutable std::mutex mutex_;
  std::condition_variable_any work_cv_;
  std::condition_variable_any finished_cv_;

  std::deque<Job*> queue_;
  std::vector<std::thread> workers_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::map<std::string, JobProgressCallback> callbacks_;

  std::uint64_t next_id_ = 1;
  bool stopping_ = false;
};

/** Peak resident set size of this process, in bytes. 0 if unavailable. */
std::int64_t process_peak_memory_bytes();

/** Hardware concurrency, clamped to a sane range. */
unsigned resolve_worker_threads(int requested);

}  // namespace server
}  // namespace sovereign
