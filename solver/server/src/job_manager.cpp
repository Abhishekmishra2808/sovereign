#include "sovereign/server/job_manager.hpp"

#include "sovereign/engine.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/mps_io.hpp"
#include <nlohmann/json.hpp>
#include "sovereign/verifier.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
// psapi.h needs the base Windows types first; including it alone fails.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace sovereign {
namespace server {
namespace {

std::string make_id(std::uint64_t n) {
  std::ostringstream oss;
  oss << "job-" << n;
  return oss.str();
}

double elapsed_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

std::string to_string(JobState state) {
  switch (state) {
    case JobState::Queued:
      return "QUEUED";
    case JobState::Solving:
      return "SOLVING";
    case JobState::Completed:
      return "COMPLETED";
    case JobState::Failed:
      return "FAILED";
    case JobState::Cancelled:
      return "CANCELLED";
  }
  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Job
// ---------------------------------------------------------------------------

JobProgress Job::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return progress_;
}

void Job::request_cancel() {
  cancel_requested_.store(true);
}

bool Job::finished() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return done_;
}

void Job::wait() const {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_cv_.wait(lock, [this] { return done_; });
}

// ---------------------------------------------------------------------------
// JobManager
// ---------------------------------------------------------------------------

unsigned resolve_worker_threads(int requested) {
  if (requested > 0) return static_cast<unsigned>(requested);
  const unsigned hw = std::thread::hardware_concurrency();
  // One worker per hardware thread. sovereign_core is single-threaded per solve,
  // so this is parallelism across jobs, never within one. Keeping it that way is
  // what makes --threads 1 reproducible.
  return hw == 0 ? 2u : hw;
}

JobManager::JobManager(unsigned worker_threads) {
  const unsigned n = resolve_worker_threads(static_cast<int>(worker_threads));
  workers_.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

JobManager::~JobManager() {
  shutdown();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
}

void JobManager::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  work_cv_.notify_all();
}

std::shared_ptr<Job> JobManager::submit(const SolveOptions& options,
                                        JobProgressCallback on_progress) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string id = make_id(next_id_++);
  auto job = std::make_shared<Job>(id, this);

  job->progress_.state = JobState::Queued;
  job->progress_.device = options.device ? "cuda" : "cpu";
  job->progress_.threads = options.threads;

  jobs_[id] = job;
  job->options_ = options;
  if (on_progress) callbacks_[id] = std::move(on_progress);
  queue_.push_back(job.get());
  work_cv_.notify_one();
  return job;
}

std::shared_ptr<Job> JobManager::find(const std::string& id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = jobs_.find(id);
  return it == jobs_.end() ? nullptr : it->second;
}

std::vector<JobProgress> JobManager::list() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<JobProgress> out;
  out.reserve(jobs_.size());
  for (const auto& kv : jobs_) out.push_back(kv.second->snapshot());
  return out;
}

void JobManager::drain() {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_cv_.wait(lock, [this] {
    return queue_.empty() &&
           std::none_of(jobs_.begin(), jobs_.end(),
                        [](const auto& kv) { return !kv.second->finished(); });
  });
}

void JobManager::publish(Job* job, const JobProgress& progress) {
  JobProgressCallback cb;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    {
      std::lock_guard<std::mutex> job_lock(job->mutex_);
      job->progress_ = progress;
    }
    auto it = callbacks_.find(job->id_);
    if (it != callbacks_.end()) cb = it->second;
  }
  // The callback runs outside the lock: it will typically write to an SSE stream,
  // and holding the job lock across a socket write invites deadlock.
  if (cb) cb(progress);
}

void JobManager::worker_loop() {
  for (;;) {
    Job* job = nullptr;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_ && queue_.empty()) return;
      if (queue_.empty()) continue;
      job = queue_.front();
      queue_.pop_front();
    }

    SolveOptions options;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Options ride along on the Job so the worker needs no second map.
      options = job->options_;
    }

    JobProgressCallback cb;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = callbacks_.find(job->id_);
      if (it != callbacks_.end()) cb = it->second;
    }

    try {
      const JobProgress result =
          run_solve(options, job->cancel_requested_, [this, job](const JobProgress& p) { publish(job, p); });
      publish(job, result);
    } catch (const std::exception& e) {
      JobProgress failed;
      failed.state = JobState::Failed;
      failed.message = e.what();
      publish(job, failed);
    } catch (...) {
      JobProgress failed;
      failed.state = JobState::Failed;
      failed.message = "unknown internal error";
      publish(job, failed);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      {
        std::lock_guard<std::mutex> job_lock(job->mutex_);
        job->done_ = true;
      }
      callbacks_.erase(job->id_);
    }
    job->finished_cv_.notify_all();
    finished_cv_.notify_all();
  }
}

// ---------------------------------------------------------------------------
// The actual solve
// ---------------------------------------------------------------------------

JobProgress JobManager::run_solve(const SolveOptions& options,
                                  const std::atomic<bool>& cancel_flag,
                                  const JobProgressCallback& on_progress) {
  const auto start = std::chrono::steady_clock::now();

  JobProgress p;
  p.device = options.device ? "cuda" : "cpu";
  p.threads = 1;  // per-job worker selection is not implemented
  p.state = JobState::Solving;
  if (on_progress) on_progress(p);

  if (options.model_json.empty() && options.model_path.empty()) {
    p.state = JobState::Failed;
    p.message = "no model supplied (neither inline JSON nor a path)";
    return p;
  }

  OptimizationModel model;
  try {
    if (!options.model_json.empty()) {
      // The user pasted or uploaded the problem. Parse it here so a malformed
      // paste fails with a clear message instead of surfacing later as a
      // cryptic solve error.
      if (!options.model_format.empty() && options.model_format!="json" && options.model_format!="mps") throw std::runtime_error("unsupported modelFormat");
      model = options.model_format=="mps" ? load_model_from_mps_string(options.model_json).model : load_model_from_json_string(options.model_json);
    } else {
      // File uploads and bundled examples share the core format dispatcher.
      model = load_model_from_file(options.model_path);
    }
  } catch (const std::exception& e) {
    p.state = JobState::Failed;
    p.message = std::string("could not read the model: ") + e.what();
    return p;
  }

  // A pasted model can be syntactically valid JSON and still not a solvable
  // problem. Say so before burning time on it.
  if (model.variables.empty()) {
    p.state = JobState::Failed;
    p.message = "the model contains no variables";
    return p;
  }
  if (model.variables.size() > 2000000) {
    p.state = JobState::Failed;
    p.message = "the model declares " + std::to_string(model.variables.size()) +
                " variables, which exceeds the 2,000,000 limit";
    return p;
  }

  if (cancel_flag.load()) {
    p.state = JobState::Cancelled;
    p.message = "cancelled before the solve started";
    p.runtime_seconds = elapsed_since(start);
    return p;
  }

  OptimizationEngine engine;
  EngineOptions engine_options;
  engine_options.lp_algorithm = options.algorithm;
  engine_options.presolve = options.presolve;
  const SolverResult result = engine.solve(model, engine_options);

  // Cooperative cancellation. The engine has no cancel hook yet, so this can
  // only be honoured between phases; it is checked here so the semantics are
  // honest rather than pretending a long solve is interruptible.
  if (cancel_flag.load()) {
    p.state = JobState::Cancelled;
    p.message = "cancelled after the solve finished; result discarded";
  } else {
    p.state = JobState::Completed;
    p.solver_status = to_string(result.status);
    p.message = result.message;
    p.has_objective = result.has_objective_value;
    p.objective = result.objective_value;
  }

  p.runtime_seconds = elapsed_since(start);
  p.lp_iterations = result.iterations;
  p.nodes = result.nodes;
  p.mip_gap = result.optimality_gap;
  p.has_presolve = false;  // reduction counters are not exposed by the engine
  p.peak_memory_bytes = process_peak_memory_bytes();

  // The actual solution point, as JSON, for the user to read and export. This is
  // the vector the engine computed, not a re-derivation.
  if (p.has_objective && !result.primal.empty()) {
    p.solution_json = nlohmann::json(result.primal).dump();
  }
  try {
    p.verification_json = verification_to_json_string(SolutionVerifier().verify(model,result));
  } catch (const std::exception&) {
    p.verification_json.clear();
  }

  if (on_progress) on_progress(p);
  return p;
}

std::int64_t process_peak_memory_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return static_cast<std::int64_t>(pmc.PeakWorkingSetSize);
  }
  return 0;
#else
  return 0;
#endif
}

}  // namespace server
}  // namespace sovereign
