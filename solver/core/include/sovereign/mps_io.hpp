#pragma once

#include "sovereign/types.hpp"

#include <string>

namespace sovereign {

/** Fixed/free linear MPS reader. Supports objective selection, ranges, integer
 * markers, and UP/LO/FX/FR/MI/PL/BV/LI/UI bounds. First RHS/range/bound set wins.
 * SC/SI, quadratic sections and compressed files are rejected explicitly.
 * Numeric values must be finite. Resource caps include expanded range rows.
 */

struct MpsParseOptions {
  /**
   * True for classic fixed-column MPS, false for free/whitespace-delimited.
   * Real files are a mix, so this defaults to fixed and the reader falls back
   * to free-format field splitting when a fixed-column read comes up short.
   */
  bool fixed_format = true;

  /** Cap on accepted problem size, as a guard against a malformed or hostile file. */
  std::size_t max_variables = 5'000'000;
  std::size_t max_constraints = 5'000'000;
  std::size_t max_nonzeros = 200'000'000;
};

struct MpsParseResult {
  OptimizationModel model;
  /** Row, column and nonzero counts, for reporting. */
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t nonzeros = 0;
  /** Name of the objective row, for diagnostics. */
  std::string objective_name;
  /** Non-fatal notes: ignored N-rows, default bounds applied, parser diagnostics. */
  std::vector<std::string> notes;
};

/**
 * Parse an MPS file.
 * @throws std::runtime_error with a specific message on malformed input.
 */
MpsParseResult load_model_from_mps_file(const std::string& path,
                                       const MpsParseOptions& options = {});

/** Parse MPS from memory. Same contract as the file variant. */
MpsParseResult load_model_from_mps_string(const std::string& text,
                                          const MpsParseOptions& options = {});

/**
 * Dispatch on file extension.
 *
 * The CLI and the local server both go through this so that a `.mps` file is
 * read by the right parser and a `.json` file is unchanged. Previously feeding
 * an MPS path to the JSON reader produced a raw nlohmann parse exception, which
 * is a terrible error message for a user who simply picked the wrong file.
 */
OptimizationModel load_model_from_file(const std::string& path);

}  // namespace sovereign
