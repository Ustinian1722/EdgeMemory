#pragma once

#include "hexagonflow/interference.hpp"

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace hexagonflow {

class Scheduler;

// Reads the versioned, delimiter-safe TSV emitted by the private hardware
// observation exporter. The caller is responsible for verifying its run
// manifest before opening the profile.
[[nodiscard]] std::vector<InterferenceObservation>
read_interference_observations_tsv(std::istream& input);

// Parses the entire profile before updating the scheduler, so malformed later
// rows cannot leave a partially ingested calibration set.
std::size_t load_interference_observations_tsv(std::istream& input,
                                              Scheduler& scheduler);

}  // namespace hexagonflow
