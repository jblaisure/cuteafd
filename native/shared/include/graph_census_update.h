#pragma once

#include "cuteafd_graph_diagnostics.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuteafd {
// Process-local, ordered function identities and indexed edges. Equality also
// checks the full signature: a hash collision must not select an update donor.
struct CensusTopology {
  std::vector<uint64_t> signature;
  cudaError_t status = cudaSuccess;
  uint64_t hash = 14695981039346656037ULL;
};

inline CensusTopology census_topology(cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& nodes, std::vector<uint64_t> signature,
    cudaError_t status) {
  CensusTopology out{std::move(signature), status};
  if (status != cudaSuccess) return out;
  std::unordered_map<cudaGraphNode_t, size_t> index;
  for (size_t n = 0; n < nodes.size(); ++n) index[nodes[n]] = n;
  size_t capacity = std::max(size_t(1), nodes.size());
  if (capacity > std::numeric_limits<size_t>::max() / 64) {
    out.status = cudaErrorInvalidValue; return out;
  }
  const size_t cap = capacity * 64;
  std::vector<cudaGraphNode_t> from, to;
  std::vector<cudaGraphEdgeData> data;
  size_t edges = 0;
  for (;;) {
    from.resize(capacity); to.resize(capacity); data.resize(capacity);
    edges = capacity;
    out.status = diagnostic_cuda(cudaGraphGetEdges(graph, from.data(), to.data(), data.data(), &edges));
    if (out.status != cudaSuccess) return out;
    if (edges < capacity) break;
    if (capacity == cap) { out.status = cudaErrorNotSupported; return out; }
    capacity = std::min(cap, capacity * 2);
  }
  std::vector<std::array<uint64_t, 5>> ordered;
  for (size_t e = 0; e < edges; ++e) {
    if (!index.count(from[e]) || !index.count(to[e])) {
      out.status = cudaErrorInvalidValue; return out;
    }
    ordered.push_back({index[from[e]], index[to[e]], data[e].from_port, data[e].to_port, data[e].type});
  }
  std::sort(ordered.begin(), ordered.end());
  out.signature.push_back(edges);
  for (const auto& edge : ordered) out.signature.insert(out.signature.end(), edge.begin(), edge.end());
  for (uint64_t word : out.signature) for (int byte = 0; byte < 8; ++byte) {
    out.hash ^= (word >> (byte * 8)) & 255;
    out.hash *= 1099511628211ULL;
  }
  return out;
}

struct CensusUpdateMeasurement {
  uint64_t donor_id = 0;
  int donor_status = cudaSuccess;
  int update_status = cudaSuccess;
  int update_result = -1;
  int scratch_destroy_status = cudaSuccess;
  int clone_status = cudaSuccess;
  int memory_status = cudaSuccess;
  void* error_node = nullptr;
  void* error_from_node = nullptr;
  double update_us = 0;
  int64_t clone_bytes = 0;
  int64_t scratch_peak_bytes = 0;
  int64_t scratch_retained_bytes = 0;
  int update_memory_status = cudaErrorNotSupported;
  int64_t update_bytes = 0;
};

class CensusUpdateProbe {
  struct Source { cudaGraph_t graph; CensusTopology topology; int device; uint64_t id; };
  std::mutex mutex_;
  std::unordered_map<cudaGraphExec_t, Source> sources_;

 public:
  // Disabled means no CUDA query, clone, scratch executable or retained source.
  CensusUpdateMeasurement capture(bool enabled, cudaGraph_t graph, cudaGraphExec_t exec,
      const CensusTopology& topology, int device, uint64_t id) {
    CensusUpdateMeasurement out;
    if (!enabled || topology.status != cudaSuccess) return out;
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& entry : sources_) {
      const auto& source = entry.second;
      if (source.device != device || source.topology.hash != topology.hash ||
          source.topology.signature != topology.signature) continue;
      out.donor_id = source.id;
      size_t before = 0, peak = 0, after = 0, total = 0;
      out.memory_status = diagnostic_cuda(cudaMemGetInfo(&before, &total));
      cudaGraphExec_t scratch = nullptr;
      out.donor_status = diagnostic_cuda(cudaGraphInstantiate(&scratch, source.graph, 0));
      if (out.donor_status == cudaSuccess && scratch) {
        const auto peak_status = diagnostic_cuda(cudaMemGetInfo(&peak, &total));
        // The sample precedes the timer; no memory query enters update_us.
        size_t update_after = 0;
        cudaGraphExecUpdateResultInfo result{};
        const auto start = std::chrono::steady_clock::now();
        const auto status = cudaGraphExecUpdate(scratch, graph, &result);
        const auto stop = std::chrono::steady_clock::now();
        out.update_us = std::chrono::duration<double, std::micro>(stop - start).count();
        out.update_status = diagnostic_cuda(status);
        const auto update_after_status = diagnostic_cuda(cudaMemGetInfo(&update_after, &total));
        out.update_memory_status = peak_status != cudaSuccess ? peak_status : update_after_status;
        if (out.update_memory_status == cudaSuccess)
          out.update_bytes = static_cast<int64_t>(peak) - static_cast<int64_t>(update_after);
        out.update_result = static_cast<int>(result.result);
        out.error_node = reinterpret_cast<void*>(result.errorNode);
        out.error_from_node = reinterpret_cast<void*>(result.errorFromNode);
        if (out.memory_status == cudaSuccess) out.memory_status = peak_status;
      }
      if (scratch) out.scratch_destroy_status = diagnostic_cuda(cudaGraphExecDestroy(scratch));
      const auto after_status = diagnostic_cuda(cudaMemGetInfo(&after, &total));
      if (out.memory_status == cudaSuccess) out.memory_status = after_status;
      if (out.memory_status == cudaSuccess && out.donor_status == cudaSuccess) {
        out.scratch_peak_bytes = static_cast<int64_t>(before) - static_cast<int64_t>(peak);
        out.scratch_retained_bytes = static_cast<int64_t>(before) - static_cast<int64_t>(after);
      }
      break;
    }
    size_t before = 0, after = 0, total = 0;
    const auto before_status = diagnostic_cuda(cudaMemGetInfo(&before, &total));
    cudaGraph_t clone = nullptr;
    out.clone_status = diagnostic_cuda(cudaGraphClone(&clone, graph));
    const auto after_status = diagnostic_cuda(cudaMemGetInfo(&after, &total));
    if (out.memory_status == cudaSuccess) out.memory_status = before_status;
    if (out.memory_status == cudaSuccess) out.memory_status = after_status;
    if (before_status == cudaSuccess && after_status == cudaSuccess)
      out.clone_bytes = static_cast<int64_t>(before) - static_cast<int64_t>(after);
    if (out.clone_status == cudaSuccess && clone) {
      sources_.emplace(exec, Source{clone, topology, device, id});
    } else if (clone) { diagnostic_cuda(cudaGraphDestroy(clone)); }
    return out;
  }

  cudaError_t retire(cudaGraphExec_t exec) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = sources_.find(exec);
    if (found == sources_.end()) return cudaSuccess;
    const auto status = diagnostic_cuda(cudaGraphDestroy(found->second.graph));
    sources_.erase(found);
    return status;
  }
};
}  // namespace cuteafd
