/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <algorithm>
#include <optional>

#include "DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

llvm::SmallVector<unsigned, 4>
buildPrefixCount(const llvm::SmallVector<unsigned, 4> &syncs, unsigned idx) {
  llvm::SmallVector<unsigned, 4> prefix(idx + 1);
  unsigned acc = 0;
  for (unsigned i = 0, si = 0; i < idx; ++i) {
    prefix[i] = acc;
    if (si < syncs.size() && syncs[si] == i) {
      ++acc;
      ++si;
    }
  }
  prefix[idx] = acc;
  return prefix;
}

static llvm::DenseSet<Operation*> syncWallOps;

static bool isSyncWallOp(Operation *op) {
  if (syncWallOps.contains(op)) {
    return true;
  }
  if (isExternalSyncOp(op)) {
    syncWallOps.insert(op);
    return true;
  }
  
  // Interrupted when find inner sync wall ops, then will mark current op as sync wall op
  auto result = op->walk([&](Operation *inner) {
    if (isSyncWallOp(op)) {
      syncWallOps.insert(op);
      return WalkResult::interrupt();
    }
  });

  return result.wasInterrupted();
}

template <typename T>
std::optional<std::reference_wrapper<T>> getByCore(CoreType core, T &cube,
                                                   T &vector) {
  if (core == CoreType::CUBE_ONLY) {
    return std::cref(cube);
  }
  if (core == CoreType::VECTOR_ONLY) {
    return std::cref(vector);
  }
  return std::nullopt;
}
} // namespace

SyncWall::SyncWall(Block *block) {
  unsigned idx = 0;
  llvm::SmallDenseSet<unsigned, 4> seen;
  block->walk([&](Operation *op) {
    Operation *owner = getAncestorInBlock(op, block);
    if (owner == nullptr) {
      return;
    }
    if (!ordinal.contains(owner)) {
      ordinal[owner] = idx++;
    }
    if (isExternalSyncOp(op)) {
      unsigned pos = ordinal[owner];
      auto coreType = CVPipeline::getOpCoreType(op);
      if (seen.insert(pos).second) {
        if (auto syncPositions =
                getByCore(coreType, cubeSyncPositions, vectorSyncPositions)) {
          syncPositions->get().push_back(pos);
        }
      }

      // UNDETERMINED syncs barrier nothing (no matching wall), drop them.
    }
  });
  llvm::sort(cubeSyncPositions);
  llvm::sort(vectorSyncPositions);
  cubePrefixCount = buildPrefixCount(cubeSyncPositions, idx);
  vectorPrefixCount = buildPrefixCount(vectorSyncPositions, idx);
}

unsigned SyncWall::positionOf(Operation *op) const {
  auto it = ordinal.find(op);
  if (it != ordinal.end()) {
    return it->second;
  }
  return 0;
}

bool SyncWall::hasSyncBetween(Operation *a, Operation *b) const {
  auto aCore = CVPipeline::getOpCoreType(a);
  if (aCore != CVPipeline::getOpCoreType(b)) {
    return false;
  }

  if (auto syncs = getByCore(aCore, cubeSyncPositions, vectorSyncPositions)) {
    const auto &syncPos = syncs->get();
    unsigned lo = positionOf(a);
    unsigned hi = positionOf(b);
    if (lo > hi) {
      std::swap(lo, hi);
    }
    auto it = std::upper_bound(syncPos.begin(), syncPos.end(), lo);
    return it != syncPos.end() && *it < hi;
  }
  return false; // UNDETERMINED: no wall
}

unsigned SyncWall::segmentOf(Operation *op) const {
  unsigned pos = positionOf(op);
  auto core = CVPipeline::getOpCoreType(op);
  if (auto prefix = getByCore(core, cubePrefixCount, vectorPrefixCount)) {
    const auto &prefixCount = prefix->get();
    if (pos < prefixCount.size()) {
      return prefixCount[pos];
    }
  }
  return 0;
}

bool SyncWall::sameSegment(Operation *a, Operation *b) const {
  return CVPipeline::getOpCoreType(a) == CVPipeline::getOpCoreType(b) &&
         segmentOf(a) == segmentOf(b);
}
