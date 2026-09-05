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
 * all copies or substantial portions of the Software![](substantial portions of the Software).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT![](BUT NOT) LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN![](OTHER DEALINGS IN)
 * THE SOFTWARE.
 */
#include <algorithm>

#include "DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

// prefixCount[i] = #syncPoints at positions < i, for i in [0, idx]. O(idx).
llvm::SmallVector<unsigned, 4>
buildPrefixCount(const llvm::SmallVector<SyncPoint, 4> &syncs, unsigned idx) {
  llvm::SmallVector<unsigned, 4> prefix(idx + 1);
  unsigned acc = 0;
  for (unsigned i = 0, si = 0; i < idx; ++i) {
    prefix[i] = acc;
    if (si < syncs.size() && syncs[si].position == i) {
      ++acc;
      ++si;
    }
  }
  prefix[idx] = acc;
  return prefix;
}

} // namespace

SyncWall::SyncWall(Block *block) {
  // Phase 1: assign source-order ordinals to block-level ops (PreOrder).
  unsigned idx = 0;
  block->walk<WalkOrder::PreOrder>([&](Operation *op) {
    Operation *owner = getAncestorInBlock(op, block);
    if (owner == nullptr) {
      return;
    }
    if (!ordinal.contains(owner)) {
      ordinal[owner] = idx++;
    }
  });

  // Phase 2: identify all-level sync points via PostOrder walk. An op is a
  // sync point if it is an external sync op or any direct child is already a
  // sync point (PostOrder guarantees children are visited first). This
  // propagates "contains a sync op" up to every enclosing op, so a sync point
  // is exactly an op that is itself a sync op or contains one.
  block->walk<WalkOrder::PostOrder>([&](Operation *op) {
    bool isCube = isExternalSyncOp(op) &&
                  CVPipeline::getOpCoreType(op) == CoreType::CUBE_ONLY;
    bool isVector = isExternalSyncOp(op) &&
                    CVPipeline::getOpCoreType(op) == CoreType::VECTOR_ONLY;
    for (Region &region : op->getRegions()) {
      for (Block &b : region) {
        for (Operation &child : b) {
          if (cube.syncSet.contains(&child)) {
            isCube = true;
          }
          if (vector.syncSet.contains(&child)) {
            isVector = true;
          }
        }
      }
    }
    if (isCube) {
      cube.syncSet.insert(op);
    }
    if (isVector) {
      vector.syncSet.insert(op);
    }
  });

  // Phase 3: collect block-level sync points (same level as the ops we build
  // edges for), sorted by source-order position.
  for (Operation &op : *block) {
    auto it = ordinal.find(&op);
    if (it == ordinal.end()) {
      continue;
    }
    unsigned pos = it->second;
    if (cube.syncSet.contains(&op)) {
      cube.syncPoints.push_back({&op, pos});
    }
    if (vector.syncSet.contains(&op)) {
      vector.syncPoints.push_back({&op, pos});
    }
  }
  auto byPos = [](const SyncPoint &a, const SyncPoint &b) {
    return a.position < b.position;
  };
  llvm::sort(cube.syncPoints, byPos);
  llvm::sort(vector.syncPoints, byPos);
  cube.prefixCount = buildPrefixCount(cube.syncPoints, idx);
  vector.prefixCount = buildPrefixCount(vector.syncPoints, idx);
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
  auto syncs = syncPointsOf(aCore);
  unsigned lo = positionOf(a);
  unsigned hi = positionOf(b);
  if (lo > hi) {
    std::swap(lo, hi);
  }
  auto it = std::upper_bound(syncs.begin(), syncs.end(), lo,
                             [](unsigned v, const SyncPoint &s) {
                               return v < s.position;
                             });
  return it != syncs.end() && it->position < hi;
}

unsigned SyncWall::segmentOf(Operation *op) const {
  unsigned pos = positionOf(op);
  auto core = CVPipeline::getOpCoreType(op);
  const llvm::SmallVector<unsigned, 4> *prefix = nullptr;
  if (core == CoreType::CUBE_ONLY) {
    prefix = &cube.prefixCount;
  } else if (core == CoreType::VECTOR_ONLY) {
    prefix = &vector.prefixCount;
  }
  if (prefix != nullptr && pos < prefix->size()) {
    return (*prefix)[pos];
  }
  return 0;
}

bool SyncWall::sameSegment(Operation *a, Operation *b) const {
  return CVPipeline::getOpCoreType(a) == CVPipeline::getOpCoreType(b) &&
         segmentOf(a) == segmentOf(b);
}

ArrayRef<SyncPoint> SyncWall::syncPointsOf(CoreType core) const {
  return core == CoreType::VECTOR_ONLY ? vector.syncPoints : cube.syncPoints;
}

bool SyncWall::isSyncPoint(Operation *op, CoreType core) const {
  if (core == CoreType::CUBE_ONLY) {
    return cube.syncSet.contains(op);
  }
  if (core == CoreType::VECTOR_ONLY) {
    return vector.syncSet.contains(op);
  }
  return false;
}
