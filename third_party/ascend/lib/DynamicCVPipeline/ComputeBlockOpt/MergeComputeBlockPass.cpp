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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "DynamicCVPipeline/Common/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-compute-block";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

// ============================================================================
// 数据结构
// ============================================================================

/// 代表一个 ComputeBlock：同一 block_id 的一组 op
struct ComputeBlock {
    int id;                            // block_id 值
    CVPipeline::CoreType coreType;     // CUBE_ONLY / VECTOR_ONLY
    SmallVector<Operation *> ops;      // 组内所有 op（按 IR 顺序）
};

using BlockIdx = unsigned;  // 在 blocks 数组中的下标

// ============================================================================
// 子函数实现
// ============================================================================

/// 收集 ModuleOp 中最内层包含 linalg::MatmulOp 的 scf::ForOp 的 body Block（去重）。
static void collectInnermostMatmulLoopBlocks(ModuleOp module, SmallVectorImpl<Block *> &blocks)
{
    DenseSet<Block *> seen;
    module->walk([&](linalg::MatmulOp matmul) {
        Operation *parent = matmul->getParentOp();
        while (parent) {
            if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
                Block *body = forOp.getBody();
                if (seen.insert(body).second) {
                    blocks.push_back(body);
                }
                break;
            }
            parent = parent->getParentOp();
        }
    });
}

/// Step 1+2: 分组并建立 block 级 SSA 依赖图。
static void groupAndBuildGraph(Block *block, SmallVectorImpl<ComputeBlock> &blocks,
                               DenseMap<int, BlockIdx> &idToIdx,
                               SmallVectorImpl<SmallVector<BlockIdx>> &succs,
                               SmallVectorImpl<SmallVector<BlockIdx>> &preds)
{
    // Step 1: 按 block_id 分组（按 IR 出现顺序，递归包含嵌套区域）
    block->walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::IsTerminator>()) {
            return;
        }
        auto optId = CVPipeline::getOpBlockId(op);
        if (!optId.has_value()) {
            return;
        }
        int bid = *optId;
        auto it = idToIdx.find(bid);
        if (it == idToIdx.end()) {
            BlockIdx idx = blocks.size();
            idToIdx[bid] = idx;
            blocks.push_back({bid, CVPipeline::getOpCoreType(op), {}});
            it = idToIdx.find(bid);
        }
        blocks[it->second].ops.push_back(op);
    });

    if (blocks.empty()) {
        return;
    }

    // Step 2: 构建 ComputeBlock 级 SSA 依赖图
    succs.resize(blocks.size());
    preds.resize(blocks.size());

    // 记录当前 op 所在的 ComputeBlock 下标
    DenseMap<Operation *, BlockIdx> opToBlockIdx;
    for (BlockIdx idx = 0; idx < blocks.size(); ++idx) {
        for (Operation *op : blocks[idx].ops) {
            opToBlockIdx[op] = idx;
        }
    }

    DenseSet<std::pair<BlockIdx, BlockIdx>> seenEdges;
    for (BlockIdx curIdx = 0; curIdx < blocks.size(); ++curIdx) {
        for (Operation *op : blocks[curIdx].ops) {
            for (Value operand : op->getOperands()) {
                Operation *defOp = operand.getDefiningOp();
                if (!defOp) {
                    continue; // block argument
                }
                Operation *ancestor = CVPipeline::getAncestorInBlock(defOp, block);
                if (!ancestor) {
                    continue; // 不在本 block 中
                }
                auto ancIt = opToBlockIdx.find(ancestor);
                if (ancIt == opToBlockIdx.end()) {
                    continue;
                }
                BlockIdx ancIdx = ancIt->second;
                if (ancIdx == curIdx) {
                    continue; // 同一个 ComputeBlock 内部边，忽略
                }

                // 去重
                if (!seenEdges.insert({ancIdx, curIdx}).second) {
                    continue;
                }
                succs[ancIdx].push_back(curIdx);
                preds[curIdx].push_back(ancIdx);
            }
        }
    }
}

/// Step 3: 找出候选 VECTOR 块。
static void findCandidates(const SmallVectorImpl<ComputeBlock> &blocks,
                           const SmallVectorImpl<SmallVector<BlockIdx>> &succs,
                           const SmallVectorImpl<SmallVector<BlockIdx>> &preds,
                           DenseSet<BlockIdx> &candidates)
{
    auto hasCUBEPred = [&](BlockIdx bid) {
        for (BlockIdx p : preds[bid]) {
            if (blocks[p].coreType == CVPipeline::CoreType::CUBE_ONLY) {
                return true;
            }
        }
        return false;
    };
    auto hasCUBESucc = [&](BlockIdx bid) {
        for (BlockIdx s : succs[bid]) {
            if (blocks[s].coreType == CVPipeline::CoreType::CUBE_ONLY) {
                return true;
            }
        }
        return false;
    };
    auto hasTensorResult = [&](const ComputeBlock &blk) {
        for (Operation *op : blk.ops) {
            for (Value result : op->getResults()) {
                if (isa<TensorType>(result.getType())) {
                    return true;
                }
            }
        }
        return false;
    };

    for (BlockIdx bid = 0; bid < blocks.size(); ++bid) {
        if (blocks[bid].coreType != CVPipeline::CoreType::VECTOR_ONLY) {
            continue;
        }
        if (!hasCUBEPred(bid) || !hasCUBESucc(bid)) {
            continue;
        }
        if (!hasTensorResult(blocks[bid])) {
            continue;
        }
        candidates.insert(bid);
    }
}

/// Step 4: Union-Find 发现候选 VECTOR 块连通分量。
static void unionFindGroups(const DenseSet<BlockIdx> &candidates,
                            const SmallVectorImpl<SmallVector<BlockIdx>> &succs,
                            const SmallVectorImpl<SmallVector<BlockIdx>> &preds,
                            SmallVectorImpl<SmallVector<BlockIdx>> &groups)
{
    // 初始化 Union-Find
    DenseMap<BlockIdx, BlockIdx> parent;
    for (BlockIdx bid : candidates) {
        parent[bid] = bid;
    }

    auto find = [&](BlockIdx x) -> BlockIdx {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](BlockIdx a, BlockIdx b) {
        BlockIdx ra = find(a);
        BlockIdx rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    };

    // 遍历候选块之间的直接边
    for (BlockIdx u : candidates) {
        for (BlockIdx v : succs[u]) {
            if (candidates.contains(v)) {
                unite(u, v);
            }
        }
        for (BlockIdx v : preds[u]) {
            if (candidates.contains(v)) {
                unite(u, v);
            }
        }
    }

    // 按连通分量分组，只保留 size > 1 的 group
    DenseMap<BlockIdx, SmallVector<BlockIdx>> groupsByRoot;
    for (BlockIdx bid : candidates) {
        groupsByRoot[find(bid)].push_back(bid);
    }
    for (auto &kv : groupsByRoot) {
        if (kv.second.size() > 1) {
            groups.push_back(std::move(kv.second));
        }
    }
}

/// Step 5: 执行合并。
static void applyMerge(const SmallVectorImpl<SmallVector<BlockIdx>> &groups,
                       const SmallVectorImpl<ComputeBlock> &blocks,
                       CVPipeline::ComputeBlockIdManager &bm)
{
    for (const auto &group : groups) {
        // blocks 数组按 IR 顺序构建，min(group) 即 IR 中最先出现的块
        BlockIdx target = *std::min_element(group.begin(), group.end());
        int targetId = blocks[target].id;

        LOG_DEBUG("Merging " << group.size() << " vector blocks, target block_id=" << targetId);

        for (BlockIdx bid : group) {
            if (bid == target) {
                continue;
            }
            for (Operation *op : blocks[bid].ops) {
                bm.updateBlockId(op, targetId);
            }
        }
    }
}

/// 对一个 Block 执行合并流程：分组 → 建图 → 找候选 → 连通分量 → 执行合并
static void tryMergeInBlock(Block *block, CVPipeline::ComputeBlockIdManager &bm)
{
    // Step 1+2: 分组并建立依赖图
    SmallVector<ComputeBlock> blocks;
    DenseMap<int, BlockIdx> idToIdx;
    SmallVector<SmallVector<BlockIdx>> succs;
    SmallVector<SmallVector<BlockIdx>> preds;
    groupAndBuildGraph(block, blocks, idToIdx, succs, preds);

    if (blocks.empty()) {
        return;
    }

    // Step 3: 找出候选 VECTOR 块
    DenseSet<BlockIdx> candidates;
    findCandidates(blocks, succs, preds, candidates);

    if (candidates.size() < 2) {
        LOG_DEBUG("Found " << candidates.size() << " candidate(s), need at least 2 to merge");
        return;
    }

    // Step 4: 发现连通分量
    SmallVector<SmallVector<BlockIdx>> groups;
    unionFindGroups(candidates, succs, preds, groups);

    if (groups.empty()) {
        LOG_DEBUG("No adjacent candidate groups to merge");
        return;
    }

    // Step 5: 执行合并
    applyMerge(groups, blocks, bm);

    LOG_DEBUG("Merged " << groups.size() << " group(s) in block");
}

// ============================================================================
// Pass 定义
// ============================================================================

namespace {

class MergeComputeBlockPass : public PassWrapper<MergeComputeBlockPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeComputeBlockPass)

    MergeComputeBlockPass() = default;

    StringRef getArgument() const override { return "merge-compute-block"; }

    StringRef getDescription() const override
    {
        return "Merge adjacent vector compute blocks between CUBE blocks";
    }

    void runOnOperation() override
    {
        if (!CVPipeline::isMergeComputeBlockEnabled()) {
            return;
        }
        ModuleOp module = getOperation();
        LOG_DEBUG("Before: " << *module);

        SmallVector<Block *> blocksToProcess;
        collectInnermostMatmulLoopBlocks(module, blocksToProcess);

        if (blocksToProcess.empty()) {
            LOG_DEBUG("No innermost matmul loop blocks found, skipping");
            return;
        }

        CVPipeline::ComputeBlockIdManager bm(module);
        for (Block *block : blocksToProcess) {
            tryMergeInBlock(block, bm);
        }

        LOG_DEBUG("After: " << *module);
    }
};

} // namespace

// ============================================================================
// Pass 注册
// ============================================================================

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeComputeBlockPass()
{
    return std::make_unique<MergeComputeBlockPass>();
}

void registerMergeComputeBlockPass()
{
    PassRegistration<MergeComputeBlockPass> reg;
}

} // namespace triton
} // namespace mlir
