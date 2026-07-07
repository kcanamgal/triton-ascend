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

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include <chrono>

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "pre-check-dynamic-cv-pipeline-available";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)\
  LLVM_DEBUG({\
    DBGS();\
    llvm::dbgs() << __VA_ARGS__ << "\n";\
})

void PreCheckAvailablePass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Enter PreCheckAvailable pass.");

    auto startTime = std::chrono::steady_clock::now();
    LDBG("Before PreCheck:\n" << module);

    if (llvm::failed(runPreCheckBlacklist(module)) || llvm::failed(runPreCheckMatmul(module))) {
        CVPipeline::setFallbackAttr(module);
        signalPassFailure();
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    llvm::errs() << "[PreCheckAvailable] pass elapsed time: " << duration << " ms\n";
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckAvailablePass()
{
    return std::make_unique<PreCheckAvailablePass>();
}

void registerPreCheckAvailablePasses()
{
    registerPass(createPreCheckBlacklistPass);
    registerPass(createPreCheckMatmulPass);
    registerPass(createPreCheckAvailablePass);
}

} // namespace triton
} // namespace mlir