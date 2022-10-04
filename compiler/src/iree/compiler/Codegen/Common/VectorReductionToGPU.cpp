// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"

#define DEBUG_TYPE "iree-codegen-reduction-distribution"

namespace mlir {
namespace iree_compiler {

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(), {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
  } else {
    memrefType = MemRefType::get({1}, type, {},
                                 gpu::GPUDialect::getWorkgroupAddressSpace());
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Emit warp reduction code sequence for a given input.
static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           vector::CombiningKind kind, uint32_t size) {
  Value laneVal = input;
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, laneVal, i,
                                                 /*width=*/size,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  return laneVal;
}

/// Emit reduction across a group for a given input.
static Value groupReduction(Location loc, OpBuilder &builder, Value input,
                            vector::CombiningKind kind, uint32_t size,
                            const int warpSize) {
  assert(
      size % warpSize == 0 &&
      "Group reduction only support for sizes aligned on warp size for now.");
  Value laneVal = warpReduction(loc, builder, input, kind, warpSize);
  // if we have more than one warp, reduce across warps.
  if (size > warpSize) {
    uint32_t numWarp = size / warpSize;
    MemRefType memrefType =
        MemRefType::get(numWarp, input.getType(), {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    Value alloc = builder.create<memref::AllocOp>(loc, memrefType);
    Value threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::x);
    Value cstWarpSize = builder.create<arith::ConstantIndexOp>(loc, warpSize);
    Value warpId = builder.create<arith::DivUIOp>(loc, threadX, cstWarpSize);
    // Store the reduction for each warp.
    SmallVector<Value> indices = {warpId};
    builder.create<memref::StoreOp>(loc, laneVal, alloc, indices);
    builder.create<gpu::BarrierOp>(loc);
    VectorType vecType = VectorType::get(numWarp, input.getType());
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value vec =
        builder.create<vector::TransferReadOp>(loc, vecType, alloc, zero);
    laneVal = builder.create<vector::ReductionOp>(loc, kind, vec);
  }
  return laneVal;
}

/// Hoist uniform operations as well as special hal operations that have side
/// effect but are safe to move out of the warp single lane region.
static void moveScalarAndBindingUniformCode(
    vector::WarpExecuteOnLane0Op warpOp) {
  /// Hoist ops without side effect as well as special binding ops.
  auto canBeHoisted = [](Operation *op,
                         function_ref<bool(Value)> definedOutside) {
    return llvm::all_of(op->getOperands(), definedOutside) &&
           (isSideEffectFree(op) ||
            isa<IREE::HAL::InterfaceBindingSubspanOp,
                IREE::HAL::InterfaceConstantLoadOp, memref::AssumeAlignmentOp>(
                op)) &&
           op->getNumRegions() == 0;
  };
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return result.getType().isa<VectorType>();
    });
    if (!hasVectorResult && canBeHoisted(&op, isDefinedOutsideOfBody))
      opsToMove.insert(&op);
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove) op->moveBefore(warpOp);
}

namespace {

/// Pattern to convert InsertElement to broadcast, this is a workaround until
/// MultiDimReduction distribution is supported.
class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
 public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1) return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

class VectorReduceToGPUPass
    : public VectorReduceToGPUBase<VectorReduceToGPUPass> {
 public:
  explicit VectorReduceToGPUPass(std::function<int(func::FuncOp)> getWarpSize)
      : getWarpSize(getWarpSize) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, gpu::GPUDialect,
                    AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Pre-process multiDimReductions.
    // TODO: Remove once MultiDimReduce is supported by distribute patterns.
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      // Add clean up patterns after lowering of multidimreduce lowering.
      patterns.add<InsertElementToBroadcast>(ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs()
          << "\n--- After Step 1: Preprocessing of reduction ops ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    assert(workgroupSize[1] == 1 && workgroupSize[2] == 1);
    // 2. Create the warp op and move the function body into it.
    const int groupSize = workgroupSize[0];
    Location loc = funcOp.getLoc();
    OpBuilder builder(funcOp);
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                   gpu::Dimension::x);
    auto cstGroupSize = builder.create<arith::ConstantIndexOp>(loc, groupSize);
    auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
        loc, TypeRange(), threadX.getResult(), groupSize);
    warpOp.getWarpRegion().takeBody(funcOp.getBody());
    Block &newBlock = funcOp.getBody().emplaceBlock();
    threadX->moveBefore(&newBlock, newBlock.end());
    cstGroupSize->moveBefore(&newBlock, newBlock.end());
    warpOp->moveBefore(&newBlock, newBlock.end());
    warpOp.getWarpRegion().getBlocks().back().back().moveBefore(&newBlock,
                                                                newBlock.end());
    builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
    builder.create<vector::YieldOp>(loc);

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 2: Adding the distribution op ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 3. Hoist the scalar code outside of the warp region.
    moveScalarAndBindingUniformCode(warpOp);

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 3: Hoist uniform code ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 4. Distribute transfer write operations.
    {
      auto distributionFn = [](vector::TransferWriteOp writeOp) {
        // Create a map (d0, d1) -> (d1) to distribute along the inner
        // dimension. Once we support n-d distribution we can add more
        // complex cases.
        int64_t vecRank = writeOp.getVectorType().getRank();
        OpBuilder builder(writeOp.getContext());
        auto map =
            AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
        return map;
      };
      RewritePatternSet patterns(ctx);
      vector::populateDistributeTransferWriteOpPatterns(patterns,
                                                        distributionFn);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 4: Distribute transfer write ops ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 5. Propagate vector distribution.
    {
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(patterns);
      auto getWarpSize = this->getWarpSize ? this->getWarpSize
                                           : [](func::FuncOp) { return 32; };
      auto groupReductionFn = [&](Location loc, OpBuilder &builder, Value input,
                                  vector::CombiningKind kind, uint32_t size) {
        return groupReduction(loc, builder, input, kind, size,
                              getWarpSize(funcOp));
      };
      vector::populateDistributeReduction(patterns, groupReductionFn);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 5: Propagate distribution ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 6. Lower the remaining WarpExecuteOnLane0 ops.
    {
      RewritePatternSet patterns(ctx);
      vector::WarpExecuteOnLane0LoweringOptions options;
      options.warpAllocationFn = allocateGlobalSharedMemory;
      options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp) {
        builder.create<gpu::BarrierOp>(loc);
      };
      vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 6: Lower remaining ops ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });
  }

 private:
  std::function<int(func::FuncOp)> getWarpSize;
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass(
    std::function<int(func::FuncOp)> getWarpSize) {
  return std::make_unique<VectorReduceToGPUPass>(getWarpSize);
}

}  // namespace iree_compiler
}  // namespace mlir
