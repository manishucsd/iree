// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// clang-format off

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h.inc" // IWYU pragma: export

// clang-format on

//===---------------------------------------------------------------------===//
// LinalgExt Dialect Helpers
//===---------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Returns the encoding attribute from the type if there is an encoding.
/// Otherwise, returns null.
EncodingAttr getEncodingAttr(RankedTensorType type);

/// Returns the ContractionDimensions for the encoding user_indexing_maps.
FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding);
} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
