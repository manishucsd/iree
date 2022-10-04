// RUN: iree-dialects-opt --transform-dialect-interpreter --split-input-file --verify-diagnostics --allow-unregistered-dialect %s

func.func public @no_outlining() {
  // expected-note @below {{target op}}
  "some.operation"() ({}, {}) : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some_operation : benefit(1) {
    %0 = operation "some.operation"
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @some_operation in %arg1
    // Make sure we don't crash on wrong operation type.
    // expected-error@below {{failed to outline}}
    transform.loop.outline %0 {func_name = "outlined"}
  }
}

// -----

func.func @repeated_match(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // expected-error @below {{operation tracked by two handles}}
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target1 : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute = @repeated_match
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  // An exact copy of the above, but with a different name.
  pdl.pattern @pdl_target2 : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute = @repeated_match
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    // expected-note @below {{handle}}
    %0 = pdl_match @pdl_target1 in %arg1
    // expected-note @below {{handle}}
    %1 = pdl_match @pdl_target2 in %arg1

    // Add references to handles produced by match so that they are not DCE'd.
    transform.structured.tile %0 [32, 32, 32]
    transform.structured.tile %1 [32, 32, 32]
  }
}
