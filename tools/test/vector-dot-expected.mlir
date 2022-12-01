// RUN: (iree-compile --iree-hal-target-backends=cuda %s | iree-run-module --device=cuda --entry_function=vector_dot   --function_input="4xf32=[1 1 1 1]" --function_input="4xf32=[2 2 2 2]" --function_input=f32=0.0 --expected_output=f32=8.0) | FileCheck %s --check-prefix=SUCCESS-INPUT-COMMAND-LINE
// RUN: (iree-compile --iree-hal-target-backends=cuda %s | iree-run-module --device=cuda --entry_function=vector_dot   --function_input=vector_input.npy --function_input=vector_input.npy --function_input=f32=0.0 --expected_output=f32=4.0) | FileCheck %s --check-prefix=SUCCESS-INPUT-NPY

// SUCCESS-INPUT-COMMAND-LINE: [SUCCESS]
// SUCCESS-INPUT-NPY: [SUCCESS]
func.func @vector_dot(%vector_a: tensor<4xf32>, %vector_b: tensor<4xf32>, %out: tensor<f32>) -> tensor<f32> {
  %dot = linalg.dot ins(%vector_a, %vector_b : tensor<4xf32>, tensor<4xf32>)
                    outs(%out : tensor<f32>) -> tensor<f32>
  return %dot : tensor<f32>
}