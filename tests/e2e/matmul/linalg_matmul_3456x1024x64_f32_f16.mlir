// problem size = 3456x1024x64
// accum data type = f32
// input data type = f16
// matmul label convention is matmul_accumDataType_inputDataType
func.func @matmul_f32_f16() {
  %lhs = util.unfoldable_constant dense<0.5> : tensor<3456x64xf16>
  %rhs = util.unfoldable_constant dense<0.5> : tensor<64x1024xf16>
  %c0 = arith.constant 0.0 : f32
  %init = linalg.init_tensor[3456, 1024] : tensor<3456x1024xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<3456x64xf16>, tensor<64x1024xf16>)
                    outs(%CC: tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  check.expect_almost_eq_const(%D, dense<16.00> : tensor<3456x1024xf32>) : tensor<3456x1024xf32>
  return
}