// problem size = 3456x1024x8192
// accum data type = f16
// input data type = f16
// matmul label convention is matmul_accumDataType_inputDataType
func.func @matmul_f16_f16() {
  %lhs = util.unfoldable_constant dense<1.00> : tensor<3456x8192xf16>
  %rhs = util.unfoldable_constant dense<0.01> : tensor<8192x1024xf16>
  %c0 = arith.constant 0.0 : f16
  %init = linalg.init_tensor[3456, 1024] : tensor<3456x1024xf16>
  %CC = linalg.fill ins(%c0 : f16) outs(%init : tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<3456x8192xf16>, tensor<8192x1024xf16>)
                    outs(%CC: tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  check.expect_almost_eq_const(%D, dense<20.0938> : tensor<3456x1024xf16>) : tensor<3456x1024xf16>
  return
}