module {
  func.func @gemm(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %sum = arith.constant 0.0 : f32
        %acc = scf.for %k = %c0 to %c64 step %c1 iter_args(%s = %sum) -> f32 {
          %a = memref.load %A[%i, %k] : memref<64x64xf32>
          %b = memref.load %B[%k, %j] : memref<64x64xf32>
          %m = arith.mulf %a, %b : f32
          %t = arith.addf %s, %m : f32
          scf.yield %t : f32
        }
        memref.store %acc, %C[%i, %j] : memref<64x64xf32>
      }
    }
    return
  }

  func.func @softmax_exp(%x: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %empty = tensor.empty() : tensor<16x64xf32>
    %out = scf.for %r = %c0 to %c16 step %c1 iter_args(%acc = %empty) -> tensor<16x64xf32> {
      %row = scf.for %c = %c0 to %c64 step %c1 iter_args(%racc = %acc) -> tensor<16x64xf32> {
        %v = tensor.extract %x[%r, %c] : tensor<16x64xf32>
        %e = math.exp %v : f32
        %n = tensor.insert %e into %racc[%r, %c] : tensor<16x64xf32>
        scf.yield %n : tensor<16x64xf32>
      }
      scf.yield %row : tensor<16x64xf32>
    }
    return %out : tensor<16x64xf32>
  }
}
