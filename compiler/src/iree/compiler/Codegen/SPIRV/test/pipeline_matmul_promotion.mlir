// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>

hal.executable @matmul_128x256x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [65535, 65535, 65535],
      subgroup_size = 32>>}> {
    hal.executable.export public @matmul_128x256x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_128x256x64() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x512xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:512x256xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x256xf32>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x512xf32> -> tensor<128x512xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x256xf32> -> tensor<512x256xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<128x256xf32>
        %7 = linalg.init_tensor [128, 256] : tensor<128x256xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %9 = linalg.matmul ins(%4, %5 : tensor<128x512xf32>, tensor<512x256xf32>) outs(%8 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
                ins(%9, %6 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%7 : tensor<128x256xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %11 = arith.divf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<128x256xf32>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : tensor<128x256xf32> -> !flow.dispatch.tensor<writeonly:128x256xf32>
        return
      }
    }
  }
}

// CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<256 x vector<4xf32>>)>, Workgroup>
// CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x vector<4xf32>>)>, Workgroup>

// CHECK-LABEL: spirv.func @matmul_128x256x64

//   CHECK-COUNT-5: spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>

//           CHECK: spirv.mlir.loop
//           CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-5:   spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//  CHECK-COUNT-64:   spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-128:   spirv.GL.Fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-5:   spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//           CHECK:   spirv.mlir.merge

//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-5: spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//  CHECK-COUNT-64: spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-128: spirv.GL.Fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-4: spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//   CHECK-COUNT-4: spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-4: spirv.Store "StorageBuffer" %{{.+}}, %{{.+}} : vector<4xf32>
