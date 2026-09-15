"""Small trusted CUDA Driver adapter. It loads cubins, never candidate Python."""
from __future__ import annotations

import ctypes as ct

import numpy as np


class Driver:
    def __init__(self, *, create_context=True):
        try:
            self.lib = ct.CDLL("libcuda.so.1")
        except OSError as error:
            raise RuntimeError("CUDA Driver unavailable; native execution requires the H100 host") from error
        self._bind("cuInit", [ct.c_uint])
        self._bind("cuDeviceGet", [ct.POINTER(ct.c_int), ct.c_int])
        self._bind("cuCtxCreate_v2", [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int])
        self._bind("cuCtxDestroy_v2", [ct.c_void_p])
        self._bind("cuCtxSynchronize", [])
        self._bind("cuModuleLoadData", [ct.POINTER(ct.c_void_p), ct.c_void_p])
        self._bind("cuModuleGetFunction", [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p])
        self._bind("cuMemAlloc_v2", [ct.POINTER(ct.c_uint64), ct.c_size_t])
        self._bind("cuMemFree_v2", [ct.c_uint64])
        self._bind("cuMemcpyHtoD_v2", [ct.c_uint64, ct.c_void_p, ct.c_size_t])
        self._bind("cuMemcpyDtoH_v2", [ct.c_void_p, ct.c_uint64, ct.c_size_t])
        self._bind("cuLaunchKernel", [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint,
                                     ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_void_p,
                                     ct.POINTER(ct.c_void_p), ct.c_void_p])
        self._bind("cuFuncSetAttribute", [ct.c_void_p, ct.c_int, ct.c_int])
        self._bind("cuEventCreate", [ct.POINTER(ct.c_void_p), ct.c_uint])
        self._bind("cuEventRecord", [ct.c_void_p, ct.c_void_p])
        self._bind("cuEventElapsedTime", [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p])
        self._bind("cuEventDestroy_v2", [ct.c_void_p])
        self.check(self.lib.cuInit(0))
        self.context = None
        if create_context:
            device = ct.c_int()
            self.check(self.lib.cuDeviceGet(ct.byref(device), 0))
            self.context = ct.c_void_p()
            self.check(self.lib.cuCtxCreate_v2(ct.byref(self.context), 0, device))
        self.allocations = []

    def _bind(self, name, types):
        fn = getattr(self.lib, name)
        fn.argtypes = types
        fn.restype = ct.c_int

    def check(self, code):
        if code:
            raise RuntimeError(f"CUDA Driver error {code}; discard this worker context")

    def load(self, binary, symbol, shared_bytes=0):
        module, function = ct.c_void_p(), ct.c_void_p()
        image = ct.create_string_buffer(binary)
        self.check(self.lib.cuModuleLoadData(ct.byref(module), ct.cast(image, ct.c_void_p)))
        self.check(self.lib.cuModuleGetFunction(ct.byref(function), module, symbol.encode()))
        if shared_bytes > 48 * 1024:
            self.check(self.lib.cuFuncSetAttribute(function, 8, shared_bytes))
        return function

    def upload(self, array):
        array = np.ascontiguousarray(array)
        pointer = ct.c_uint64()
        self.check(self.lib.cuMemAlloc_v2(ct.byref(pointer), array.nbytes))
        self.allocations.append(pointer.value)
        self.check(self.lib.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
        return pointer.value

    def download(self, pointer, shape, dtype):
        result = np.empty(shape, dtype=dtype)
        self.check(self.lib.cuMemcpyDtoH_v2(result.ctypes.data, pointer, result.nbytes))
        return result

    def launch(self, function, args, *, grid, block, shared=0, stream=0):
        params = (ct.c_void_p * len(args))(*(ct.cast(ct.byref(arg), ct.c_void_p) for arg in args))
        self.check(self.lib.cuLaunchKernel(function, *grid, block, 1, 1, shared,
                                            ct.c_void_p(stream), params, None))

    def synchronize(self):
        self.check(self.lib.cuCtxSynchronize())

    def event(self):
        event = ct.c_void_p()
        self.check(self.lib.cuEventCreate(ct.byref(event), 0))
        return event

    def record(self, event):
        self.check(self.lib.cuEventRecord(event, None))

    def elapsed(self, start, end):
        ms = ct.c_float()
        self.check(self.lib.cuEventElapsedTime(ct.byref(ms), start, end))
        return ms.value

    def close(self):
        for pointer in self.allocations:
            self.check(self.lib.cuMemFree_v2(pointer))
        self.allocations.clear()
        if self.context:
            self.check(self.lib.cuCtxDestroy_v2(self.context))
            self.context = None


def quantizer_args(input_ptr, codes_ptr, scales_ptr, case):
    return [ct.c_uint64(input_ptr), ct.c_uint64(codes_ptr), ct.c_uint64(scales_ptr),
            ct.c_int64(case.rows), ct.c_int64(case.columns), ct.c_int64(case.row_stride),
            ct.c_int64(case.column_stride), ct.c_int64((case.rows + 3) // 4 * 4), ct.c_float(case.epsilon)]
