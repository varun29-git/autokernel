import triton
import triton.language as tl


@triton.jit
def fp8_rows(X, Codes, Scales, rows, width, row_stride, column_stride, scale_stride,
             epsilon, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * 128 + tl.arange(0, 128)
    x = tl.load(X + r[:, None].to(tl.int64) * row_stride + c[None, :] * column_stride,
                (r[:, None] < rows) & (c[None, :] < width), other=0).to(tl.float32)
    amplitude = tl.maximum(tl.max(tl.abs(x), axis=1), epsilon)
    scale = tl.where(amplitude > 0, tl.maximum(amplitude * (1.0 / 448), 1.1754943508222875e-38),
                     tl.where(amplitude == 0, 1., amplitude))
    quotient = tl.div_rn(x, scale[:, None])
    codes = quotient.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(Codes + r[:, None].to(tl.int64) * width + c[None, :], codes,
             (r[:, None] < rows) & (c[None, :] < width))
    tl.store(Scales + r.to(tl.int64) + tl.program_id(1) * scale_stride, scale, r < rows)
