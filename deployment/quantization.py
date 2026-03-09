"""
Quantization utilities for deployment code generation.

Provides post-training quantization (PTQ) for model weights:
- INT8 symmetric per-tensor quantization (75% memory reduction)
- INT16 symmetric per-tensor quantization (50% memory reduction)
- FLOAT16 reduced precision (50% memory reduction, simpler)

These transform float32 model weights into lower-precision representations
that are emitted as C arrays in the generated embedded code.  At inference
time the microcontroller dequantizes on-the-fly during matrix–vector
multiplications, trading a small accuracy loss for significantly lower
flash/RAM usage.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class QuantizedTensor:
    """A tensor that has been quantized from float32."""
    data: np.ndarray          # int8, int16, or float16
    scale: float              # per-tensor scale factor (float32)
    zero_point: int = 0       # zero point (symmetric → always 0)
    original_shape: tuple = ()
    dtype_name: str = 'int8'  # 'int8', 'int16', 'float16'

    @property
    def memory_bytes(self) -> int:
        """Memory footprint in bytes."""
        return self.data.nbytes

    @property
    def original_bytes(self) -> int:
        """Original float32 memory footprint."""
        n = 1
        for d in self.original_shape:
            n *= d
        return n * 4  # float32 = 4 bytes

    @property
    def compression_ratio(self) -> float:
        if self.original_bytes == 0:
            return 1.0
        return self.original_bytes / self.memory_bytes


@dataclass
class QuantizationReport:
    """Summary of a quantization pass over an entire model."""
    mode: str                              # 'int8', 'int16', 'float16', 'none'
    tensors: Dict[str, QuantizedTensor] = field(default_factory=dict)
    total_original_bytes: int = 0
    total_quantized_bytes: int = 0
    max_quantization_error: float = 0.0    # max absolute error across all tensors
    mean_quantization_error: float = 0.0   # mean absolute error

    @property
    def compression_ratio(self) -> float:
        if self.total_quantized_bytes == 0:
            return 1.0
        return self.total_original_bytes / self.total_quantized_bytes

    @property
    def memory_saved_bytes(self) -> int:
        return self.total_original_bytes - self.total_quantized_bytes


# ─────────────────────────────────────────────────────────────────────
# Core quantization functions
# ─────────────────────────────────────────────────────────────────────

def quantize_symmetric_int8(tensor: np.ndarray) -> QuantizedTensor:
    """Symmetric per-tensor INT8 quantization.

    Maps [-max_abs, +max_abs] → [-127, +127].  Zero maps to 0 exactly.

    Parameters
    ----------
    tensor : np.ndarray
        Float32 weight tensor of any shape.

    Returns
    -------
    QuantizedTensor with int8 data and float32 scale.
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    max_abs = float(np.max(np.abs(tensor)))
    if max_abs < 1e-10:
        # All-zero tensor
        return QuantizedTensor(
            data=np.zeros(tensor.shape, dtype=np.int8),
            scale=1.0,
            zero_point=0,
            original_shape=tensor.shape,
            dtype_name='int8',
        )

    scale = max_abs / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)

    return QuantizedTensor(
        data=quantized,
        scale=float(scale),
        zero_point=0,
        original_shape=tensor.shape,
        dtype_name='int8',
    )


def quantize_symmetric_int16(tensor: np.ndarray) -> QuantizedTensor:
    """Symmetric per-tensor INT16 quantization.

    Maps [-max_abs, +max_abs] → [-32767, +32767].

    Parameters
    ----------
    tensor : np.ndarray
        Float32 weight tensor of any shape.

    Returns
    -------
    QuantizedTensor with int16 data and float32 scale.
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    max_abs = float(np.max(np.abs(tensor)))
    if max_abs < 1e-10:
        return QuantizedTensor(
            data=np.zeros(tensor.shape, dtype=np.int16),
            scale=1.0,
            zero_point=0,
            original_shape=tensor.shape,
            dtype_name='int16',
        )

    scale = max_abs / 32767.0
    quantized = np.clip(np.round(tensor / scale), -32767, 32767).astype(np.int16)

    return QuantizedTensor(
        data=quantized,
        scale=float(scale),
        zero_point=0,
        original_shape=tensor.shape,
        dtype_name='int16',
    )


def quantize_float16(tensor: np.ndarray) -> QuantizedTensor:
    """Convert float32 → float16 (half precision).

    This is the simplest form of quantization — no scale factor needed,
    but values are stored as float16 in C (using ``_Float16`` on supported
    compilers, or packed uint16 with a helper).

    For code generation we still emit float literals (with reduced
    precision) so this mainly cuts the decimal digits emitted.
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    fp16 = tensor.astype(np.float16)

    return QuantizedTensor(
        data=fp16,
        scale=1.0,            # no separate scale needed
        zero_point=0,
        original_shape=tensor.shape,
        dtype_name='float16',
    )


# ─────────────────────────────────────────────────────────────────────
# Quantize a whole model (dispatcher)
# ─────────────────────────────────────────────────────────────────────

QUANTIZATION_MODES = ('none', 'int8', 'int16', 'float16')


def quantize_tensor(tensor: np.ndarray, mode: str) -> QuantizedTensor:
    """Quantize a single tensor using the specified mode."""
    if mode == 'int8':
        return quantize_symmetric_int8(tensor)
    elif mode == 'int16':
        return quantize_symmetric_int16(tensor)
    elif mode == 'float16':
        return quantize_float16(tensor)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def compute_quantization_error(original: np.ndarray, qt: QuantizedTensor) -> Tuple[float, float]:
    """Compute max and mean absolute error of quantized vs original.

    Returns (max_abs_error, mean_abs_error).
    """
    original = np.asarray(original, dtype=np.float32).flatten()
    if qt.dtype_name in ('int8', 'int16'):
        dequantized = qt.data.astype(np.float32).flatten() * qt.scale
    else:
        dequantized = qt.data.astype(np.float32).flatten()
    errors = np.abs(original - dequantized)
    return float(np.max(errors)), float(np.mean(errors))


# ─────────────────────────────────────────────────────────────────────
# C code-generation helpers
# ─────────────────────────────────────────────────────────────────────

def get_c_type(mode: str) -> str:
    """Return the C type name for the given quantization mode."""
    return {
        'int8': 'int8_t',
        'int16': 'int16_t',
        'float16': 'float',   # we emit float literals with fewer digits
        'none': 'float',
    }[mode]


def format_quantized_1d(qt: QuantizedTensor, name: str, precision: int = 6,
                        static: bool = False, const: bool = True) -> str:
    """Format a quantized 1-D tensor as a C array declaration + scale constant.

    Example output (INT8):
        static const int8_t name[128] = { ... };
        static const float name_scale = 0.00358f;
    """
    prefix = ''
    if static:
        prefix += 'static '
    if const:
        prefix += 'const '

    c_type = get_c_type(qt.dtype_name)
    n = qt.data.size
    flat = qt.data.flatten()

    # Format values
    if qt.dtype_name in ('int8', 'int16'):
        vals = ', '.join(str(int(v)) for v in flat)
    else:
        vals = ', '.join(f'{float(v):.{precision}f}' for v in flat)

    # Wrap long lines
    lines = _wrap_c_values(vals)

    result = f"{prefix}{c_type} {name}[{n}] = {{\n    {lines}\n}};"

    # Emit scale constant for integer modes
    if qt.dtype_name in ('int8', 'int16'):
        result += f"\n{prefix}float {name}_scale = {qt.scale:.8f}f;"

    return result


def format_quantized_2d(qt: QuantizedTensor, name: str, rows: int, cols: int,
                        precision: int = 6, static: bool = False, const: bool = True) -> str:
    """Format a quantized 2-D tensor as a flattened C array + scale."""
    prefix = ''
    if static:
        prefix += 'static '
    if const:
        prefix += 'const '

    c_type = get_c_type(qt.dtype_name)
    flat = qt.data.flatten()
    n = flat.size

    if qt.dtype_name in ('int8', 'int16'):
        vals = ', '.join(str(int(v)) for v in flat)
    else:
        vals = ', '.join(f'{float(v):.{precision}f}' for v in flat)

    lines = _wrap_c_values(vals)

    result = (f"// shape ({rows}, {cols})\n"
              f"{prefix}{c_type} {name}[{n}] = {{\n    {lines}\n}};")

    if qt.dtype_name in ('int8', 'int16'):
        result += f"\n{prefix}float {name}_scale = {qt.scale:.8f}f;"

    return result


def format_quantized_3d(qt: QuantizedTensor, name: str,
                        d0: int, d1: int, d2: int,
                        precision: int = 6, static: bool = True, const: bool = True) -> str:
    """Format a quantized 3-D tensor as a flattened C array + scale."""
    prefix = ''
    if static:
        prefix += 'static '
    if const:
        prefix += 'const '

    c_type = get_c_type(qt.dtype_name)
    flat = qt.data.flatten()
    n = flat.size

    if qt.dtype_name in ('int8', 'int16'):
        vals = ', '.join(str(int(v)) for v in flat)
    else:
        vals = ', '.join(f'{float(v):.{precision}f}' for v in flat)

    lines = _wrap_c_values(vals)

    result = (f"// shape ({d0}, {d1}, {d2})\n"
              f"{prefix}{c_type} {name}[{n}] = {{\n    {lines}\n}};")

    if qt.dtype_name in ('int8', 'int16'):
        result += f"\n{prefix}float {name}_scale = {qt.scale:.8f}f;"

    return result


def generate_dequant_helper(mode: str) -> str:
    """Generate C helper for dequantizing during inference.

    For INT8/INT16 modes, the generated code multiplies quantized values
    by their scale factor on-the-fly during matrix–vector products.
    """
    if mode == 'int8':
        return """
// ---- INT8 dequantization helpers ----
// dot_product_q8: computes dot product of int8 weights and float inputs
// result = scale * sum(w_q[i] * x[i])
static inline float dot_product_q8(const int8_t* w, const float* x, int n, float scale) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += (float)w[i] * x[i];
    }
    return acc * scale;
}
"""
    elif mode == 'int16':
        return """
// ---- INT16 dequantization helpers ----
static inline float dot_product_q16(const int16_t* w, const float* x, int n, float scale) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += (float)w[i] * x[i];
    }
    return acc * scale;
}
"""
    else:
        return ""


def generate_quantization_info_comment(report: QuantizationReport) -> str:
    """Generate a C comment block summarising the quantization."""
    if report.mode == 'none':
        return "// Quantization: None (float32 weights)"

    lines = [
        f"// Quantization: {report.mode.upper()}",
        f"// Original size:  {report.total_original_bytes:,} bytes",
        f"// Quantized size: {report.total_quantized_bytes:,} bytes",
        f"// Compression:    {report.compression_ratio:.1f}x "
        f"({report.memory_saved_bytes:,} bytes saved)",
        f"// Max quantization error:  {report.max_quantization_error:.6f}",
        f"// Mean quantization error: {report.mean_quantization_error:.6f}",
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────

def _wrap_c_values(vals: str, width: int = 100) -> str:
    """Insert line-breaks into a comma-separated value string."""
    tokens = vals.split(', ')
    lines: List[str] = []
    line = ''
    for t in tokens:
        candidate = f'{line}, {t}' if line else t
        if len(candidate) > width and line:
            lines.append(line)
            line = t
        else:
            line = candidate
    if line:
        lines.append(line)
    return ',\n    '.join(lines) if len(lines) > 1 else (lines[0] if lines else '')
