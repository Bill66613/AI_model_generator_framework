"""
FeatureBlock — generates har_features.h and har_features.cpp.

The generated C++ code extracts features from a raw sensor window.
It MUST exactly match Python utils/feature_extraction.py.

Statistical formula parity (verified against pandas/numpy):
  std        — population (numpy.std, ddof=0)
  skewness   — pandas adjusted Fisher-Pearson:
               n / ((n-1)(n-2)) * sum((x-mean)^3 / sample_std^3)
               where sample_std uses ddof=1
  kurtosis   — pandas excess kurtosis:
               n(n+1)/((n-1)(n-2)(n-3)) * sum(z^4) - 3(n-1)^2/((n-2)(n-3))
               where z uses sample_std (ddof=1)
  zero_crossings       — count where data[i-1]*data[i] < 0
  mean_crossing_rate   — count where (x-mean)[i-1]*(x-mean)[i] < 0  /  n

Feature extraction order for orientation_invariant_time_only (33 features):
  acc_mag:      mean, std, min, max, range, median, q25, q75, iqr,
                skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate
  gyro_mag:     same 15
  acc_jerk_mag: mean, std, max
  (+ gyro_jerk_mag mean/std/max if present)
  (+ optional scalar features: acc_sma, tilt_pitch, tilt_roll,
     acc_mag_autocorr_lag1, acc_jerk_mag_peak_count)

This order is the CANONICAL C++ order. The Python feature names stored in
model metadata have been reordered (by the factory) to match this BEFORE
the generator is called. Do not change this order without updating the
factory's get_cpp_feature_order() function.
"""

from __future__ import annotations
from typing import Dict, List, Tuple


def _cf(v: float) -> str:
    """Format a Python float as a valid C99 float literal (always has decimal point).

    The :.Xg format specifier drops the decimal for integer-valued floats
    (e.g. 1.0 → '1'), which produces '1f' — an invalid GCC literal.
    """
    s = f"{v:.10g}"
    if '.' not in s and 'e' not in s.lower():
        s += '.0'
    return s + 'f'


def _cf(v: float) -> str:
    """Format a Python float as a valid C99 float literal (always has decimal point)."""
    s = f"{v:.10g}"
    # :.Xg drops the decimal for integer-valued floats (e.g. 1.0 → '1').
    # Appending just 'f' would give '1f' which GCC rejects.
    if '.' not in s and 'e' not in s.lower():
        s += '.0'
    return s + 'f'


class FeatureBlock:
    """Generates har_features.h and har_features.cpp."""

    # Canonical 15-stat order per magnitude signal
    _MAG_STATS = [
        "mean", "std", "min", "max", "range",
        "median", "q25", "q75", "iqr",
        "skewness", "kurtosis", "rms", "energy",
        "zero_crossings", "mean_crossing_rate",
    ]

    def __init__(
        self,
        feature_names: List[str],
        window_size: int,
        sampling_rate: int,
        preprocessing: Dict,
        platform: str,
    ):
        self.feature_names = feature_names
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.preprocessing = preprocessing
        self.platform = platform

        fset = set(feature_names)
        self.has_acc_mag = any(f.startswith("acc_mag_") for f in fset)
        self.has_gyro_mag = any(f.startswith("gyro_mag_") for f in fset)
        self.has_acc_jerk = any(f.startswith("acc_jerk_mag_") for f in fset)
        self.has_gyro_jerk = any(f.startswith("gyro_jerk_mag_") for f in fset)
        self.has_per_axis = any(
            f.startswith(("aX_", "aY_", "aZ_", "gX_", "gY_", "gZ_")) for f in fset
        )
        self.has_freq = any(
            "dominant_frequency" in f or "spectral" in f or "energy_low_freq" in f
            for f in fset
        )
        self.has_sma = "acc_sma" in fset
        self.has_tilt = "tilt_pitch" in fset or "tilt_roll" in fset
        self.has_autocorr = "acc_mag_autocorr_lag1" in fset
        self.has_peak_count = "acc_jerk_mag_peak_count" in fset

    def generate(self) -> Tuple[str, str]:
        """Return (har_features.h content, har_features.cpp content)."""
        return self._header(), self._impl()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> str:
        feature_list = "\n".join(
            f" *   [{i:3d}] {name}" for i, name in enumerate(self.feature_names)
        )
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_features.h — Feature extraction for HAR
 *
 * Extracts HAR_NUM_FEATURES features from a raw sensor window.
 *
 * Input:  window[HAR_WINDOW_SIZE][HAR_N_CHANNELS]
 *         channels: [aX, aY, aZ, gX, gY, gZ]
 *         units: m/s² (accelerometer), deg/s (gyroscope)
 *
 * Output: features[HAR_NUM_FEATURES]  in canonical C++ order:
{feature_list}
 *
 * PARITY NOTE: Formulas match Python utils/feature_extraction.py exactly.
 * Do NOT modify statistical formulas without updating the Python side too.
 */
void har_extract_features(
    const float window[HAR_WINDOW_SIZE][HAR_N_CHANNELS],
    float features[HAR_NUM_FEATURES]
);
"""

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _impl(self) -> str:
        preproc_functions = self._preprocessing_functions()
        preproc_calls = self._preprocessing_calls()

        if self.has_per_axis and not (self.has_acc_mag or self.has_gyro_mag):
            extraction = self._per_axis_extraction()
        else:
            extraction = self._orientation_invariant_extraction()

        sorted_arr_size = max(self.window_size + 10, 210)  # a bit of headroom

        return f"""\
#include "har_features.h"
#include <math.h>
#include <string.h>

/* ===========================================================
 * Statistical helper functions.
 * All formulas match Python/pandas exactly — do NOT change.
 * =========================================================== */

static float _mean(const float *x, int n) {{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    return s / (float)n;
}}

/* Population std (numpy.std, ddof=0) — used for 'std' feature. */
static float _std_pop(const float *x, int n, float mn) {{
    float s = 0.0f;
    for (int i = 0; i < n; i++) {{ float d = x[i] - mn; s += d * d; }}
    return sqrtf(s / (float)n);
}}

/* Sample std (ddof=1) — used ONLY as denominator for skewness/kurtosis z-scores. */
static float _std_sample(const float *x, int n, float mn) {{
    if (n <= 1) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {{ float d = x[i] - mn; s += d * d; }}
    return sqrtf(s / (float)(n - 1));
}}

/* Insertion sort for median/percentiles — in-place. */
static void _sort(float *buf, int n) {{
    for (int i = 1; i < n; i++) {{
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {{ buf[j + 1] = buf[j]; j--; }}
        buf[j + 1] = key;
    }}
}}

/* Linear-interpolation percentile (matches numpy.percentile default). */
static float _percentile(const float *sorted, int n, float p) {{
    float idx = (float)(n - 1) * p;
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= n) return sorted[n - 1];
    return sorted[lo] + (idx - (float)lo) * (sorted[hi] - sorted[lo]);
}}

/*
 * Pandas-compatible skewness (adjusted Fisher-Pearson):
 *   n / ((n-1)(n-2)) * sum( ((x-mean)/sample_std)^3 )
 */
static float _skewness(const float *x, int n, float mn, float ss) {{
    if (n < 3 || ss < 1e-7f) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {{
        float z = (x[i] - mn) / ss;
        s += z * z * z;
    }}
    float fn = (float)n;
    return (fn / ((fn - 1.0f) * (fn - 2.0f))) * s;
}}

/*
 * Pandas-compatible excess kurtosis:
 *   n(n+1)/((n-1)(n-2)(n-3)) * sum(z^4) - 3(n-1)^2/((n-2)(n-3))
 *   where z = (x - mean) / sample_std
 */
static float _kurtosis(const float *x, int n, float mn, float ss) {{
    if (n < 4 || ss < 1e-7f) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {{
        float z = (x[i] - mn) / ss;
        float z2 = z * z;
        s += z2 * z2;
    }}
    float fn = (float)n;
    float A = fn * (fn + 1.0f) / ((fn - 1.0f) * (fn - 2.0f) * (fn - 3.0f)) * s;
    float B = 3.0f * (fn - 1.0f) * (fn - 1.0f) / ((fn - 2.0f) * (fn - 3.0f));
    return A - B;
}}

/*
 * Extract 15 statistics from signal x[n] into out[0..14].
 *
 * Output order (CANONICAL — matches Python & har_config.h feature list):
 *   [0]  mean
 *   [1]  std (population)
 *   [2]  min
 *   [3]  max
 *   [4]  range
 *   [5]  median
 *   [6]  q25
 *   [7]  q75
 *   [8]  iqr
 *   [9]  skewness  (pandas formula)
 *   [10] kurtosis  (pandas excess kurtosis)
 *   [11] rms
 *   [12] energy
 *   [13] zero_crossings
 *   [14] mean_crossing_rate
 */
static void _extract_15_stats(const float *x, int n, float *out) {{
    /* Sorted copy for percentiles — stack allocation */
    float sorted[{sorted_arr_size}];
    for (int i = 0; i < n; i++) sorted[i] = x[i];
    _sort(sorted, n);

    float mn  = _mean(x, n);
    float std = _std_pop(x, n, mn);
    float ss  = _std_sample(x, n, mn);   /* for skew/kurt only */

    float rms_val = 0.0f, energy = 0.0f;
    for (int i = 0; i < n; i++) {{ float v2 = x[i] * x[i]; rms_val += v2; energy += v2; }}
    rms_val = sqrtf(rms_val / (float)n);

    int zc = 0;
    for (int i = 1; i < n; i++) if (x[i - 1] * x[i] < 0.0f) zc++;

    int mc = 0;
    for (int i = 1; i < n; i++) {{
        float a = x[i - 1] - mn, b = x[i] - mn;
        if (a * b < 0.0f) mc++;
    }}

    float q25 = _percentile(sorted, n, 0.25f);
    float q75 = _percentile(sorted, n, 0.75f);
    float med = _percentile(sorted, n, 0.50f);

    out[0]  = mn;
    out[1]  = std;
    out[2]  = sorted[0];
    out[3]  = sorted[n - 1];
    out[4]  = sorted[n - 1] - sorted[0];
    out[5]  = med;
    out[6]  = q25;
    out[7]  = q75;
    out[8]  = q75 - q25;
    out[9]  = _skewness(x, n, mn, ss);
    out[10] = _kurtosis(x, n, mn, ss);
    out[11] = rms_val;
    out[12] = energy;
    out[13] = (float)zc;
    out[14] = (float)mc / (float)n;
}}

{preproc_functions}

/* ===========================================================
 * Main feature extraction entry point
 * =========================================================== */
void har_extract_features(
    const float window[HAR_WINDOW_SIZE][HAR_N_CHANNELS],
    float features[HAR_NUM_FEATURES]
) {{
    /* ---- Optional preprocessing ---- */
    float data[HAR_WINDOW_SIZE][HAR_N_CHANNELS];
    for (int i = 0; i < HAR_WINDOW_SIZE; i++)
        for (int ch = 0; ch < HAR_N_CHANNELS; ch++)
            data[i][ch] = window[i][ch];
{preproc_calls}

{extraction}
}}
"""

    # ------------------------------------------------------------------
    # Preprocessing code sections
    # ------------------------------------------------------------------

    def _preprocessing_functions(self) -> str:
        parts = []
        if self.preprocessing.get("low_pass_filter"):
            parts.append(self._iir_filtfilt_code())
        if self.preprocessing.get("savgol_filter"):
            parts.append(self._savgol_code())
        if self.preprocessing.get("kalman_filter"):
            parts.append(self._kalman_code())
        if self.preprocessing.get("fft_filter"):
            parts.append(self._fft_filter_code())
        return "\n".join(parts)

    def _preprocessing_calls(self) -> str:
        calls = []
        if self.preprocessing.get("low_pass_filter"):
            calls.append("    iir_filtfilt_window(data, HAR_WINDOW_SIZE, data);")
        if self.preprocessing.get("savgol_filter"):
            calls.append("    savgol_smooth_window(data, HAR_WINDOW_SIZE, data);")
        if self.preprocessing.get("kalman_filter"):
            calls.append("    kalman_filter_window(data, HAR_WINDOW_SIZE, data);")
        if self.preprocessing.get("fft_filter"):
            calls.append("    fft_lowpass_window(data, HAR_WINDOW_SIZE, data);")
        return "\n".join(calls) if calls else "    /* no preprocessing */"

    # ------------------------------------------------------------------
    # Orientation-invariant (magnitude-based) feature extraction
    # ------------------------------------------------------------------

    def _orientation_invariant_extraction(self) -> str:
        blocks = ["    int fi = 0;  /* feature output index */\n"]

        if self.has_acc_mag:
            blocks.append(self._mag_block("acc_mag", "0", "1", "2"))  # aX=ch0, aY=ch1, aZ=ch2
        if self.has_gyro_mag:
            blocks.append(self._mag_block("gyro_mag", "3", "4", "5"))  # gX=ch3, gY=ch4, gZ=ch5

        if self.has_acc_jerk:
            blocks.append(self._jerk_block("acc_jerk_mag", "0", "1", "2"))
        if self.has_gyro_jerk:
            blocks.append(self._jerk_block("gyro_jerk_mag", "3", "4", "5"))

        if self.has_sma:
            blocks.append(self._sma_block())
        if self.has_tilt:
            blocks.append(self._tilt_block())
        if self.has_autocorr:
            blocks.append(self._autocorr_block())
        if self.has_peak_count:
            blocks.append(self._peak_count_block())

        if self.has_freq:
            if self.has_acc_mag:
                blocks.append(self._freq_block("acc_freq", "0", "1", "2"))
            if self.has_gyro_mag:
                blocks.append(self._freq_block("gyro_freq", "3", "4", "5"))

        blocks.append("    (void)fi;  /* suppress unused-variable warning */")
        return "\n".join(blocks)

    def _mag_block(self, name: str, ch0: str, ch1: str, ch2: str) -> str:
        """Generate magnitude computation + 15-stat extraction for one signal."""
        n = self.window_size
        return f"""\
    /* ---- {name}: centered magnitude ---- */
    {{
        float mag[{n}];
        /* Compute per-window mean of each axis */
        float m0 = 0, m1 = 0, m2 = 0;
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) {{
            m0 += data[i][{ch0}]; m1 += data[i][{ch1}]; m2 += data[i][{ch2}];
        }}
        m0 /= HAR_WINDOW_SIZE; m1 /= HAR_WINDOW_SIZE; m2 /= HAR_WINDOW_SIZE;
        /* Centered magnitude: remove static offset, then compute Euclidean norm */
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) {{
            float a = data[i][{ch0}] - m0;
            float b = data[i][{ch1}] - m1;
            float c = data[i][{ch2}] - m2;
            mag[i] = sqrtf(a*a + b*b + c*c);
        }}
        _extract_15_stats(mag, HAR_WINDOW_SIZE, features + fi);
        fi += 15;
    }}
"""

    def _jerk_block(self, name: str, ch0: str, ch1: str, ch2: str) -> str:
        """Generate jerk magnitude + 3-stat (mean, std, max) extraction."""
        n = self.window_size
        return f"""\
    /* ---- {name}: jerk magnitude (mean, std, max) ---- */
    {{
        float jerk[{n - 1}];
        for (int i = 0; i < HAR_WINDOW_SIZE - 1; i++) {{
            float da = data[i+1][{ch0}] - data[i][{ch0}];
            float db = data[i+1][{ch1}] - data[i][{ch1}];
            float dc = data[i+1][{ch2}] - data[i][{ch2}];
            jerk[i] = sqrtf(da*da + db*db + dc*dc);
        }}
        int jn = HAR_WINDOW_SIZE - 1;
        float jmn = _mean(jerk, jn);
        features[fi++] = jmn;                        /* mean */
        features[fi++] = _std_pop(jerk, jn, jmn);   /* std  */
        /* max */
        float jmax = jerk[0];
        for (int i = 1; i < jn; i++) if (jerk[i] > jmax) jmax = jerk[i];
        features[fi++] = jmax;                       /* max  */
    }}
"""

    def _sma_block(self) -> str:
        return f"""\
    /* ---- acc_sma: Signal Magnitude Area = mean(|aX|+|aY|+|aZ|) ---- */
    {{
        float sma = 0.0f;
        for (int i = 0; i < HAR_WINDOW_SIZE; i++)
            sma += fabsf(data[i][0]) + fabsf(data[i][1]) + fabsf(data[i][2]);
        features[fi++] = sma / HAR_WINDOW_SIZE;
    }}
"""

    def _tilt_block(self) -> str:
        return f"""\
    /* ---- tilt_pitch, tilt_roll: mean-acceleration tilt angles ---- */
    {{
        float mx = 0, my = 0, mz = 0;
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) {{
            mx += data[i][0]; my += data[i][1]; mz += data[i][2];
        }}
        mx /= HAR_WINDOW_SIZE; my /= HAR_WINDOW_SIZE; mz /= HAR_WINDOW_SIZE;
        features[fi++] = atan2f(my, sqrtf(mx*mx + mz*mz));  /* pitch */
        features[fi++] = atan2f(-mx, sqrtf(my*my + mz*mz)); /* roll  */
    }}
"""

    def _autocorr_block(self) -> str:
        return f"""\
    /* ---- acc_mag_autocorr_lag1: lag-1 autocorrelation of acc magnitude ---- */
    {{
        /* Recompute centered acc magnitude to avoid storing it globally */
        float mag[{self.window_size}];
        float m0 = 0, m1 = 0, m2 = 0;
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) {{
            m0 += data[i][0]; m1 += data[i][1]; m2 += data[i][2];
        }}
        m0 /= HAR_WINDOW_SIZE; m1 /= HAR_WINDOW_SIZE; m2 /= HAR_WINDOW_SIZE;
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) {{
            float a = data[i][0]-m0, b = data[i][1]-m1, c = data[i][2]-m2;
            mag[i] = sqrtf(a*a+b*b+c*c);
        }}
        float mn = _mean(mag, HAR_WINDOW_SIZE);
        float var = 0.0f;
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) {{ float d = mag[i]-mn; var += d*d; }}
        float cov = 0.0f;
        for (int i = 0; i < HAR_WINDOW_SIZE-1; i++)
            cov += (mag[i]-mn) * (mag[i+1]-mn);
        features[fi++] = (var > 1e-12f) ? (cov / var) : 0.0f;
    }}
"""

    def _peak_count_block(self) -> str:
        return f"""\
    /* ---- acc_jerk_mag_peak_count: local maxima count in jerk magnitude ---- */
    {{
        float jerk[{self.window_size - 1}];
        for (int i = 0; i < HAR_WINDOW_SIZE - 1; i++) {{
            float da = data[i+1][0]-data[i][0], db = data[i+1][1]-data[i][1],
                  dc = data[i+1][2]-data[i][2];
            jerk[i] = sqrtf(da*da+db*db+dc*dc);
        }}
        int jn = HAR_WINDOW_SIZE - 1;
        float jmn = _mean(jerk, jn);
        int peaks = 0;
        for (int i = 1; i < jn-1; i++)
            if (jerk[i] > jerk[i-1] && jerk[i] > jerk[i+1] && jerk[i] > jmn)
                peaks++;
        features[fi++] = (float)peaks;
    }}
"""

    def _freq_block(self, name: str, ch0: str, ch1: str, ch2: str) -> str:
        """11 frequency-domain features from magnitude FFT (matches Python DFT path)."""
        n = self.window_size
        cutoff_hz = 20.0
        cutoff_bin = max(1, int(cutoff_hz * n / self.sampling_rate))
        mid_bin = max(1, cutoff_bin // 2)
        return f"""\
    /* ---- {name}: 11 frequency features from centered magnitude ---- */
    {{
        /* Recompute centered magnitude */
        float mag[{n}];
        float m0=0,m1=0,m2=0;
        for (int i=0;i<HAR_WINDOW_SIZE;i++) {{
            m0+=data[i][{ch0}]; m1+=data[i][{ch1}]; m2+=data[i][{ch2}];
        }}
        m0/=HAR_WINDOW_SIZE; m1/=HAR_WINDOW_SIZE; m2/=HAR_WINDOW_SIZE;
        for (int i=0;i<HAR_WINDOW_SIZE;i++) {{
            float a=data[i][{ch0}]-m0, b=data[i][{ch1}]-m1, c=data[i][{ch2}]-m2;
            mag[i]=sqrtf(a*a+b*b+c*c);
        }}

        /* DC-remove + Hann window (matches Python _fft_with_windowing) */
        float win_mean = _mean(mag, HAR_WINDOW_SIZE);
        float hann_mag[{n}];
        for (int i=0;i<HAR_WINDOW_SIZE;i++) {{
            float h = 0.5f*(1.0f - cosf(6.283185307f*(float)i/(float)(HAR_WINDOW_SIZE-1)));
            hann_mag[i] = (mag[i] - win_mean) * h;
        }}

        /* DFT magnitude for positive bins 1..N/2 */
        int nhalf = HAR_WINDOW_SIZE/2;
        /* Stack-allocate FFT scratch (max 200 bins) */
        float fft_mag[{min(n // 2 + 1, 101)}];
        int nbins = nhalf < {min(n // 2 + 1, 101)} ? nhalf : {min(n // 2 + 1, 101) - 1};
        for (int k=1;k<=nbins;k++) {{
            float re=0,im=0;
            float step=6.283185307f*(float)k/(float)HAR_WINDOW_SIZE;
            for (int i=0;i<HAR_WINDOW_SIZE;i++) {{
                re+=hann_mag[i]*cosf(step*(float)i);
                im-=hann_mag[i]*sinf(step*(float)i);
            }}
            fft_mag[k-1]=sqrtf(re*re+im*im);
        }}

        /* Derived frequency features */
        float dom_freq=0, dom_mag=0, spec_centroid=0, total_e=0;
        float e_low=0, e_mid=0, e_high=0;
        for (int k=0;k<nbins;k++) {{
            float freq=(float)(k+1)*(float)HAR_SAMPLE_RATE/(float)HAR_WINDOW_SIZE;
            float e=fft_mag[k]*fft_mag[k];
            total_e+=e;
            if (fft_mag[k]>dom_mag) {{ dom_mag=fft_mag[k]; dom_freq=freq; }}
            spec_centroid+=freq*e;
            if (k<{cutoff_bin // 3}) e_low+=e;
            else if (k<{mid_bin})    e_mid+=e;
            else                     e_high+=e;
        }}
        if (total_e>1e-12f) spec_centroid/=total_e;

        /* Spectral rolloff: freq where 85% of energy is below */
        float rolloff_freq=0, cum_e=0, threshold=0.85f*total_e;
        for (int k=0;k<nbins;k++) {{
            cum_e+=fft_mag[k]*fft_mag[k];
            if (cum_e>=threshold) {{
                rolloff_freq=(float)(k+1)*(float)HAR_SAMPLE_RATE/(float)HAR_WINDOW_SIZE;
                break;
            }}
        }}

        /* Spectral shape stats: rms, skewness, kurtosis, entropy */
        float spec_mn=0;
        if (nbins>0) {{ for(int k=0;k<nbins;k++) spec_mn+=fft_mag[k]; spec_mn/=(float)nbins; }}
        float spec_sq=0;
        for(int k=0;k<nbins;k++) spec_sq+=fft_mag[k]*fft_mag[k];
        float spec_rms=sqrtf(spec_sq/(float)nbins);
        float spec_var=spec_sq/(float)nbins - spec_mn*spec_mn;
        float spec_ss=sqrtf(spec_var>0?spec_var:0);
        float spec_skew=_skewness(fft_mag,nbins,spec_mn,spec_ss);
        float spec_kurt=_kurtosis(fft_mag,nbins,spec_mn,spec_ss);
        float spec_ent=0;
        if (total_e>1e-12f) {{
            for(int k=0;k<nbins;k++) {{
                float p=fft_mag[k]*fft_mag[k]/total_e;
                if(p>1e-12f) spec_ent-=p*logf(p);
            }}
        }}

        features[fi++] = dom_freq;
        features[fi++] = dom_mag;
        features[fi++] = spec_centroid;
        features[fi++] = e_low;
        features[fi++] = e_mid;
        features[fi++] = e_high;
        features[fi++] = rolloff_freq;
        features[fi++] = spec_rms;
        features[fi++] = spec_skew;
        features[fi++] = spec_kurt;
        features[fi++] = spec_ent;
    }}
"""

    # ------------------------------------------------------------------
    # Per-axis feature extraction (time_domain mode)
    # ------------------------------------------------------------------

    def _per_axis_extraction(self) -> str:
        n = self.window_size
        axes = [("aX", 0), ("aY", 1), ("aZ", 2), ("gX", 3), ("gY", 4), ("gZ", 5)]
        fset = set(self.feature_names)
        blocks = ["    int fi = 0;\n"]
        for axis_name, ch in axes:
            if any(f.startswith(f"{axis_name}_") for f in fset):
                blocks.append(f"""\
    /* ---- {axis_name} (channel {ch}): 15 stats ---- */
    {{
        float axis[{n}];
        for (int i = 0; i < HAR_WINDOW_SIZE; i++) axis[i] = data[i][{ch}];
        _extract_15_stats(axis, HAR_WINDOW_SIZE, features + fi);
        fi += 15;
    }}
""")
        blocks.append("    (void)fi;")
        return "\n".join(blocks)

    # ------------------------------------------------------------------
    # Preprocessing implementations (copy-pasted from base_generator but
    # now in a separate block so feature extraction is always clean)
    # ------------------------------------------------------------------

    def _iir_filtfilt_code(self) -> str:
        cutoff = self.preprocessing.get("lpf_cutoff_hz", 5)
        order = self.preprocessing.get("lpf_order", 2)
        try:
            from scipy.signal import butter
            b, a = butter(order, cutoff, btype="low", fs=self.sampling_rate)
        except Exception:
            return "/* WARNING: scipy unavailable, IIR filter not generated */\n"
        b_str = ", ".join(_cf(v) for v in b)
        a_str = ", ".join(_cf(v) for v in a)
        return f"""\
/* IIR filtfilt: Butterworth zero-phase, cutoff={cutoff}Hz, order={order}, fs={self.sampling_rate}Hz
 * Two-pass (forward+backward) matches Python scipy.signal.filtfilt exactly. */
#define _IIR_ORDER {order}
static const float _iir_b[{len(b)}] = {{{b_str}}};
static const float _iir_a[{len(a)}] = {{{a_str}}};

static void iir_filtfilt_window(float in[][HAR_N_CHANNELS], int n, float out[][HAR_N_CHANNELS]) {{
    for (int i=0;i<n;i++) for (int ch=0;ch<HAR_N_CHANNELS;ch++) out[i][ch]=in[i][ch];
    float xh[HAR_N_CHANNELS][_IIR_ORDER], yh[HAR_N_CHANNELS][_IIR_ORDER];
    /* Forward pass */
    for (int ch=0;ch<HAR_N_CHANNELS;ch++) for (int k=0;k<_IIR_ORDER;k++) {{xh[ch][k]=0;yh[ch][k]=0;}}
    for (int i=0;i<n;i++) for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        float x=out[i][ch], y=_iir_b[0]*x;
        for (int k=1;k<=_IIR_ORDER;k++) {{ y+=_iir_b[k]*xh[ch][k-1]; y-=_iir_a[k]*yh[ch][k-1]; }}
        for (int k=_IIR_ORDER-1;k>0;k--) {{ xh[ch][k]=xh[ch][k-1]; yh[ch][k]=yh[ch][k-1]; }}
        xh[ch][0]=x; yh[ch][0]=y; out[i][ch]=y;
    }}
    /* Reverse */
    for (int i=0;i<n/2;i++) for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        float tmp=out[i][ch]; out[i][ch]=out[n-1-i][ch]; out[n-1-i][ch]=tmp;
    }}
    /* Backward pass */
    for (int ch=0;ch<HAR_N_CHANNELS;ch++) for (int k=0;k<_IIR_ORDER;k++) {{xh[ch][k]=0;yh[ch][k]=0;}}
    for (int i=0;i<n;i++) for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        float x=out[i][ch], y=_iir_b[0]*x;
        for (int k=1;k<=_IIR_ORDER;k++) {{ y+=_iir_b[k]*xh[ch][k-1]; y-=_iir_a[k]*yh[ch][k-1]; }}
        for (int k=_IIR_ORDER-1;k>0;k--) {{ xh[ch][k]=xh[ch][k-1]; yh[ch][k]=yh[ch][k-1]; }}
        xh[ch][0]=x; yh[ch][0]=y; out[i][ch]=y;
    }}
    /* Reverse back */
    for (int i=0;i<n/2;i++) for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        float tmp=out[i][ch]; out[i][ch]=out[n-1-i][ch]; out[n-1-i][ch]=tmp;
    }}
}}
"""

    def _savgol_code(self) -> str:
        wl = int(self.preprocessing.get("savgol_window_length", 5))
        po = int(self.preprocessing.get("savgol_polyorder", 2))
        try:
            from scipy.signal import savgol_coeffs
            coeffs = savgol_coeffs(wl, po)
        except Exception:
            return "/* WARNING: scipy unavailable, SG filter not generated */\n"
        half = wl // 2
        coeffs_str = ", ".join(f"{c:.8f}f" for c in coeffs)
        return f"""\
/* Savitzky-Golay FIR smoothing, window={wl}, poly={po} */
#define _SG_HALF {half}
#define _SG_LEN  {wl}
static const float _sg_k[{wl}] = {{{coeffs_str}}};

static void savgol_smooth_window(float in[][HAR_N_CHANNELS], int n, float out[][HAR_N_CHANNELS]) {{
    for (int i=0;i<n;i++) for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        float acc=0;
        for (int k=0;k<_SG_LEN;k++) {{
            int s=i-_SG_HALF+k;
            if (s<0) s=0; if (s>=n) s=n-1;
            acc+=_sg_k[k]*in[s][ch];
        }}
        out[i][ch]=acc;
    }}
}}
"""

    def _kalman_code(self) -> str:
        q = self.preprocessing.get("kalman_process_noise", 1e-3)
        r = self.preprocessing.get("kalman_measurement_noise", 0.1)
        dt = 1.0 / self.sampling_rate
        q00 = q * (dt ** 3) / 3.0
        q01 = q * (dt ** 2) / 2.0
        q11 = q * dt
        # Pre-compute C literals so brace escaping inside f-string stays simple.
        kQ = "{{" + _cf(q00) + "," + _cf(q01) + "},{" + _cf(q01) + "," + _cf(q11) + "}}"
        kR = _cf(r)
        kDT = _cf(dt)
        return f"""\
/* Kalman filter (constant-velocity), Q={q}, R={r}, fs={self.sampling_rate}Hz */
static const float _kQ[2][2]={kQ};
static const float _kR={kR}, _kDT={kDT};

static void kalman_filter_window(float in[][HAR_N_CHANNELS], int n, float out[][HAR_N_CHANNELS]) {{
    for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        float x0=in[0][ch], x1=0;
        float P00=_kR,P01=0,P10=0,P11=_kR;
        for (int i=0;i<n;i++) {{
            float px0=x0+_kDT*x1, px1=x1;
            float pP00=P00+_kDT*(P10+P01)+_kDT*_kDT*P11+_kQ[0][0];
            float pP01=P01+_kDT*P11+_kQ[0][1];
            float pP10=P10+_kDT*P11+_kQ[1][0];
            float pP11=P11+_kQ[1][1];
            float S=pP00+_kR, K0=pP00/S, K1=pP10/S;
            float inn=in[i][ch]-px0;
            x0=px0+K0*inn; x1=px1+K1*inn;
            P00=(1-K0)*pP00; P01=(1-K0)*pP01;
            P10=pP10-K1*pP00; P11=pP11-K1*pP01;
            out[i][ch]=x0;
        }}
    }}
}}
"""

    def _fft_filter_code(self) -> str:
        import math
        cutoff_hz = float(self.preprocessing.get("fft_cutoff_hz", 10))
        n = self.window_size
        fs = self.sampling_rate
        cutoff_bin = max(1, math.ceil(cutoff_hz * n / fs))
        return f"""\
/* FFT brick-wall LPF, cutoff={cutoff_hz}Hz → bin {cutoff_bin}, N={n}, fs={fs}Hz */
#define _FFT_CBIN {cutoff_bin}
static float _fft_re[_FFT_CBIN], _fft_im[_FFT_CBIN];  /* static scratch */

static void fft_lowpass_window(float in[][HAR_N_CHANNELS], int n, float out[][HAR_N_CHANNELS]) {{
    float tpi_n=6.283185307f/(float)n, inv_n=1.0f/(float)n;
    for (int ch=0;ch<HAR_N_CHANNELS;ch++) {{
        for (int k=0;k<_FFT_CBIN;k++) {{
            float re=0,im=0,step=tpi_n*(float)k;
            for (int i=0;i<n;i++) {{ re+=in[i][ch]*cosf(step*(float)i); im-=in[i][ch]*sinf(step*(float)i); }}
            _fft_re[k]=re; _fft_im[k]=im;
        }}
        for (int i=0;i<n;i++) {{
            float val=_fft_re[0], step=tpi_n*(float)i;
            for (int k=1;k<_FFT_CBIN;k++) val+=2.0f*(_fft_re[k]*cosf((float)k*step)-_fft_im[k]*sinf((float)k*step));
            out[i][ch]=val*inv_n;
        }}
    }}
}}
"""
