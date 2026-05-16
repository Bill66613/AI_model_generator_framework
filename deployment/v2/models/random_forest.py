"""RandomForest model block — generates har_model.h/.cpp for Random Forest."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

from .base import ModelBlock


class RandomForestModelBlock(ModelBlock):
    """
    Generates C++ tree traversal code for a RandomForest model.

    Tree data format in model_data['trees']:
      [{ 'feature_indices': [...], 'thresholds': [...],
         'left_children': [...], 'right_children': [...],
         'values': [[[ cls0_count, cls1_count, ... ]], ...] }, ...]

    Feature indices have already been reordered to C++ canonical order
    by the factory before the generator is constructed.
    """

    def __init__(self, model_data, feature_names, classes, precision):
        super().__init__(model_data, feature_names, classes, precision)
        self._trees = model_data.get("trees", [])

    def generate(self) -> Tuple[str, str]:
        n_trees = len(self._trees)
        return self._header(n_trees), self._impl(n_trees)

    # ------------------------------------------------------------------

    def _header(self, n_trees: int) -> str:
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_model.h — Random Forest ({n_trees} trees)
 *
 * Input:  features[HAR_NUM_FEATURES]  — ALREADY StandardScaler-normalized
 * Output: probabilities[HAR_NUM_CLASSES] — vote fractions (sum to 1.0)
 *
 * The caller (har_classifier.cpp) handles scaling and thresholding.
 * Do NOT scale features inside this function.
 */
void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
);
"""

    def _impl(self, n_trees: int) -> str:
        if not self._trees:
            return self._empty_impl()

        node_structs, tree_starts = self._build_node_arrays()
        n_nodes = sum(len(t) for t in node_structs)

        nodes_code = self._emit_node_array(node_structs, n_nodes)
        starts_code = "    " + ", ".join(str(s) for s in tree_starts)

        return f"""\
#include "har_model.h"
#include <math.h>

/* ---- Tree node structure ---- */
typedef struct {{
    int   feature_idx;   /* -2 = leaf */
    float threshold;
    int   left;
    int   right;
    int   leaf_class;    /* valid when feature_idx == -2 */
}} HARTreeNode;

#define HAR_RF_N_NODES  {n_nodes}
#define HAR_RF_N_TREES  {n_trees}

static const HARTreeNode har_rf_nodes[HAR_RF_N_NODES] = {{
{nodes_code}
}};

static const int har_rf_tree_start[HAR_RF_N_TREES] = {{
{starts_code}
}};

static int _rf_predict_tree(const float *f, int start) {{
    int idx = start;
    while (har_rf_nodes[idx].feature_idx != -2) {{
        int fi = har_rf_nodes[idx].feature_idx;
        if (fi >= 0 && fi < HAR_NUM_FEATURES && f[fi] <= har_rf_nodes[idx].threshold)
            idx = har_rf_nodes[idx].left;
        else
            idx = har_rf_nodes[idx].right;
    }}
    return har_rf_nodes[idx].leaf_class;
}}

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
    int votes[HAR_NUM_CLASSES] = {{0}};
    for (int t = 0; t < HAR_RF_N_TREES; t++) {{
        int cls = _rf_predict_tree(features, har_rf_tree_start[t]);
        if (cls >= 0 && cls < HAR_NUM_CLASSES) votes[cls]++;
    }}
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        probabilities[c] = (float)votes[c] / (float)HAR_RF_N_TREES;
}}
"""

    def _build_node_arrays(self):
        """Convert sklearn tree dicts to flat node arrays."""
        import numpy as np

        all_nodes = []
        tree_starts = []
        offset = 0

        for tree_data in self._trees:
            tree_starts.append(offset)
            feature_idx = tree_data.get("feature_indices", [])
            thresholds = tree_data.get("thresholds", [])
            left_ch = tree_data.get("left_children", [])
            right_ch = tree_data.get("right_children", [])
            values = tree_data.get("values", [])

            nodes = []
            for i in range(len(feature_idx)):
                fi = int(feature_idx[i])
                # sklearn uses -2 for leaf nodes (TREE_LEAF constant)
                is_leaf = fi < 0

                if is_leaf:
                    # Determine leaf class from value counts
                    if values and i < len(values):
                        v = values[i]
                        # values shape is [1, n_classes] per sklearn
                        if isinstance(v, (list, tuple)) and len(v) > 0:
                            counts = v[0] if isinstance(
                                v[0], (list, tuple)) else v
                            leaf_cls = int(np.argmax(counts))
                        else:
                            leaf_cls = 0
                    else:
                        leaf_cls = 0
                    nodes.append((-2, 0.0, 0, 0, leaf_cls))
                else:
                    thr = float(thresholds[i]) if thresholds else 0.0
                    lc = int(left_ch[i]) + offset if left_ch else 0
                    rc = int(right_ch[i]) + offset if right_ch else 0
                    nodes.append((fi, thr, lc, rc, -1))

            all_nodes.append(nodes)
            offset += len(nodes)

        return all_nodes, tree_starts

    def _emit_node_array(self, node_arrays, n_total: int) -> str:
        lines = []
        for nodes in node_arrays:
            for fi, thr, lc, rc, lcls in nodes:
                lines.append(
                    f"    {{{fi:5d}, {thr:.{self.precision}f}f, {lc:5d}, {rc:5d}, {lcls:3d}}}"
                )
        return ",\n".join(lines)

    def _empty_impl(self) -> str:
        n_cls = self.n_classes
        return f"""\
#include "har_model.h"
/* No tree data available — uniform prediction */
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]) {{
    (void)features;
    float p = 1.0f / (float)HAR_NUM_CLASSES;
    for (int i = 0; i < HAR_NUM_CLASSES; i++) probabilities[i] = p;
}}
"""
