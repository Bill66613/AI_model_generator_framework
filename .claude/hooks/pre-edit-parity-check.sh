#!/bin/bash
# Pre-edit parity check hook
# Warns when editing files that are part of the training-deployment parity chain

FILE="$1"

PARITY_FILES=(
  "utils/feature_extraction.py"
  "deployment/base_generator.py"
  "deployment/micropython_generator.py"
  "deployment/code_generator_factory.py"
  "deployment/neural_network_generator.py"
  "deployment/random_forest_generator.py"
  "deployment/svm_generator.py"
  "deployment/cnn_generator.py"
)

for pf in "${PARITY_FILES[@]}"; do
  if [[ "$FILE" == *"$pf"* ]]; then
    echo "⚠️  PARITY-CRITICAL FILE: $pf"
    echo "   Any formula changes must be mirrored between Python and C++."
    echo "   Check: utils/feature_extraction.py ↔ deployment/base_generator.py"
    echo "   Check: deployment/base_generator.py ↔ deployment/micropython_generator.py"
    echo "   See: academic-paper-vietnamese/TECHNICAL_FINDINGS.md"
    exit 0
  fi
done

exit 0
