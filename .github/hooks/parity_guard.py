"""
Pre-tool-use hook: Parity guard for code generator files.
Reads tool invocation from stdin, warns if editing parity-critical files.
"""
import sys
import json

PARITY_CRITICAL_FILES = [
    "utils/feature_extraction.py",
    "deployment/base_generator.py",
    "deployment/micropython_generator.py",
    "deployment/code_generator_factory.py",
    "deployment/neural_network_generator.py",
    "deployment/random_forest_generator.py",
    "deployment/svm_generator.py",
    "deployment/cnn_generator.py",
]

def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        # No input or invalid JSON — allow
        json.dump({"continue": True}, sys.stdout)
        return

    # Check if this is a file edit operation
    tool_name = input_data.get("toolName", "")
    tool_input = input_data.get("toolInput", {})
    file_path = tool_input.get("filePath", "") or tool_input.get("path", "")

    if not file_path:
        json.dump({"continue": True}, sys.stdout)
        return

    # Normalize path separators
    normalized = file_path.replace("\\", "/")

    for pf in PARITY_CRITICAL_FILES:
        if normalized.endswith(pf):
            msg = (
                f"⚠️ PARITY-CRITICAL FILE: {pf}\n"
                f"Any formula changes must be mirrored between Python ↔ C++ ↔ MicroPython.\n"
                f"Check TECHNICAL_FINDINGS.md for known parity issues."
            )
            json.dump({
                "continue": True,
                "systemMessage": msg
            }, sys.stdout)
            return

    json.dump({"continue": True}, sys.stdout)

if __name__ == "__main__":
    main()
