"""
Post-tool-use hook: Reminds about testing and sync after file edits.
"""
import sys
import json

def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        json.dump({"continue": True}, sys.stdout)
        return

    tool_input = input_data.get("toolInput", {})
    file_path = tool_input.get("filePath", "") or tool_input.get("path", "")

    if not file_path:
        json.dump({"continue": True}, sys.stdout)
        return

    normalized = file_path.replace("\\", "/")
    messages = []

    if "deployment/" in normalized:
        if "base_generator" in normalized:
            messages.append("Check micropython_generator.py for sync.")
        messages.append("Run: python -m pytest tests/ -v")

    if "callbacks/" in normalized:
        messages.append("Test UI: python app.py → http://127.0.0.1:8050")

    if "academic-paper" in normalized:
        messages.append("Compile: cd academic-paper-vietnamese && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")

    if messages:
        json.dump({
            "continue": True,
            "systemMessage": "📋 Reminders:\n" + "\n".join(f"  • {m}" for m in messages)
        }, sys.stdout)
    else:
        json.dump({"continue": True}, sys.stdout)

if __name__ == "__main__":
    main()
