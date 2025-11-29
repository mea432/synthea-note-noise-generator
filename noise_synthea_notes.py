import json
import base64
import time
import random
import argparse
from pathlib import Path
from openai import OpenAI

client = OpenAI()


MODEL = "gpt-5-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 1024
DELAY = 0.35
RETRY_DELAYS = [1, 2, 5, 10]
# ----------------------------------------

CLINICIAN_STYLES = [
    "Short, blunt clinician with lots of abbreviations and light typos.",
    "Verbose clinician who repeats details and rambles a bit.",
    "Dictation-style note with odd punctuation and run-on sentences.",
    "Nursing-style succinct bullet-like lines with minimal punctuation.",
    "Uncertain clinician who uses phrases like 'likely', '?viral', and mid-sentence restarts.",
]


def build_prompt(original_text, profile):
    return f"""
Rewrite the clinical note below to sound more human and imperfect, while keeping all medical facts.
Use the style described here: {profile}
Introduce mild typos, shorthand, pauses, and messy structure, but do NOT change diagnoses, medications, or objective facts. Only use ASCII characters.

Original note:
** start **
{original_text}
** end **

Return only the rewritten note.
"""


LOG_FILE = "note_rewrite_log.txt"


def call_openai(prompt):
    for i, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                # max_output_tokens=MAX_TOKENS,
            )
            rewritten = response.output_text.strip()

            if len(rewritten) <= 5:
                print("LLM output: ", rewritten)
                raise ValueError("LLM responded with an empty output.")

            # Log the input and output
            with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                log_f.write("\n" + "=" * 80 + "\n")
                log_f.write("PROMPT:\n")
                log_f.write(prompt + "\n")
                log_f.write("-" * 80 + "\n")
                log_f.write("OUTPUT:\n")
                log_f.write(rewritten + "\n")
                log_f.write("=" * 80 + "\n")

            return rewritten

        except Exception as e:
            print(f"[Retry {i}] OpenAI error: {e}")

    raise RuntimeError("OpenAI API call failed after retries.")


def rewrite_base64_note(b64_text):
    try:
        decoded = base64.b64decode(b64_text).decode("utf-8")
    except Exception:
        print("Warning: could not decode Base64 note, leaving unchanged.")
        return b64_text

    style = random.choice(CLINICIAN_STYLES)
    prompt = build_prompt(decoded, style)
    new_text = call_openai(prompt)

    # if len(new_text) < max(20, len(decoded) * 0.5):
    #     print("Warning: rewritten note extremely short; keeping original.")
    #     return b64_text

    return base64.b64encode(new_text.encode("utf-8")).decode("ascii")


def process_resource(resource):
    count = 0
    if "presentedForm" not in resource:
        return 0

    pf_list = resource.get("presentedForm", [])
    for pf in pf_list:
        if "data" in pf:
            original = pf["data"]
            pf["data"] = rewrite_base64_note(original)
            count += 1

    return count


def process_file(path, out_dir):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rewritten_count = 0

    if "entry" in data:
        for entry in data["entry"]:
            resource = entry.get("resource")
            if resource:
                rewritten_count += process_resource(resource)
                print("Processed note ", rewritten_count)
                # if rewritten_count >= 5:
                #     break

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{path.name}: rewritten {rewritten_count} notes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    for file in in_dir.glob("*.json"):
        process_file(file, out_dir)
        time.sleep(DELAY)


if __name__ == "__main__":
    main()
