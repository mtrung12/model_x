def parse_llm_output(raw_output: str):
    if not raw_output:
        return None
    raw_output = raw_output.strip()

    last_line = raw_output.splitlines()[-1].strip().lower()
    if "high" in last_line:
        return "high"
    if "low" in last_line:
        return "low"

    lowered = raw_output.lower()
    if "high" in lowered:
        return "high"
    if "low" in lowered:
        return "low"

    return None
