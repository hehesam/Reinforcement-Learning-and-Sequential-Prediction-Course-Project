import re
import json


def parse_log_qp(input_path: str, output_path: str) -> list[dict]:
    """
    Parse a training log file into a list of records and save as JSON.

    Expected line format:
        EP 50 eval_return -118.15 success 1.0 max|Q| 58.44

    Parameters
    ----------
    input_path  : path to the .txt log file
    output_path : path for the output .json file

    Returns
    -------
    List of dicts with keys: EP, eval_return, success
    """
    pattern = re.compile(
        r"EP\s+(?P<EP>\d+)"
        r".*?eval_return\s+(?P<eval_return>-?\d+(?:\.\d+)?)"
        r".*?success\s+(?P<success>-?\d+(?:\.\d+)?)"
    )

    records = []
    with open(input_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    "EP":          int(m.group("EP")),
                    "eval_return": float(m.group("eval_return")),
                    "success":     float(m.group("success")),
                })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Parsed {len(records)} records → {output_path}")
    return records


if __name__ == "__main__":
    # Example usage – adjust paths as needed
    parse_log_qp("q_learning_log2000.txt", "q_learning_log2000.json")
