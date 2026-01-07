import argparse
import json
import os
from urllib.request import urlopen, urlretrieve


def fetch_record(record_id):
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urlopen(api_url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Download CFRP wavefield datasets from Zenodo.")
    parser.add_argument("--record", required=True, help="Zenodo record ID")
    parser.add_argument("--out", default="data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    record = fetch_record(args.record)

    files = record.get("files", [])
    if not files:
        raise RuntimeError("No files found in the Zenodo record.")

    for file_info in files:
        filename = file_info.get("key")
        url = file_info.get("links", {}).get("self")
        if not filename or not url:
            continue
        out_path = os.path.join(args.out, filename)
        print(f"Downloading {filename} -> {out_path}")
        urlretrieve(url, out_path)


if __name__ == "__main__":
    main()
