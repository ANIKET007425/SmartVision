"""Downloads the small age model (genderage.onnx, about 1.3 MB) into the "models" folder.

Run it ONCE from the project folder (the one that has app.py):

    python get_age_model.py

It needs only internet (GitHub). It checks the file with a fingerprint (SHA-256), so a broken
or empty download is caught. Afterwards the models folder must contain genderage.onnx of
about 1,291 KB (1,322,532 bytes).

The model comes from the InsightFace project (pre-trained models, non-commercial research
use only). It is stored inside a big zip (buffalo_s.zip, about 120 MB) on GitHub. This
script reads ONLY the 1.3 MB it needs, and falls back to the whole zip if that does not work.
"""
import hashlib
import io
import os
import struct
import sys
import urllib.request
import zipfile
import zlib

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
NAME = "genderage.onnx"
SIZE = 1322532
SHA256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "models", NAME)


def get(url, start=None, end=None, tail=None, timeout=60):
    """Download (a part of) a URL. Redirects are followed."""
    request = urllib.request.Request(url, headers={"User-Agent": "SmartVision-model-download"})
    if tail is not None:
        request.add_header("Range", f"bytes=-{tail}")
    elif start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.status


def from_zip_parts():
    """Read only the part of the big zip that holds genderage.onnx."""
    tail, status = get(URL, tail=65536)
    if status != 206:
        raise RuntimeError("the server does not support partial downloads")
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("zip end marker not found")
    entries_total, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
    # total size of the zip = where the tail starts + tail length
    # the central directory (list of files) is right before the end marker
    cd_start_in_tail = len(tail) - (len(tail) - eocd) - cd_size
    if cd_start_in_tail < 0:
        raise RuntimeError("file list is not inside the downloaded end part")
    cd = tail[cd_start_in_tail:cd_start_in_tail + cd_size]
    position = 0
    while position + 46 <= len(cd) and cd[position:position + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[position + 10:position + 12])[0]
        comp_size, size = struct.unpack("<II", cd[position + 20:position + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", cd[position + 28:position + 34])
        local_offset = struct.unpack("<I", cd[position + 42:position + 46])[0]
        name = cd[position + 46:position + 46 + name_len].decode("utf-8", "replace")
        if name == NAME:
            header, _ = get(URL, start=local_offset, end=local_offset + 29)
            n, m = struct.unpack("<HH", header[26:30])
            data_start = local_offset + 30 + n + m
            blob, status = get(URL, start=data_start, end=data_start + comp_size - 1)
            if method == 0:
                return blob
            if method == 8:
                return zlib.decompress(blob, -15)
            raise RuntimeError(f"unknown zip method {method}")
        position += 46 + name_len + extra_len + comment_len
    raise RuntimeError(f"{NAME} not found in the zip")


def from_whole_zip():
    print("Downloading the whole zip (about 120 MB). This can take a few minutes ...")
    data, _ = get(URL, timeout=600)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(NAME)


def good(data):
    return data is not None and len(data) == SIZE and hashlib.sha256(data).hexdigest() == SHA256


def main():
    if os.path.exists(TARGET):
        with open(TARGET, "rb") as f:
            if good(f.read()):
                print("models/genderage.onnx is already there and correct. Nothing to do.")
                return 0
        print("models/genderage.onnx exists but is empty or broken. It will be replaced.")

    data = None
    try:
        print("Downloading genderage.onnx (about 1.3 MB) ...")
        data = from_zip_parts()
    except Exception as e:
        print("Small download did not work:", type(e).__name__, e)
    if not good(data):
        try:
            data = from_whole_zip()
        except Exception as e:
            print("Whole-zip download did not work:", type(e).__name__, e)
    if not good(data):
        print("\nFAILED. Please download the zip by hand and take genderage.onnx out of it:")
        print("  1) open  " + URL)
        print("  2) unzip it, copy genderage.onnx into the models folder")
        return 1

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "wb") as f:
        f.write(data)
    print(f"OK. Saved {len(data):,} bytes to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())