"""Security Lab for SmartVision: four tools whose answers are EXACT.

Why this page exists
--------------------
The AI parts of SmartVision (objects, emotion) give estimates: they can be wrong.
These four tools are different. They are plain calculations, so the answer is either
exactly right or the tool says clearly that it could not do it:

  1. Secret message in a photo (steganography): hide a text in the smallest bit of the
     colour numbers of a PNG, and read it back. The read-back is checked before you
     download the picture.
  2. QR code reader: OpenCV reads a QR code. If it reads one, the text is exactly what is
     stored in the code. Links also get a few simple safety hints (hints, not a promise).
  3. File fingerprint (SHA-256): a file becomes a fixed 64-character fingerprint. Compare
     two files, or a file and a fingerprint, to see if anything was changed.
  4. Image facts: size, format, brightness, sharpness and the most used colours, all
     calculated from the pixels.

Nothing is saved on disk and nothing is sent anywhere: a picture or file is handled in
memory and thrown away. No new package is needed (Pillow, NumPy, OpenCV and the Python
standard library are already in the app).

The functions at the top do the work and do NOT need Streamlit, so the tests can run them
anywhere. The page itself (show_security_lab) is at the bottom of the file.
"""

import hashlib
import hmac
import io
import ipaddress
import math
import os
import re
import struct
import zlib
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps

# ======================================================================
# 1. Secret message in a photo
# ======================================================================
#
# The picture is a table of numbers (0-255) for red, green and blue of every pixel.
# Changing the LAST bit of a number changes the colour by at most 1 out of 255, which the
# eye cannot see. We write the bits of the message into those last bits, one after
# another. What is stored, in this order:
#
#     "SVMSG1"   6 bytes  marker: "this picture holds a SmartVision message"
#     mode       1 byte   0 = plain text, 1 = scrambled with a password
#     length     4 bytes  how many bytes the body has
#     body       N bytes  the message (or salt + check + scrambled message)
#     crc        4 bytes  CRC-32 of the body: a changed picture is noticed
#
# It only works with a lossless picture (PNG). A JPG changes pixels when it is saved,
# so the last bits are lost. That is why the result is always a PNG.

MAGIC = b"SVMSG1"
MODE_PLAIN, MODE_PASSWORD = 0, 1
HEADER_LEN = len(MAGIC) + 1 + 4          # marker + mode + length
CRC_LEN = 4
MAX_MESSAGE_BYTES = 100_000
MAX_PIXELS = 16_000_000                   # protects the free server's memory
HIDE_MAX_PIXELS = 8_000_000               # a bigger carrier picture is shrunk to this before hiding
PBKDF_ROUNDS = 100_000
SALT_LEN, TAG_LEN = 16, 8


class StegoError(ValueError):
    """Base class: the message could not be hidden or read."""


class NoMessageError(StegoError):
    """The picture does not hold a SmartVision message."""


class DamagedError(StegoError):
    """A message marker was found, but the picture was changed afterwards."""


class PasswordError(StegoError):
    """A password is needed, or the password is wrong."""


def load_rgb(data, upright=True):
    """Open picture bytes as an RGB uint8 array (height, width, 3).

    upright=True turns phone photos the right way up first (EXIF). Use False when
    READING a message, so the pixel order is exactly as stored.
    """
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:  # unreadable file, not a picture ...
        raise StegoError("This file could not be opened as a picture.") from e
    if img.width * img.height > MAX_PIXELS:      # checked BEFORE the pixels are loaded (memory)
        raise StegoError(
            f"The picture is too big ({img.width} x {img.height}). "
            f"Please use one with at most {MAX_PIXELS // 1_000_000} million pixels."
        )
    try:
        img.load()
    except Exception as e:  # damaged file, decompression bomb ...
        raise StegoError("This file could not be opened as a picture.") from e
    if upright:
        img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def fit_pixels(rgb, limit=None):
    """Shrink a picture that has more than `limit` pixels (keeps the shape). Returns (picture, was_shrunk).

    Only used when HIDING (the free server has little memory). A picture that holds a message is
    never resized when READING, because resizing destroys the message.
    """
    limit = limit or HIDE_MAX_PIXELS
    h, w = rgb.shape[:2]
    if h * w <= limit:
        return rgb, False
    k = math.sqrt(limit / float(h * w))
    new_w, new_h = max(1, int(w * k)), max(1, int(h * k))
    return np.ascontiguousarray(cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)), True


def _keystream(key, length):
    """Pseudo-random bytes from HMAC-SHA256 in counter mode."""
    blocks = []
    for counter in range((length + 31) // 32):
        blocks.append(hmac.new(key, struct.pack(">Q", counter), hashlib.sha256).digest())
    return b"".join(blocks)[:length]


def _xor(a, b):
    return (np.frombuffer(a, dtype=np.uint8) ^ np.frombuffer(b, dtype=np.uint8)).tobytes()


def _scramble(plain, password):
    salt = os.urandom(SALT_LEN)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF_ROUNDS)
    tag = hmac.new(key, b"tag" + plain, hashlib.sha256).digest()[:TAG_LEN]
    return salt + tag + _xor(plain, _keystream(key, len(plain)))


def _unscramble(body, password):
    if len(body) < SALT_LEN + TAG_LEN:
        raise DamagedError("The hidden message is damaged.")
    salt, tag, cipher = body[:SALT_LEN], body[SALT_LEN:SALT_LEN + TAG_LEN], body[SALT_LEN + TAG_LEN:]
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF_ROUNDS)
    plain = _xor(cipher, _keystream(key, len(cipher)))
    expected = hmac.new(key, b"tag" + plain, hashlib.sha256).digest()[:TAG_LEN]
    if not hmac.compare_digest(tag, expected):
        raise PasswordError("Wrong password.")
    return plain


def pack_payload(message, password=None):
    """The bytes that are hidden: marker, mode, length, body, CRC."""
    plain = message.encode("utf-8")
    if len(plain) > MAX_MESSAGE_BYTES:
        raise StegoError(f"The message is too long (at most {MAX_MESSAGE_BYTES:,} bytes).")
    if password:
        mode, body = MODE_PASSWORD, _scramble(plain, password)
    else:
        mode, body = MODE_PLAIN, plain
    return MAGIC + bytes([mode]) + struct.pack(">I", len(body)) + body + struct.pack(">I", zlib.crc32(body))


def capacity_bytes(shape):
    """How many bytes a picture of this shape can hold (1 bit per colour value)."""
    h, w = shape[:2]
    return (h * w * 3) // 8


def embed(rgb, payload):
    """Return a COPY of the picture with the payload written into the last bits."""
    flat = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(-1).copy()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if bits.size > flat.size:
        raise StegoError(
            f"The message does not fit in this picture: it needs {len(payload):,} bytes, "
            f"the picture can hold {flat.size // 8:,}. Use a bigger picture or a shorter message."
        )
    flat[:bits.size] = (flat[:bits.size] & 0xFE) | bits
    return flat.reshape(rgb.shape)


def extract(rgb):
    """Read the hidden bytes back. Returns (mode, body). Raises NoMessageError / DamagedError."""
    flat = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(-1)
    if flat.size < HEADER_LEN * 8:
        raise NoMessageError("This picture is too small to hold a message.")
    header = np.packbits(flat[:HEADER_LEN * 8] & 1).tobytes()
    if header[:len(MAGIC)] != MAGIC:
        raise NoMessageError("No hidden SmartVision message was found in this picture.")
    mode = header[len(MAGIC)]
    (length,) = struct.unpack(">I", header[len(MAGIC) + 1:HEADER_LEN])
    total_bits = (HEADER_LEN + length + CRC_LEN) * 8
    if mode not in (MODE_PLAIN, MODE_PASSWORD) or length > MAX_MESSAGE_BYTES + SALT_LEN + TAG_LEN or total_bits > flat.size:
        raise DamagedError("A message marker was found, but the picture was changed after the message was hidden.")
    data = np.packbits(flat[:total_bits] & 1).tobytes()
    body = data[HEADER_LEN:HEADER_LEN + length]
    (crc,) = struct.unpack(">I", data[HEADER_LEN + length:HEADER_LEN + length + CRC_LEN])
    if zlib.crc32(body) != crc:
        raise DamagedError(
            "A message marker was found, but the picture was changed after the message was hidden "
            "(for example saved as a JPG, resized or sent through an app that compresses photos)."
        )
    return mode, body


def reveal(rgb, password=None):
    """The hidden text of a picture. Raises StegoError (or a subclass) if there is none."""
    mode, body = extract(rgb)
    if mode == MODE_PASSWORD:
        if not password:
            raise PasswordError("This message is protected with a password. Type it below.")
        body = _unscramble(body, password)
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DamagedError("The hidden message is damaged.") from e


def png_bytes(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def hide_message(rgb, message, password=None):
    """Hide `message` in the picture and CHECK that it can be read back from the PNG.

    Returns a dict: png (bytes), width, height, capacity, payload, used_percent,
    max_change (biggest change of any colour value, always 0 or 1), changed_values,
    total_values, verified (True: the PNG was opened again and gave back the same text).
    """
    payload = pack_payload(message, password or None)
    stego = embed(rgb, payload)
    data = png_bytes(stego)
    try:
        verified = reveal(load_rgb(data, upright=False), password or None) == message
    except StegoError:
        verified = False
    # Only the last bit of a value can differ, so a changed value differs by exactly 1.
    changed = np.bitwise_xor(stego, rgb)
    capacity = capacity_bytes(rgb.shape)
    return {
        "png": data,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "capacity": capacity,
        "payload": len(payload),
        "used_percent": 100.0 * len(payload) / capacity if capacity else 100.0,
        "max_change": int(changed.max()) if changed.size else 0,
        "changed_values": int(np.count_nonzero(changed)),
        "total_values": int(changed.size),
        "verified": bool(verified),
    }


# ======================================================================
# 2. QR code reader
# ======================================================================

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly",
    "buff.ly", "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly", "lnkd.in",
}
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "password", "bank", "wallet",
    "payment", "prize", "free", "gift", "claim", "urgent", "suspend", "otp", "kyc", "lottery",
]
QR_MAX_SIDE = 1600


def _qr_variants(img_bgr):
    """The same picture in a few forms, because a QR code is easier to read in some of them."""
    h, w = img_bgr.shape[:2]
    big = max(h, w)
    base = img_bgr
    if big > QR_MAX_SIDE:
        k = QR_MAX_SIDE / float(big)
        base = cv2.resize(img_bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    yield base
    yield gray
    yield cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)   # quiet zone
    if max(gray.shape) < 900:
        yield cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


def decode_qr(img_bgr):
    """All QR codes OpenCV can read in the picture: a list of texts (can be empty).

    An empty list means "no QR code could be read", never a wrong text.
    """
    found = []
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return found
    for variant in _qr_variants(img_bgr):
        detector = cv2.QRCodeDetector()
        try:
            ok, texts, _points, _ = detector.detectAndDecodeMulti(variant)
            candidates = list(texts) if ok and texts is not None else []
        except Exception:
            candidates = []
        if not candidates:
            try:
                single, _points, _ = detector.detectAndDecode(variant)
                candidates = [single] if single else []
            except Exception:
                candidates = []
        for text in candidates:
            if text and text not in found:
                found.append(text)
        if found:
            break
    return found


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return host.isdigit()          # for example http://3232235777


def link_checks(text):
    """What kind of thing is in the QR code, plus a few simple hints.

    Returns (kind, hints). Each hint is (level, text) with level "ok", "info" or "warn".
    These are simple rules, not a guarantee: no hint does NOT mean the link is safe.
    """
    value = (text or "").strip()
    lower = value.lower()
    hints = []

    if lower.startswith(("upi:", "upi://")):
        return "UPI payment request", [("warn", "This code asks you to PAY. Check the receiver and the amount before you pay anything.")]
    if lower.startswith("wifi:"):
        return "Wi-Fi login", [("info", "Opening this joins a Wi-Fi network. Only join networks you trust.")]
    if lower.startswith(("mailto:", "matmsg:")):
        return "E-mail address", [("info", "It prepares an e-mail. Nothing is sent without you.")]
    if lower.startswith(("tel:", "sms:", "smsto:")):
        return "Phone number or SMS", [("info", "It prepares a call or an SMS. Check the number before you send.")]
    if lower.startswith("geo:"):
        return "Map location", [("info", "A place on the map.")]
    if lower.startswith("begin:vcard"):
        return "Contact card", [("info", "A contact (name, number ...) that can be added to your phone.")]

    looks_like_web = lower.startswith(("http://", "https://")) or bool(re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(/|:|\?|$)", lower))
    if not looks_like_web:
        return "Plain text", [("info", "It is just text, not a link.")]

    to_parse = value if "://" in value else "http://" + value
    try:
        parsed = urlparse(to_parse)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return "Web link", [("warn", "This web address is written in a strange way. Do not open it.")]

    if lower.startswith("https://"):
        hints.append(("ok", "Starts with https, so the connection is encrypted. (This does not prove the site is honest.)"))
    elif lower.startswith("http://"):
        hints.append(("warn", "Starts with http, not https: the connection is NOT encrypted."))
    else:
        hints.append(("warn", "The address has no http:// or https:// in front of it."))
    if host and _is_ip(host):
        hints.append(("warn", "The address is a number (an IP address), not a normal website name."))
    if "@" in parsed.netloc:
        hints.append(("warn", "It has an @ sign: the real website is the part AFTER the @."))
    if "xn--" in host:
        hints.append(("warn", "The website name uses special look-alike letters (xn--)."))
    if host in SHORTENERS or any(host.endswith("." + s) for s in SHORTENERS):
        hints.append(("warn", "A link shortener: you cannot see where it really goes."))
    if host.count(".") >= 4:
        hints.append(("info", "The website name has many parts (sub-domains). Check the LAST two parts before the first slash."))
    if port not in (None, 80, 443):
        hints.append(("info", f"It uses an unusual port ({port})."))
    if len(value) > 100:
        hints.append(("info", "The link is very long."))
    words = [w for w in SUSPICIOUS_WORDS if w in (host + parsed.path + parsed.query).lower()]
    if words:
        hints.append(("info", "Words often used in scam links: " + ", ".join(words) + "."))
    if not any(level == "warn" for level, _ in hints) and not words:
        hints.append(("info", "These simple rules found no warning sign. That is NOT a promise that the link is safe."))
    return "Web link", hints


# ======================================================================
# 3. File fingerprint (SHA-256)
# ======================================================================

MAX_FILE_BYTES = 100 * 1024 * 1024


def sha256_hex(data):
    """The SHA-256 fingerprint (64 hex characters) of some bytes."""
    return hashlib.sha256(data).hexdigest()


def normalize_fingerprint(text):
    """A pasted fingerprint as 64 lower-case hex characters, or None if it is not one."""
    value = re.sub(r"\s+", "", text or "").lower()
    if value.startswith("sha256:"):
        value = value[len("sha256:"):]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def same_fingerprint(a, b):
    """True if two fingerprints are equal (compared in constant time)."""
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))


# ======================================================================
# 4. Image facts
# ======================================================================

def _ratio_text(w, h):
    g = math.gcd(w, h) or 1
    a, b = w // g, h // g
    if max(a, b) <= 50:
        return f"{a}:{b}"
    return f"{w / h:.2f}:1" if w >= h else f"1:{h / w:.2f}"


def brightness_label(mean):
    if mean < 60:
        return "dark"
    if mean < 110:
        return "a bit dark"
    if mean <= 170:
        return "normal"
    if mean <= 210:
        return "bright"
    return "very bright"


def sharpness_label(score):
    if score < 50:
        return "very smooth or blurry"
    if score < 150:
        return "soft"
    if score < 500:
        return "sharp"
    return "very sharp or very detailed"


def dominant_colours(rgb, top=5):
    """The most used colours. All pixels are counted, in groups of similar colours
    (8 steps per channel = 512 groups). Returns [(hex, percent), ...], biggest first;
    the colour shown is the exact average of the pixels in the group."""
    flat = rgb.reshape(-1, 3)
    counts = np.zeros(512, dtype=np.int64)
    sums = np.zeros((512, 3), dtype=np.float64)
    step = 1_000_000
    for start in range(0, len(flat), step):
        chunk = flat[start:start + step]
        key = ((chunk[:, 0] >> 5).astype(np.int32) << 6) | ((chunk[:, 1] >> 5).astype(np.int32) << 3) | (chunk[:, 2] >> 5).astype(np.int32)
        counts += np.bincount(key, minlength=512)
        for channel in range(3):
            sums[:, channel] += np.bincount(key, weights=chunk[:, channel], minlength=512)
    total = counts.sum()
    result = []
    for index in np.argsort(counts)[::-1][:top]:
        if counts[index] == 0:
            break
        r, g, b = (int(round(v)) for v in sums[index] / counts[index])
        result.append((f"#{r:02x}{g:02x}{b:02x}", 100.0 * counts[index] / total))
    return result


def image_facts(data):
    """Numbers about a picture, all calculated from its pixels."""
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > 40_000_000:
            raise ValueError("This picture is too big to measure here (over 40 million pixels).")
        img.load()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("This file could not be opened as a picture.") from e
    fmt = img.format or "unknown"
    mode = img.mode
    frames = getattr(img, "n_frames", 1)
    has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    shown = ImageOps.exif_transpose(img).convert("RGB")
    w, h = shown.size
    rgb = np.asarray(shown, dtype=np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    small = gray
    if max(gray.shape) > 512:
        k = 512.0 / max(gray.shape)
        small = cv2.resize(gray, (max(1, int(gray.shape[1] * k)), max(1, int(gray.shape[0] * k))), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())

    return {
        "width": w,
        "height": h,
        "megapixels": w * h / 1_000_000.0,
        "ratio": _ratio_text(w, h),
        "format": fmt,
        "mode": mode,
        "frames": frames,
        "has_alpha": has_alpha,
        "file_bytes": len(data),
        "brightness": brightness,
        "brightness_label": brightness_label(brightness),
        "contrast": contrast,
        "sharpness": sharpness,
        "sharpness_label": sharpness_label(sharpness),
        "colours": dominant_colours(rgb),
    }


# ======================================================================
# The page (needs Streamlit; everything above does not)
# ======================================================================

PICTURE_TYPES = ["png", "jpg", "jpeg", "jfif", "webp", "bmp"]


def _human_bytes(n):
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024.0


def _show_hints(st, hints):
    for level, text in hints:
        if level == "warn":
            st.warning(text)
        elif level == "ok":
            st.success(text)
        else:
            st.info(text)


def _stego_tab(st):
    st.write(
        "Hide a short text inside a picture, or read a text out of a picture. "
        "To the eye the picture looks the same. The result is a **PNG**, because PNG keeps every pixel exactly."
    )
    action = st.radio("What do you want to do?", ["🤫 Hide a message", "🔎 Reveal a message"], horizontal=True, key="sl_stego_action")

    if action.startswith("🤫"):
        uploaded = st.file_uploader("1. Pick the picture that will carry the message", type=PICTURE_TYPES, key="sl_hide_file")
        message = st.text_area("2. Type the secret message", max_chars=5000, key="sl_hide_text")
        password = st.text_input(
            "3. Password (optional)", type="password", key="sl_hide_password",
            help="With a password the message is scrambled, so only someone who knows the password can read it.",
        )
        if uploaded is None:
            st.info("Pick a picture to start. A bigger picture can hold a longer message.")
            return
        data = uploaded.getvalue()
        try:
            rgb = load_rgb(data)
        except StegoError as e:
            st.error(str(e))
            return
        rgb, shrunk = fit_pixels(rgb)
        if shrunk:
            st.info(
                f"This picture is very big, so it is shrunk to {rgb.shape[1]} x {rgb.shape[0]} pixels before the message "
                "is hidden (the free server has little memory)."
            )
        capacity = capacity_bytes(rgb.shape)
        st.caption(f"This picture can hold about {max(0, capacity - HEADER_LEN - CRC_LEN - SALT_LEN - TAG_LEN):,} bytes of message "
                   "(one letter is 1 byte, a Hindi letter or an emoji is 3-4 bytes).")

        signature = hashlib.sha256(data + b"\0" + message.encode("utf-8") + b"\0" + password.encode("utf-8")).hexdigest()
        if st.button("🔒 Hide the message", key="sl_hide_button", disabled=not message.strip()):
            try:
                with st.spinner("Hiding the message..."):
                    info = hide_message(rgb, message, password or None)
                st.session_state["sl_stego"] = {"signature": signature, "info": info, "name": uploaded.name or "picture"}
            except StegoError as e:
                st.session_state.pop("sl_stego", None)
                st.error(str(e))
                return
        saved = st.session_state.get("sl_stego")
        if not saved:
            return
        if saved["signature"] != signature:
            st.info("The picture, the message or the password changed. Press **Hide the message** again.")
            return

        info = saved["info"]
        if info["verified"]:
            st.success("Done. The new PNG was opened again and the message was read back exactly. ✅")
        else:
            st.error("The message could not be read back from the new picture, so do not use it.")
            return
        st.image(info["png"], caption="Picture with the hidden message (it looks the same)", width="stretch")
        stem = (saved["name"] or "picture").rsplit(".", 1)[0]
        st.download_button(
            "⬇️ Download the PNG with the hidden message",
            data=info["png"], file_name=f"{stem}_secret.png", mime="image/png", key="sl_hide_download",
        )
        st.caption(
            f"Used {info['payload']:,} of {info['capacity']:,} bytes ({info['used_percent']:.2f}%). "
            f"{info['changed_values']:,} of {info['total_values']:,} colour numbers changed, "
            f"by at most {info['max_change']} out of 255."
        )
        st.warning(
            "Send it as a **file or document**. If you send it as a photo through WhatsApp, Instagram or similar apps, "
            "they compress it to a JPG and the message is destroyed."
        )
        st.caption(
            "Without a password this is only hiding, not locking: anyone who knows how can read it. "
            "The password scramble is a simple one for a project; do not use it for real secrets."
        )
    else:
        uploaded = st.file_uploader("Pick the picture that may hold a message", type=PICTURE_TYPES, key="sl_reveal_file")
        password = st.text_input("Password (only if the message has one)", type="password", key="sl_reveal_password")
        if uploaded is None:
            st.info("Pick the PNG you received. A JPG usually cannot hold a message any more.")
            return
        try:
            rgb = load_rgb(uploaded.getvalue(), upright=False)
            text = reveal(rgb, password or None)
        except PasswordError as e:
            st.warning(str(e))
            return
        except DamagedError as e:
            st.error(str(e))
            return
        except NoMessageError as e:
            st.info(str(e))
            return
        except StegoError as e:
            st.error(str(e))
            return
        st.success("Message found. ✅")
        st.code(text, language=None)
        st.caption(f"{len(text):,} characters. Read exactly as it was hidden.")


def _qr_tab(st):
    st.write(
        "Read a QR code from a picture. If a code is read, the text or link is **exactly** what is stored in the code. "
        "The link is never opened by the app."
    )
    uploaded = st.file_uploader("Upload a picture with a QR code", type=PICTURE_TYPES, key="sl_qr_file")
    use_camera = st.checkbox("Use the camera instead", key="sl_qr_camera_on")
    camera = st.camera_input("Show the QR code to the camera", key="sl_qr_camera") if use_camera else None
    source = camera if camera is not None else uploaded
    if source is None:
        st.info("Upload a picture, or switch on the camera. The whole code must be visible and not bent.")
        return
    try:
        rgb = load_rgb(source.getvalue())
    except StegoError as e:
        st.error(str(e))
        return
    codes = decode_qr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not codes:
        st.warning(
            "No QR code could be read. Try a sharper picture, come closer, keep the code flat and make sure "
            "the whole code (with a white border) is inside the picture."
        )
        return
    st.success(f"{len(codes)} QR code{'s' if len(codes) != 1 else ''} read.")
    for number, text in enumerate(codes, start=1):
        kind, hints = link_checks(text)
        st.markdown(f"**Code {number}: {kind}**")
        st.code(text, language=None)
        _show_hints(st, hints)
    st.caption("Never open a link or pay a request you did not expect, even if it looks fine.")


def _hash_tab(st):
    st.write(
        "A fingerprint (SHA-256) is a 64-character code made from the **whole content** of a file. The same file always gives "
        "the same fingerprint. If even one pixel or one letter changes, the fingerprint changes completely."
    )
    first = st.file_uploader("1. Pick a file (any kind)", key="sl_hash_a")
    if first is None:
        st.info("Pick a file to see its fingerprint.")
    else:
        if first.size > MAX_FILE_BYTES:
            st.error(f"This file is too big (at most {MAX_FILE_BYTES // (1024 * 1024)} MB).")
        else:
            data_a = first.getvalue()
            fingerprint_a = sha256_hex(data_a)
            st.markdown(f"**{first.name}**  ·  {_human_bytes(len(data_a))}")
            st.code(fingerprint_a, language=None)

            st.markdown("**2. Compare it** (pick one of the two ways)")
            second = st.file_uploader("with another file", key="sl_hash_b")
            pasted = st.text_input("or with a fingerprint someone gave you", key="sl_hash_paste")
            if second is not None:
                if second.size > MAX_FILE_BYTES:
                    st.error(f"The second file is too big (at most {MAX_FILE_BYTES // (1024 * 1024)} MB).")
                else:
                    fingerprint_b = sha256_hex(second.getvalue())
                    st.code(fingerprint_b, language=None)
                    if same_fingerprint(fingerprint_a, fingerprint_b):
                        st.success("✅ IDENTICAL. Both files have exactly the same content.")
                    else:
                        st.error("❌ DIFFERENT. The two files are not the same, even if they look the same.")
            elif pasted.strip():
                wanted = normalize_fingerprint(pasted)
                if wanted is None:
                    st.warning("A SHA-256 fingerprint has exactly 64 characters: the digits 0-9 and the letters a-f.")
                elif same_fingerprint(fingerprint_a, wanted):
                    st.success("✅ MATCH. This file is exactly the file the fingerprint was made from.")
                else:
                    st.error("❌ NO MATCH. This file is not the file the fingerprint was made from. It may have been changed.")
            st.caption(
                "A picture saved again (even as the same format), edited or stripped of its metadata gets a different fingerprint. "
                "Fingerprints do not tell WHO made a file."
            )

    with st.expander("Try it: the fingerprint of a text"):
        text = st.text_area("Type or paste some text, then change ONE letter and watch the fingerprint", value="SmartVision", key="sl_hash_text")
        st.code(sha256_hex(text.encode("utf-8")), language=None)


def _facts_tab(st):
    st.write("Numbers about a picture. They are calculated from the pixels, not guessed.")
    uploaded = st.file_uploader("Upload a picture", type=PICTURE_TYPES, key="sl_facts_file")
    if uploaded is None:
        st.info("Upload a picture to see its facts.")
        return
    try:
        facts = image_facts(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"The picture could not be measured: {type(e).__name__}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Size", f"{facts['width']} × {facts['height']} px")
    c2.metric("Megapixels", f"{facts['megapixels']:.2f} MP")
    c3.metric("Shape (ratio)", facts["ratio"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Format", facts["format"])
    c2.metric("File size", _human_bytes(facts["file_bytes"]))
    c3.metric("Colour mode", facts["mode"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Brightness (0-255)", f"{facts['brightness']:.1f}")
    c1.caption(facts["brightness_label"])
    c2.metric("Contrast", f"{facts['contrast']:.1f}")
    c3.metric("Sharpness score", f"{facts['sharpness']:.0f}")
    c3.caption(facts["sharpness_label"])

    if facts["has_alpha"]:
        st.caption("This picture has transparent parts. They are measured as if they were on a normal background.")
    if facts["frames"] > 1:
        st.caption(f"This file has {facts['frames']} frames (an animation). Only the first frame is measured.")

    st.markdown("**Most used colours**")
    for hex_code, percent in facts["colours"]:
        st.markdown(
            f'<span style="display:inline-block;width:1.4em;height:1.4em;border-radius:.35em;background:{hex_code};'
            f'border:1px solid rgba(255,255,255,.45);vertical-align:middle;margin-right:.6em"></span>'
            f"`{hex_code}` · {percent:.1f}% of the pixels",
            unsafe_allow_html=True,
        )
    st.caption(
        "Every number is calculated from the pixels. The words (dark, bright, sharp ...) are only a rough guide. "
        "Colours are counted in 512 groups of similar colours; the swatch shows the exact average of a group."
    )


def show_security_lab():
    """Draw the whole page."""
    import streamlit as st

    st.header("🛡️ Security Lab")
    st.caption(
        "Four tools with EXACT answers: they are calculations, not AI guesses. "
        "Your picture or file is handled in memory and is not saved by the app."
    )
    tab_hide, tab_qr, tab_hash, tab_facts = st.tabs(
        ["🤫 Secret message", "📷 QR code reader", "🔑 File fingerprint", "📐 Image facts"]
    )
    with tab_hide:
        _stego_tab(st)
    with tab_qr:
        _qr_tab(st)
    with tab_hash:
        _hash_tab(st)
    with tab_facts:
        _facts_tab(st)