"""The "Privacy Tools" page of SmartVision.

app.py shows this page when the visitor picks "🔒 Privacy Tools" in the mode list:

    elif "Privacy Tools" in mode:
        from privacy_tools import show_privacy_tools
        show_privacy_tools()

Two small tools, both for sharing photos more safely:

  1. Photo metadata checker - shows the hidden details inside a photo file
     (GPS location, phone model, time, serial number, names) and gives back a
     clean copy without them.
  2. Blur faces - finds every face with the same face finder SmartVision
     already uses (DeepFace + MTCNN) and hides them (blur, pixelate or black box).

Nothing is saved. A photo is read in memory, the result is offered as a
download, and that is all. No new packages are needed.

The file has two parts:
  * plain helper functions (no Streamlit inside) - these are tested on their own
  * the page itself (show_privacy_tools and the two _tab functions)
"""

import gc
import hashlib
import io
import math
import re

import numpy as np
import streamlit as st
from PIL import ExifTags, Image, ImageFilter, ImageOps

# Pillow 9.1+ calls these Image.Resampling.X; older versions call them Image.X
RESAMPLING = getattr(Image, "Resampling", Image)

# ---------------------------------------------------------------------------
# Limits. A photo is untrusted input from the internet, so we refuse silly sizes
# (a "decompression bomb" is a tiny file that unpacks into a gigantic picture).
# ---------------------------------------------------------------------------
MAX_FILE_BYTES = 25 * 1024 * 1024      # 25 MB
MAX_PIXELS = 25_000_000                # 25 megapixels (the server's memory is limited)
MIN_FACE_CONFIDENCE = 0.5              # same idea as the 0.5 check in app.py
FACE_PADDING = 0.25                    # hide 25% extra around each face (forehead, chin, ears)
UPLOAD_TYPES = ["jpg", "jpeg", "png", "jfif", "webp", "bmp"]

STYLE_BLUR = "Strong blur"
STYLE_PIXEL = "Pixelate"
STYLE_BLACK = "Black box (safest)"

# EXIF tag numbers we care about
TAG_MAKE, TAG_MODEL, TAG_SOFTWARE, TAG_DATETIME = 0x010F, 0x0110, 0x0131, 0x0132
TAG_ARTIST, TAG_COPYRIGHT = 0x013B, 0x8298
TAG_EXIF_IFD, TAG_GPS_IFD = 0x8769, 0x8825
TAG_DATETIME_ORIGINAL, TAG_DATETIME_DIGITIZED = 0x9003, 0x9004
TAG_MAKERNOTE = 0x927C
TAG_OWNER, TAG_BODY_SERIAL, TAG_LENS_MODEL, TAG_LENS_SERIAL = 0xA430, 0xA431, 0xA434, 0xA435

# Things in image.info that are only technical (not private)
TECHNICAL_INFO_KEYS = {
    "dpi", "icc_profile", "exif", "xmp", "jfif", "jfif_version", "jfif_unit", "jfif_density",
    "progressive", "progression", "adobe", "adobe_transform", "gamma", "transparency", "srgb",
    "chromaticity", "interlace", "compression", "duration", "loop", "background", "aspect",
    "mpoffset", "mp", "mpinfo", "photoshop", "dpi_unit",
}


# ===========================================================================
# Part 1 - plain helper functions (no Streamlit)
# ===========================================================================

def safe_text(value, limit=80):
    """Turn any value from a photo into harmless text.

    Metadata comes from a file somebody else made, so it must never be trusted.
    Here we drop control characters and the characters that Markdown uses to
    make links and pictures, so a nasty value cannot pull in a picture from the
    internet or show a fake link.
    """
    if isinstance(value, bytes):
        return f"<{len(value)} bytes of data>".replace("<", "(").replace(">", ")")
    text = str(value)
    text = re.sub(r"[\r\n\t\v\f]+", " ", text)        # line breaks and tabs become a space
    text = "".join(ch for ch in text if ch.isprintable())
    for bad in "[]<>`":
        text = text.replace(bad, " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _number(v):
    """Rational / tuple / number -> float, or None if it is broken."""
    try:
        if isinstance(v, tuple) and len(v) == 2:
            f = v[0] / v[1]
        else:
            f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _dms_to_degrees(dms, ref):
    """GPS stores 3 numbers (degrees, minutes, seconds) plus N/S/E/W."""
    try:
        parts = [_number(p) for p in dms]
    except Exception:
        return None
    if len(parts) != 3 or any(p is None for p in parts):
        return None
    degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if str(ref).strip().upper()[:1] in ("S", "W"):
        degrees = -degrees
    return degrees


def parse_gps(gps_ifd):
    """Return {"lat", "lon", "map_url"} from the GPS part of the EXIF data, or None."""
    if not gps_ifd:
        return None
    lat = _dms_to_degrees(gps_ifd.get(2), gps_ifd.get(1)) if gps_ifd.get(2) else None
    lon = _dms_to_degrees(gps_ifd.get(4), gps_ifd.get(3)) if gps_ifd.get(4) else None
    if lat is None or lon is None:
        return None
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    if lat == 0 and lon == 0:      # many cameras write 0,0 when they had no GPS signal
        return None
    return {
        "lat": lat,
        "lon": lon,
        "map_url": f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=16/{lat:.5f}/{lon:.5f}",
    }


def _nice_date(raw):
    """'2024:05:03 14:22:10' -> '2024-05-03 14:22:10'."""
    text = safe_text(raw)
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}:\d{2}:\d{2})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}"
    return text or None


def _open_checked(data):
    """Open picture bytes safely. Raises ValueError with a friendly message."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"This file is too big (more than {MAX_FILE_BYTES // (1024 * 1024)} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size
    except Exception:
        raise ValueError("This file could not be opened as a picture.") from None
    if width * height > MAX_PIXELS:
        raise ValueError(f"This picture is too large (more than {MAX_PIXELS // 1_000_000} megapixels). "
                         "Please use a smaller copy of it.")
    return img


def read_metadata(data):
    """Look inside a photo file and report the hidden details.

    Returns a dict:
      format, size, device, taken_at, gps, serial, names, software, lens,
      other_notes (XMP / text notes), tags (every EXIF field, as text),
      level ("none" | "medium" | "high")
    """
    img = _open_checked(data)

    base, exif_ifd, gps_ifd = {}, {}, {}
    try:
        exif = img.getexif()
        base = dict(exif)
        try:
            exif_ifd = dict(exif.get_ifd(TAG_EXIF_IFD))
        except Exception:
            exif_ifd = {}
        try:
            gps_ifd = dict(exif.get_ifd(TAG_GPS_IFD))
        except Exception:
            gps_ifd = {}
    except Exception:
        pass

    def pick(*sources_and_tags):
        for source, tag in sources_and_tags:
            value = source.get(tag)
            if value not in (None, "", b""):
                text = safe_text(value)
                if text:
                    return text
        return None

    make = pick((base, TAG_MAKE))
    model = pick((base, TAG_MODEL))
    device = None
    if make and model:
        device = model if model.lower().startswith(make.lower()) else f"{make} {model}"
    else:
        device = model or make

    names = []
    for source, tag in ((base, TAG_ARTIST), (exif_ifd, TAG_OWNER), (base, TAG_COPYRIGHT)):
        text = pick((source, tag))
        if text and text not in names:
            names.append(text)

    other_notes = []
    info = getattr(img, "info", {}) or {}
    if "xmp" in info or "XML:com.adobe.xmp" in info:
        other_notes.append("XMP data (can hold editing history, names or places)")
    for key, value in info.items():
        if str(key).lower() in TECHNICAL_INFO_KEYS:
            continue
        if isinstance(value, (str, bytes)) and value:
            other_notes.append(f"Text note \"{safe_text(key, 30)}\": {safe_text(value, 60)}")

    tags = []
    for source, lookup in ((base, ExifTags.TAGS), (exif_ifd, ExifTags.TAGS), (gps_ifd, ExifTags.GPSTAGS)):
        for tag, value in source.items():
            if tag in (TAG_EXIF_IFD, TAG_GPS_IFD, TAG_MAKERNOTE):
                continue
            name = lookup.get(tag, f"Tag 0x{tag:04X}") if isinstance(tag, int) else str(tag)
            tags.append((safe_text(name, 40), safe_text(value, 100)))

    result = {
        "format": img.format or "?",
        "size": img.size,
        "device": device,
        "taken_at": (_nice_date(exif_ifd.get(TAG_DATETIME_ORIGINAL)
                                or exif_ifd.get(TAG_DATETIME_DIGITIZED)
                                or base.get(TAG_DATETIME)) if (
            exif_ifd.get(TAG_DATETIME_ORIGINAL) or exif_ifd.get(TAG_DATETIME_DIGITIZED)
            or base.get(TAG_DATETIME)) else None),
        "gps": parse_gps(gps_ifd),
        "serial": pick((exif_ifd, TAG_BODY_SERIAL), (exif_ifd, TAG_LENS_SERIAL)),
        "names": names,
        "software": pick((base, TAG_SOFTWARE)),
        "lens": pick((exif_ifd, TAG_LENS_MODEL)),
        "other_notes": other_notes,
        "tags": tags,
    }

    if result["gps"]:
        result["level"] = "high"
    elif any(result[k] for k in ("device", "taken_at", "serial", "names", "software", "lens", "other_notes")):
        result["level"] = "medium"
    else:
        result["level"] = "none"
    return result


def _to_srgb(img, icc_bytes):
    """Phone photos often use a wide colour profile. Removing the profile without
    converting first makes the colours look dull, so convert to plain sRGB first."""
    if not icc_bytes:
        return img
    try:
        from PIL import ImageCms
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        target = ImageCms.createProfile("sRGB")
        rgb = img.convert("RGB")
        converted = ImageCms.profileToProfile(rgb, source, target, outputMode="RGB")
        if "A" in img.getbands():
            converted.putalpha(img.getchannel("A"))
        return converted
    except Exception:
        return img


def _pixels_only(img):
    """A new picture that holds only the pixels, nothing else."""
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    arr = np.array(img.convert("RGBA" if has_alpha else "RGB"))
    return Image.fromarray(arr)


def _encode(img, is_jpeg):
    """Picture -> (bytes, file extension, mime type)."""
    out = io.BytesIO()
    if is_jpeg:
        img.convert("RGB").save(out, format="JPEG", quality=95, optimize=True)
        return out.getvalue(), "jpg", "image/jpeg"
    img.save(out, format="PNG")
    return out.getvalue(), "png", "image/png"


def remove_metadata(data):
    """Make a clean copy of a photo that holds only the pixels.

    Returns (clean_bytes, extension, mime). The copy is built from the pixel
    values alone, so nothing else can come along: no EXIF, GPS, XMP, text notes
    or thumbnails. The photo is turned upright first, because the "which way up"
    note is removed too.
    """
    img = _open_checked(data)
    is_jpeg = (img.format or "").upper() in ("JPEG", "MPO")
    icc = img.info.get("icc_profile")
    img.load()
    upright = ImageOps.exif_transpose(img)
    upright = _to_srgb(upright, icc)
    clean = _pixels_only(upright)
    return _encode(clean, is_jpeg)


def load_upright_rgb(data):
    """Open a photo, turn it the right way up and return it as an RGB picture."""
    img = _open_checked(data)
    img.load()
    upright = ImageOps.exif_transpose(img)
    return _to_srgb(upright, img.info.get("icc_profile")).convert("RGB")


def pad_box(x, y, w, h, img_w, img_h, pad=FACE_PADDING):
    """Grow a face box by `pad` on every side and keep it inside the picture."""
    px, py = w * pad, h * pad
    left = max(0, int(math.floor(x - px)))
    top = max(0, int(math.floor(y - py)))
    right = min(img_w, int(math.ceil(x + w + px)))
    bottom = min(img_h, int(math.ceil(y + h + py)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def hide_faces(img, boxes, style=STYLE_BLUR, strength=7):
    """Return a copy of `img` with every face box hidden.

    boxes: list of (x, y, w, h) in the picture's own pixels.
    strength (1-10) matters for blur and pixelate; the black box ignores it.
    """
    out = img.convert("RGB").copy()
    width, height = out.size
    strength = max(1, min(10, int(strength)))

    for (x, y, w, h) in boxes:
        area = pad_box(x, y, w, h, width, height)
        if area is None:
            continue
        left, top, right, bottom = area
        region = out.crop(area)
        rw, rh = region.size

        if style == STYLE_BLACK:
            region = Image.new("RGB", (rw, rh), (0, 0, 0))
        elif style == STYLE_PIXEL:
            cells = max(3, 14 - strength)                    # fewer cells = harder to recognise
            cx = max(1, round(cells * rw / max(rw, rh)))
            cy = max(1, round(cells * rh / max(rw, rh)))
            region = region.resize((cx, cy), RESAMPLING.BOX).resize((rw, rh), RESAMPLING.NEAREST)
        else:
            radius = max(rw, rh) * 0.03 * strength           # relative to the face size
            region = region.filter(ImageFilter.GaussianBlur(radius))

        out.paste(region, (left, top))
    return out


def detect_faces(img, high_detail=False):
    """Find faces with DeepFace + MTCNN (already installed for the emotion feature).

    Returns a list of (x, y, w, h) in the ORIGINAL picture's pixels.
    The photo is shrunk first to save memory, then the boxes are scaled back.
    """
    from deepface import DeepFace      # already loaded by app.py, so this is instant

    max_side = 1280 if high_detail else 800
    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    small = img if scale == 1.0 else img.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), RESAMPLING.LANCZOS)
    bgr = np.array(small.convert("RGB"))[:, :, ::-1].copy()    # DeepFace uses OpenCV's BGR order

    try:
        found = DeepFace.extract_faces(
            img_path=bgr,
            detector_backend="mtcnn",
            enforce_detection=False,       # no face -> no error
            align=False,
        )
    finally:
        gc.collect()

    boxes = []
    for item in found:
        # When no face exists, DeepFace returns the whole picture with confidence 0.
        if float(item.get("confidence", 0) or 0) < MIN_FACE_CONFIDENCE:
            continue
        area = item["facial_area"]
        boxes.append((
            area["x"] / scale, area["y"] / scale,
            area["w"] / scale, area["h"] / scale,
        ))
    return boxes


def file_key(data):
    """A short fingerprint of the file, so we only analyse a new photo once."""
    return hashlib.sha256(data).hexdigest()


# ===========================================================================
# Part 2 - the page
# ===========================================================================

def _metadata_tab():
    st.write(
        "Photos from a phone or camera often carry **hidden details**: where it was taken, "
        "which phone took it and when. Upload one to see what is inside."
    )
    uploaded = st.file_uploader("Upload a photo to check", type=UPLOAD_TYPES, key="pv_meta_upload")
    if uploaded is None:
        st.info(
            "Tip: Some apps (WhatsApp, Instagram) already remove this when you send a photo, "
            "but many websites, emails and cloud links do not."
        )
        return

    data = uploaded.getvalue()
    try:
        info = read_metadata(data)
    except ValueError as e:
        st.error(str(e))
        return

    if info["level"] == "high":
        st.error("⚠️ This photo tells people **where** it was taken (GPS location).")
    elif info["level"] == "medium":
        st.warning("This photo carries hidden details about your device, time or name.")
    else:
        st.success("✅ No hidden location, phone, time or name found in this file.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 Location", "FOUND" if info["gps"] else "None")
    with col2:
        st.metric("📱 Phone / camera", info["device"] or "None")
    with col3:
        st.metric("🕒 Taken at", info["taken_at"] or "None")

    if info["gps"]:
        gps = info["gps"]
        st.write(f"Exact spot stored in the photo: **{gps['lat']:.5f}, {gps['lon']:.5f}**")
        st.markdown(f"[🗺️ Open this spot on a map]({gps['map_url']})")

    extra = []
    if info["serial"]:
        extra.append(f"Camera serial number: {info['serial']}")
    for name in info["names"]:
        extra.append(f"Name / owner / copyright: {name}")
    if info["software"]:
        extra.append(f"Software: {info['software']}")
    if info["lens"]:
        extra.append(f"Lens: {info['lens']}")
    extra.extend(info["other_notes"])
    if extra:
        st.write("**Other hidden details**")
        st.code("\n".join(extra), language=None)

    if info["tags"]:
        with st.expander(f"See every hidden field ({len(info['tags'])})"):
            st.code("\n".join(f"{name}: {value}" for name, value in info["tags"]), language=None)

    st.subheader("🧹 Remove the hidden data")
    if info["level"] == "none":
        st.write("Nothing to remove here.")
        return

    # Build the clean copy once per photo (the download button re-runs the page)
    key = file_key(data)
    cached = st.session_state.get("pv_clean")
    if not cached or cached["key"] != key:
        try:
            clean_bytes, ext, mime = remove_metadata(data)
            check = read_metadata(clean_bytes)
        except ValueError as e:
            st.error(str(e))
            return
        cached = {"key": key, "bytes": clean_bytes, "ext": ext, "mime": mime, "level": check["level"]}
        st.session_state["pv_clean"] = cached

    stem = (uploaded.name or "photo").rsplit(".", 1)[0]
    st.download_button(
        "🧹 Remove hidden data and download the clean copy",
        data=cached["bytes"],
        file_name=f"{stem}_clean.{cached['ext']}",
        mime=cached["mime"],
    )
    if cached["level"] == "none":
        st.success("Checked the clean copy again: no hidden location, phone, time or name is left.")
    else:
        st.warning("The clean copy still has some hidden details. Please do not share it yet.")
    st.caption(
        "The clean copy is rebuilt from the picture's pixels only, so it looks the same but carries nothing else. "
        "Your original file is not changed."
    )


def _blur_tab():
    st.write(
        "Upload a photo and SmartVision hides **every face** it can find, so you can share the photo "
        "without showing people who did not agree."
    )
    uploaded = st.file_uploader("Upload a photo with faces", type=UPLOAD_TYPES, key="pv_blur_upload")

    style = st.radio("How should faces be hidden?", [STYLE_BLUR, STYLE_PIXEL, STYLE_BLACK], horizontal=True)
    strength = st.slider(
        "Strength (higher = harder to recognise)", 1, 10, 7,
        disabled=(style == STYLE_BLACK),
    )
    high_detail = st.checkbox("Group photo or small faces (slower, finds smaller faces)")

    if uploaded is None:
        st.info("Upload a photo to start. Faces are found by AI, so always look at the result before you share it.")
        return

    data = uploaded.getvalue()
    try:
        img = load_upright_rgb(data)
    except ValueError as e:
        st.error(str(e))
        return

    # Find the faces once per photo and per detail setting
    key = f"{file_key(data)}:{high_detail}"
    cached = st.session_state.get("pv_faces")
    if not cached or cached["key"] != key:
        with st.spinner("Looking for faces..."):
            try:
                boxes = detect_faces(img, high_detail=high_detail)
            except Exception as e:
                st.error(f"Face search failed: {type(e).__name__} - {e}")
                return
        cached = {"key": key, "boxes": boxes}
        st.session_state["pv_faces"] = cached
    boxes = cached["boxes"]

    if not boxes:
        st.warning(
            "No face found, so nothing was hidden. Try the 'Group photo or small faces' option. "
            "Faces that are tiny, turned sideways, covered or in the dark can be missed."
        )
        st.image(img, caption="Your photo (unchanged)", width="stretch", output_format="PNG")
        return

    result = hide_faces(img, boxes, style=style, strength=strength)
    st.success(f"Found {len(boxes)} face{'s' if len(boxes) != 1 else ''} and hid {'them' if len(boxes) != 1 else 'it'}.")
    st.image(result, caption="Faces hidden", width="stretch", output_format="PNG")

    is_jpeg = (uploaded.name or "").lower().endswith((".jpg", ".jpeg", ".jfif"))
    out_bytes, ext, mime = _encode(result, is_jpeg)
    stem = (uploaded.name or "photo").rsplit(".", 1)[0]
    st.download_button(
        "⬇️ Download the photo with hidden faces",
        data=out_bytes,
        file_name=f"{stem}_faces_hidden.{ext}",
        mime=mime,
    )
    st.caption(
        "The AI can miss a face, so check the picture yourself before sharing. "
        "The black box is the safest choice: a light blur or big pixels can sometimes be partly undone by software. "
        "The downloaded file also has no GPS or phone details, because it is rebuilt from the pixels."
    )
    del result
    gc.collect()


def show_privacy_tools():
    """Draw the whole page."""
    st.header("🔒 Privacy Tools")
    st.caption("Two small tools for sharing photos more safely. Your photo is handled in memory and is not saved by the app.")
    tab_meta, tab_blur = st.tabs(["🧾 Photo metadata checker", "🫥 Blur faces"])
    with tab_meta:
        _metadata_tab()
    with tab_blur:
        _blur_tab()