"""Build an 8k real answer-distribution dataset from official ENADE/ENEM releases.

The pipeline is intentionally resumable. It downloads official INEP microdata,
exam booklets, answer keys, and public structured transcriptions into a local
cache, then joins:

  ``DS_VT_ESC_*`` (the A-E option each examinee selected)
  + ``DS_VT_GAB_*`` (the official key)
  + released exam PDF text/options

into :class:`src.schema.Item` records. No synthetic labels or responses are
created. Items whose five choices cannot be recovered reliably are skipped.

Default output:
  data/brazil8k_train.jsonl   8,000 older-year items
  data/brazil8k_val.jsonl       200 newer-year items
  data/brazil8k_test.jsonl      400 newest-year items
  data/brazil8k_manifest.json   provenance and quality counts

The first run is network-heavy; partial downloads and per-year parsed items are
cached under ``data/brazil8k_raw``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import difflib
import hashlib
from html.parser import HTMLParser
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import unicodedata
from urllib.parse import unquote, urljoin, urlparse
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.schema import Item, read_items, write_items
from src import lp
from src.progress import Progress

PIPELINE_VERSION = 5
LETTERS = "ABCDE"
DEFAULT_YEARS = [
    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023,
]
MICRODATA_PAGE = (
    "https://www.gov.br/inep/pt-br/acesso-a-informacao/"
    "dados-abertos/microdados/enade"
)
EXAM_PAGE = (
    "https://www.gov.br/inep/pt-br/areas-de-atuacao/"
    "avaliacao-e-exames-educacionais/enade/provas-e-gabaritos/{year}"
)
ENEM_XML_URL = "https://www.ime.usp.br/~ddm/project/enem/ENEMdataset.zip"
DEFAULT_ENEM_YEARS = list(range(2009, 2024))
ENEM_CSV_BASE = (
    "https://raw.githubusercontent.com/gabriel-antonelli/"
    "extract-enem-data/main/enem-data/enem-{year}/{area}.csv"
)
ENEM_CSV_AREAS = (
    "ciencias-humanas", "ciencias-natureza", "linguagens", "matematica")


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore").decode("ascii")


def _slug(text: str) -> str:
    text = _ascii(unquote(text)).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _request_url(url: str) -> str:
    # download.inep.gov.br currently has an incomplete certificate chain for
    # some Python installations; plain HTTP is supported for the same files.
    if url.startswith("https://download.inep.gov.br/"):
        return "http://" + url[len("https://"):]
    return url


def _requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency message
        raise SystemExit(
            "Missing requests; run: pip install -r requirements-local.txt") from exc
    return requests


def fetch_html(url: str, retries: int = 4) -> str:
    requests = _requests()
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(
                _request_url(url), timeout=(30, 180),
                headers={"User-Agent": "EduLLM1-research/1.0"})
            r.raise_for_status()
            return r.text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")


def download(url: str, path: Path, retries: int = 6) -> Path:
    """Resume an HTTP download into ``path`` and atomically finalize it."""
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    requests = _requests()
    last = None
    for attempt in range(retries):
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "EduLLM1-research/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with requests.get(
                    _request_url(url), stream=True, timeout=(30, 300),
                    headers=headers) as r:
                if start and r.status_code != 206:
                    start = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                mode = "ab" if start else "wb"
                total = None
                content_range = r.headers.get("Content-Range", "")
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    total = int(match.group(1))
                elif r.headers.get("Content-Length"):
                    total = start + int(r.headers["Content-Length"])
                tracker = Progress(
                    f"download {path.name}",
                    max(1, math.ceil(total / 1048576)) if total else 1,
                    unit="MB", min_interval=30)
                if total and start:
                    tracker.update(start // 1048576, part="resuming", force=True)
                written = start
                with open(part, mode) as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                            if total:
                                tracker.update(
                                    written // 1048576,
                                    part=path.name)
                if total:
                    tracker.update(
                        math.ceil(total / 1048576),
                        part=path.name, force=True)
            os.replace(part, path)
            return path
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[download] retry {attempt+1}/{retries}: {url}: {exc}",
                  flush=True)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"download failed: {url}: {last}")


def _seven_zip() -> str | None:
    found = shutil.which("7z") or shutil.which("7z.exe")
    if found:
        return found
    for root in (os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)")):
        if root:
            candidate = Path(root) / "7-Zip" / "7z.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def normalized_zip(path: Path) -> Path:
    """Return a real ZIP even when an INEP ``.zip`` is 7z or RAR internally."""
    if zipfile.is_zipfile(path):
        return path
    normalized = path.with_name(path.stem + ".normalized.zip")
    if normalized.is_file() and zipfile.is_zipfile(normalized):
        return normalized
    seven = _seven_zip()
    if not seven:
        raise RuntimeError(
            f"{path.name} is not a ZIP; install 7-Zip to read INEP 7z/RAR archives")
    extract_dir = path.with_name(path.stem + "_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    subprocess.run(
        [seven, "x", "-y", f"-o{extract_dir}", str(path)],
        check=True, stdout=subprocess.DEVNULL)
    tmp = normalized.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in extract_dir.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(extract_dir))
    os.replace(tmp, normalized)
    shutil.rmtree(extract_dir)
    return normalized


def _discover_microdata_urls() -> dict[int, str]:
    parser = _Links()
    parser.feed(fetch_html(MICRODATA_PAGE))
    out = {}
    for href in parser.hrefs:
        m = re.search(
            r"microdados[_-]?enade[_-]?(\d{4}).*\.(?:zip|rar)", href, re.I)
        if m:
            out[int(m.group(1))] = href
    return out


def _asset_slug(url: str) -> str:
    stem = Path(urlparse(url).path).stem
    stem = re.sub(r"^enade[_-]?\d{4}[_-]", "", stem, flags=re.I)
    stem = re.sub(r"^\d{4}[_-]", "", stem)
    stem = re.sub(r"^\d{1,3}[_-]+", "", stem)
    stem = re.sub(r"^gab(?:\d{2,4})?[_-]", "", stem, flags=re.I)
    stem = re.sub(
        r"^(?:pv|gb|prova|gabarito|gab[_-]?def(?:initivo)?)[_-]",
        "", stem, flags=re.I)
    stem = re.sub(
        r"[_-](?:ledor|ampliada|super[_-]?ampliada|revisad[oa])$",
        "", stem, flags=re.I)
    return _slug(stem)


def _discover_exam_assets(year: int) -> dict[str, dict[str, str]]:
    page = EXAM_PAGE.format(year=year)
    parser = _Links()
    parser.feed(fetch_html(page))
    assets: dict[str, dict[str, str]] = defaultdict(dict)
    for href in parser.hrefs:
        url = urljoin(page, href)
        low = unquote(url).lower()
        if ("download.inep.gov.br" not in low or
                "padrao_resposta" in low):
            continue
        # A few 2022 links accidentally omit the .pdf suffix.
        if not low.endswith(".pdf") and re.search(
                rf"{year}_(?:pv|gb)_", low):
            url += ".pdf"
            low += ".pdf"
        if not low.endswith(".pdf"):
            continue
        embedded_years = re.findall(r"/(?:provas|gabaritos)/(\d{4})/", low)
        if embedded_years and str(year) not in embedded_years:
            # Some year pages contain stale links to a prior administration.
            continue
        name = Path(urlparse(url).path).stem.lower()
        slug = _asset_slug(url)
        if not slug:
            continue
        is_exam = (
            "_pv_" in name or "prova" in name or "/provas/" in low)
        is_key = (
            "_gb_" in name or name.startswith("gab") or
            "gabarit" in low or "/gabaritos/" in low)
        variant = any(x in name for x in (
            "ledor", "ampliada", "super_ampliada"))
        if is_exam:
            # Prefer a standard booklet. Fall back to the screen-reader version
            # only when no standard PDF is linked.
            if not variant or "exam" not in assets[slug]:
                assets[slug]["exam"] = url
        elif is_key:
            assets[slug]["key"] = url
    keys = {k: v["key"] for k, v in assets.items() if "key" in v}
    out = {}
    for slug, value in assets.items():
        if "exam" not in value:
            continue
        value = dict(value)
        if "key" not in value and keys:
            match = max(
                keys,
                key=lambda k: difflib.SequenceMatcher(None, slug, k).ratio())
            score = difflib.SequenceMatcher(None, slug, match).ratio()
            if score >= 0.58:
                value["key"] = keys[match]
        out[slug] = value
    return out


def _decode_header(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1", errors="replace").strip()


def _member_headers(z: zipfile.ZipFile) -> dict[str, list[str]]:
    out = {}
    for name in z.namelist():
        if not name.lower().endswith((".txt", ".csv")):
            continue
        with z.open(name) as f:
            line = _decode_header(f.readline())
        delim = ";" if line.count(";") >= line.count(",") else ","
        # LGPD-era releases (2016-2022) quote every header name, while 2023
        # does not. csv.DictReader removes these quotes from row keys; member
        # discovery must normalize them the same way.
        out[name] = [
            x.strip().lstrip("\ufeff").strip('"') for x in line.split(delim)]
    return out


def _member_with(headers, required, preferred=()):
    candidates = [
        name for name, cols in headers.items()
        if all(col in cols for col in required)
    ]
    if not candidates:
        return None
    for token in preferred:
        matches = [x for x in candidates if token.lower() in x.lower()]
        if matches:
            return min(matches, key=len)
    return min(candidates, key=len)


def _text_reader(z: zipfile.ZipFile, name: str):
    raw = z.open(name)
    # INEP files are consistently semicolon-delimited; latin-1 also decodes
    # UTF-8 ASCII headers and avoids an expensive whole-file encoding probe.
    return raw, io.TextIOWrapper(raw, encoding="latin-1", newline="")


def _id(value) -> str:
    return str(value or "").strip().removesuffix(".0")


def _choice_vector(value) -> str:
    # These are fixed-width positional vectors. Never delete placeholders:
    # doing so shifts every subsequent answer onto the wrong question.
    return re.sub(r"\s+", "", str(value or "").strip().upper())


def _valid_vector_pair(choices: str, key: str) -> bool:
    return (
        bool(choices) and len(choices) == len(key) and
        set(choices) <= set("ABCDE.*9") and
        set(key) <= set("ABCDE.89XZ*"))


def _field(cols, names):
    return next((name for name in names if name in cols), None)


@dataclass
class _Group:
    key_counts: Counter = field(default_factory=Counter)
    position_counts: list[Counter] = field(default_factory=list)

    def add(self, choices: str, key: str):
        if key:
            self.key_counts[key] += 1
        while len(self.position_counts) < len(choices):
            self.position_counts.append(Counter())
        for i, choice in enumerate(choices):
            if choice in LETTERS:
                self.position_counts[i][choice] += 1

    def key(self) -> str:
        return self.key_counts.most_common(1)[0][0] if self.key_counts else ""

    def key_consistency(self) -> float:
        total = sum(self.key_counts.values())
        return self.key_counts.most_common(1)[0][1] / total if total else 0.0


@dataclass
class _ResponseData:
    groups: dict[str, _Group]
    general: _Group
    labels: dict[str, str]


def _group_labels(z: zipfile.ZipFile) -> dict[str, str]:
    """Extract CO_GRUPO value labels from the bundled XLSX dictionary."""
    try:
        import openpyxl
    except ImportError:
        return {}
    xlsx = next((n for n in z.namelist()
                 if n.lower().endswith(".xlsx") and "dicion" in _slug(n)), None)
    if not xlsx:
        return {}
    wb = openpyxl.load_workbook(
        io.BytesIO(z.read(xlsx)), read_only=True, data_only=True)
    labels = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        for i, row in enumerate(rows):
            if not row or str(row[0] or "").strip() != "CO_GRUPO":
                continue
            j = i
            while j < len(rows):
                cur = rows[j]
                if j > i and cur and cur[0] not in (None, ""):
                    break
                for value in cur:
                    for code, label in re.findall(
                            r"(\d+)\s*=\s*([^\n;]+)", str(value or "")):
                        labels[code] = label.strip()
                j += 1
            return labels
    return labels


def read_enade_responses(zip_path: Path) -> _ResponseData:
    """Stream one ENADE archive into per-area per-option counts."""
    with zipfile.ZipFile(zip_path) as z:
        headers = _member_headers(z)
        response_name = _member_with(
            headers, ["CO_CURSO", "DS_VT_ESC_OCE"], preferred=("arq3",))
        if not response_name:
            raise ValueError(f"{zip_path.name}: no exact-choice response file")
        response_cols = headers[response_name]
        map_name = _member_with(
            headers, ["CO_CURSO", "CO_GRUPO"], preferred=("arq1",))
        course_to_group = {}
        if map_name and map_name != response_name:
            raw, text = _text_reader(z, map_name)
            try:
                for row in csv.DictReader(text, delimiter=";"):
                    course_to_group[_id(row.get("CO_CURSO"))] = _id(
                        row.get("CO_GRUPO"))
            finally:
                text.close()
                raw.close()

        oce_key = _field(response_cols, (
            "DS_VT_GAB_OCE_FIN", "DS_VT_GAB_OCE"))
        ofg_key = _field(response_cols, (
            "DS_VT_GAB_OFG_FIN", "DS_VT_GAB_OFG"))
        groups: dict[str, _Group] = defaultdict(_Group)
        general = _Group()
        raw, text = _text_reader(z, response_name)
        try:
            for row in csv.DictReader(text, delimiter=";"):
                group = _id(row.get("CO_GRUPO")) or course_to_group.get(
                    _id(row.get("CO_CURSO")), "")
                if not group:
                    continue
                oce = _choice_vector(row.get("DS_VT_ESC_OCE"))
                oce_key_value = (
                    _choice_vector(row.get(oce_key)) if oce_key else "")
                if (_valid_vector_pair(oce, oce_key_value) and
                        any(c in LETTERS for c in oce)):
                    groups[group].add(oce, oce_key_value)
                ofg = _choice_vector(row.get("DS_VT_ESC_OFG"))
                ofg_key_value = (
                    _choice_vector(row.get(ofg_key)) if ofg_key else "")
                if (_valid_vector_pair(ofg, ofg_key_value) and
                        any(c in LETTERS for c in ofg)):
                    general.add(ofg, ofg_key_value)
        finally:
            text.close()
            raw.close()
        labels = _group_labels(z)
    return _ResponseData(dict(groups), general, labels)


@dataclass
class _EnemQuestion:
    year: int
    number: int
    question: str
    options: list[str]
    correct: str


def _enem_questions(raw_dir: Path) -> dict[int, list[_EnemQuestion]]:
    archive = download(
        ENEM_XML_URL, raw_dir / "enem" / "questions" / "ENEMdataset.zip")
    by_year: dict[int, list[_EnemQuestion]] = defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            m = re.match(r"(\d{4}).*\.xml$", name)
            if not m:
                continue
            if re.match(r"\d{4}_2_", name):
                # Separate reapplication; the primary microdata/item mapping
                # path below targets the regular administration.
                continue
            year = int(m.group(1))
            root = ET.fromstring(z.read(name))
            for node in root.findall(".//question"):
                try:
                    number = int(node.attrib["id"])
                except (KeyError, ValueError):
                    continue
                options = []
                correct = ""
                answers = node.find("answers")
                if answers is None:
                    continue
                for option in answers.findall("option"):
                    options.append(_clean(["".join(option.itertext())]))
                    if option.attrib.get("correct", "").lower() == "yes":
                        correct = option.attrib.get("id", "").upper()
                header = node.find("header")
                statement = node.find("statement")
                question = _clean([
                    "".join(header.itertext()) if header is not None else "",
                    "".join(statement.itertext()) if statement is not None else "",
                ])
                if (len(options) != 5 or correct not in LETTERS or
                        len(question) < 20 or any(not x for x in options)):
                    continue
                # Strict text-only mode: images are not silently replaced.
                if node.attrib.get("image", "No").lower() != "no":
                    continue
                by_year[year].append(_EnemQuestion(
                    year, number, question, options, correct))

    # The ENEM Challenge stops at 2017. A public MIT/GPL extraction project
    # provides the same structured fields for later official booklets. We use
    # only its text-only rows and still validate every answer/form against the
    # official INEP item table and microdata before retaining an item.
    for year in range(2018, 2024):
        for area in ENEM_CSV_AREAS:
            url = ENEM_CSV_BASE.format(year=year, area=area)
            path = download(
                url, raw_dir / "enem" / "questions" / str(year) / f"{area}.csv")
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        number = int(row.get("number") or 0)
                    except ValueError:
                        continue
                    images = str(row.get("context-images") or "").strip()
                    if images not in ("", "[]", "nan", "None"):
                        continue
                    options = [_clean([row.get(c, "")]) for c in LETTERS]
                    correct = str(row.get("answer") or "").strip().upper()
                    question = _clean([
                        row.get("context", ""), row.get("question", "")])
                    if (not 1 <= number <= 180 or len(question) < 20 or
                            correct not in LETTERS or any(not x for x in options)):
                        continue
                    by_year[year].append(_EnemQuestion(
                        year, number, question, options, correct))

    for year, rows in by_year.items():
        # Some source XMLs are not in numeric order.
        unique = {}
        for row in rows:
            unique.setdefault(row.number, row)
        rows[:] = sorted(unique.values(), key=lambda x: x.number)
    return dict(by_year)


def _enem_area(value) -> str:
    value = _ascii(str(value or "")).upper().strip()
    aliases = {
        "CIENCIAS DA NATUREZA E SUAS TECNOLOGIAS": "CN",
        "CIENCIAS HUMANAS E SUAS TECNOLOGIAS": "CH",
        "LINGUAGENS, CODIGOS E SUAS TECNOLOGIAS": "LC",
        "MATEMATICA E SUAS TECNOLOGIAS": "MT",
    }
    return aliases.get(value, value)


@dataclass
class _EnemData:
    item_at: dict[tuple[str, str, int], str]
    form_keys: dict[tuple[str, str], dict[int, str]]
    item_keys: dict[str, set[str]]
    counts: dict[str, Counter]
    qc: dict


def read_enem_responses(zip_path: Path) -> _EnemData:
    """Aggregate ENEM A-E vectors by stable CO_ITEM without retaining people."""
    import numpy as np
    import pandas as pd

    with zipfile.ZipFile(zip_path) as z:
        headers = _member_headers(z)
        item_name = _member_with(
            headers, ["CO_POSICAO", "CO_ITEM", "CO_PROVA", "TX_GABARITO"],
            preferred=("ITENS_PROVA", "ITENS"))
        response_name = _member_with(
            headers, ["TX_RESPOSTAS_CN", "CO_PROVA_CN"],
            preferred=("MICRODADOS_ENEM", "DADOS"))
        if not item_name or not response_name:
            raise ValueError(
                f"{zip_path.name}: ENEM item/response table not found")

        item_cols = headers[item_name]
        area_col = _field(item_cols, ("SG_AREA", "CO_AREA"))
        if not area_col:
            raise ValueError(f"{zip_path.name}: item area column missing")
        wanted_items = [
            "CO_POSICAO", "CO_ITEM", "CO_PROVA", "TX_GABARITO", area_col]
        optional_items = [
            x for x in ("IN_ITEM_ABAN", "TX_MOTIVO_ABAN", "TP_LINGUA")
            if x in item_cols]
        item_df = pd.read_csv(
            z.open(item_name), sep=";", encoding="latin-1", dtype=str,
            usecols=wanted_items + optional_items, low_memory=False)

        item_at = {}
        form_keys: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
        item_keys: dict[str, set[str]] = defaultdict(set)
        for row in item_df.to_dict("records"):
            area = _enem_area(row.get(area_col))
            form = _id(row.get("CO_PROVA"))
            item_id = _id(row.get("CO_ITEM"))
            try:
                pos = int(float(row.get("CO_POSICAO") or 0))
            except ValueError:
                continue
            key = str(row.get("TX_GABARITO") or "").upper().strip()
            if (area not in ("CN", "CH", "LC", "MT") or not form or
                    not item_id or not 1 <= pos <= 45 or key not in LETTERS):
                continue
            if str(row.get("IN_ITEM_ABAN") or "0") in ("1", "1.0"):
                continue
            # Foreign-language positions have two item rows. They are excluded
            # below because the public XML intentionally begins at position 6.
            if area == "LC" and pos <= 5:
                continue
            item_at[(area, form, pos)] = item_id
            form_keys[(area, form)][pos] = key
            item_keys[item_id].add(key)

        response_cols = headers[response_name]
        usecols = []
        area_fields = {}
        for area in ("CN", "CH", "LC", "MT"):
            fields = {
                "responses": f"TX_RESPOSTAS_{area}",
                "form": f"CO_PROVA_{area}",
                "presence": f"TP_PRESENCA_{area}",
            }
            if all(v in response_cols for v in fields.values()):
                area_fields[area] = fields
                usecols.extend(fields.values())
        if not area_fields:
            raise ValueError(f"{zip_path.name}: no ENEM area response vectors")

        counts: dict[str, Counter] = defaultdict(Counter)
        n_valid_rows = 0
        reader = pd.read_csv(
            z.open(response_name), sep=";", encoding="latin-1", dtype=str,
            usecols=sorted(set(usecols)), chunksize=75_000, low_memory=False)
        for chunk in reader:
            for area, fields in area_fields.items():
                subset = chunk.loc[
                    chunk[fields["presence"]].isin(("1", "1.0")),
                    [fields["form"], fields["responses"]]].dropna()
                for form, rows in subset.groupby(fields["form"], sort=False):
                    form = _id(form)
                    seqs = [
                        str(x).upper() for x in rows[fields["responses"]]
                        if len(str(x)) >= 45
                    ]
                    if not seqs:
                        continue
                    n_valid_rows += len(seqs)
                    array = np.frombuffer(
                        "".join(x[:45] for x in seqs).encode("ascii", "ignore"),
                        dtype="S1").reshape(len(seqs), 45)
                    start = 5 if area == "LC" else 0
                    for idx in range(start, 45):
                        item_id = item_at.get((area, form, idx + 1))
                        if not item_id:
                            continue
                        values, freqs = np.unique(
                            array[:, idx], return_counts=True)
                        for value, freq in zip(values, freqs):
                            choice = value.decode("ascii")
                            if choice in LETTERS:
                                counts[item_id][choice] += int(freq)
    return _EnemData(
        item_at, dict(form_keys), dict(item_keys), dict(counts),
        {"valid_area_rows": n_valid_rows, "items_with_counts": len(counts)})


def _match_enem_segment(
        questions: list[_EnemQuestion], data: _EnemData):
    """Match a 45-question global-number segment to (area, numeric form)."""
    if not questions:
        return None, 0.0
    best, best_score = None, 0.0
    for (area, form), keys in data.form_keys.items():
        pairs = []
        for q in questions:
            pos = (q.number - 1) % 45 + 1
            if pos in keys:
                pairs.append((q.correct, keys[pos]))
        if len(pairs) < 20:
            continue
        score = sum(a == b for a, b in pairs) / len(pairs)
        if score > best_score:
            best, best_score = (area, form), score
    return best, best_score


def build_enem_year(
        year: int, raw_dir: Path, questions: list[_EnemQuestion],
        min_responses: int, rebuild: bool = False) -> list[Item]:
    year_dir = raw_dir / "enem" / str(year)
    cache = year_dir / f"items_v{PIPELINE_VERSION}.jsonl"
    qc_path = year_dir / f"qc_v{PIPELINE_VERSION}.json"
    if cache.is_file() and qc_path.is_file() and not rebuild:
        with open(qc_path, encoding="utf-8") as f:
            if json.load(f).get("complete"):
                print(f"[ENEM {year}] using cached {cache}", flush=True)
                return read_items(str(cache))

    url = f"https://download.inep.gov.br/microdados/microdados_enem_{year}.zip"
    archive = download(url, year_dir / "microdata.zip")
    data = read_enem_responses(archive)
    segments = defaultdict(list)
    for q in questions:
        segments[(q.number - 1) // 45].append(q)

    mappings = {}
    for segment, rows in segments.items():
        match, score = _match_enem_segment(rows, data)
        if match and score >= 0.88:
            mappings[segment] = (match, score)

    items = []
    skipped = Counter()
    for q in questions:
        segment = (q.number - 1) // 45
        if segment not in mappings:
            skipped["unmatched_segment"] += 1
            continue
        (area, form), _score = mappings[segment]
        pos = (q.number - 1) % 45 + 1
        item_id = data.item_at.get((area, form, pos))
        if not item_id:
            skipped["unmapped_position"] += 1
            continue
        keys = data.item_keys.get(item_id, set())
        if len(keys) != 1 or q.correct not in keys:
            skipped["key_conflict"] += 1
            continue
        dist, n = _distribution(data.counts.get(item_id, Counter()))
        if n < min_responses:
            skipped["low_responses"] += 1
            continue
        items.append(Item(
            id=f"enem:{year}:{item_id}", question=q.question,
            options=q.options, dist=dist,
            correct_index=LETTERS.index(q.correct), n_responses=n,
            skill=area, source=f"ENEM {year}",
            population=(
                f"Brazilian upper-secondary students taking ENEM {year}"),
            language="pt"))

    unique = {}
    for item in items:
        unique.setdefault(_content_hash(item), item)
    items = list(unique.values())
    qc = {
        "year": year, "source_questions": len(questions),
        "matched_segments": {
            str(k): {"area": v[0][0], "form": v[0][1],
                     "key_agreement": round(v[1], 4)}
            for k, v in mappings.items()},
        "items": len(items), "skipped": dict(skipped),
        **data.qc, "complete": True,
    }
    year_dir.mkdir(parents=True, exist_ok=True)
    write_items(str(cache), items)
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)
    print(f"[ENEM {year}] built {len(items)} items", flush=True)
    return items


def _pdf_text(path: Path) -> str:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency message
        raise SystemExit(
            "Missing pymupdf; run: pip install -r requirements-local.txt") from exc
    doc = pymupdf.open(path)
    try:
        return "\n".join(page.get_text("text", sort=True) for page in doc)
    finally:
        doc.close()


def parse_key_pdf(path: Path) -> dict[int, str]:
    text = _ascii(_pdf_text(path)).upper()
    answers = {}
    pattern = re.compile(
        r"QUESTAO\s+(\d{1,3})\s+"
        r"(A|B|C|D|E|9|\*{2,}|ANULAD[OA]?)(?=\s|$)")
    for number, answer in pattern.findall(text):
        answers[int(number)] = (
            answer if answer in LETTERS else "9")
    if len(answers) < 5:
        # Older keys are sometimes plain two-column tables.
        for number, answer in re.findall(
                r"(?m)^\s*(\d{1,3})\s+([A-E])\s*$", text):
            answers[int(number)] = answer
    return answers


def _clean(lines: list[str]) -> str:
    drop = re.compile(
        r"^(?:ENADE\b|EXAME NACIONAL\b|AREA LIVRE\b|"
        r"QUESTIONARIO\b|COMPONENTE ESPECIFICO\b|FORMACAO GERAL\b|"
        r"\d+\s*$)", re.I)
    kept = []
    for line in lines:
        line = unicodedata.normalize("NFKC", line)
        line = re.sub(r"hƩ\s*ps", "https", line, flags=re.I)
        line = re.sub(r"\s+", " ", line).strip()
        # INEP overlays repeated anti-copy watermarks in the PDF text layer.
        # Everything after the first watermark occurrence on a line is page
        # furniture, not question or option content.
        line = re.split(
            r"VALIDIN(?:E|3)P(?:20\d{2}|2)?|\*[Rr]\d+\*|"
            r"AS QUEST(?:O|Õ)ES ABAIXO VISAM CONHECER SUA OPINI(?:A|Ã)O|"
            r"QUEST(?:A|Ã)O DISCURSIVA|RASCUNHO",
            line,
            maxsplit=1, flags=re.I)[0].strip()
        if not line or drop.match(_ascii(line)):
            continue
        kept.append(line)
    return " ".join(kept).strip()


def _parse_block(lines: list[str]):
    marks = []
    for i, line in enumerate(lines):
        raw_line = re.sub(r"\s+", " ", line).strip()
        norm = _ascii(raw_line)
        m = re.match(r"^\(?([A-E])\)?(?:[\s.):\-]+|$)", norm)
        if m:
            letter = m.group(1)
            rest = re.sub(
                r"^\(?[A-E]\)?[\s.):\-]*", "", raw_line, count=1).strip()
            marks.append((i, letter, rest))
    starts = [
        i for i in range(max(0, len(marks) - 4))
        if [x[1] for x in marks[i:i+5]] == list(LETTERS)
    ]
    if not starts:
        return None
    chosen = marks[starts[-1]:starts[-1] + 5]
    q_end = chosen[0][0]
    question = _clean(lines[:q_end])
    options = []
    for j, (line_i, _letter, rest) in enumerate(chosen):
        end = chosen[j + 1][0] if j + 1 < 5 else len(lines)
        body = ([rest] if rest else []) + lines[line_i + 1:end]
        options.append(_clean(body))
    if len(question) < 20 or any(not x for x in options):
        return None
    normalized_options = {
        re.sub(r"\s+", " ", _ascii(x).casefold()).strip() for x in options}
    if len(normalized_options) != len(options):
        return None
    if "\ufffd" in question or any("\ufffd" in x for x in options):
        return None
    if len(question) > 12000 or any(len(x) > 4000 for x in options):
        return None
    return question, options


def parse_exam_pdf(path: Path) -> dict[int, tuple[str, list[str]]]:
    lines = _pdf_text(path).splitlines()
    markers = []
    for i, line in enumerate(lines):
        norm = _ascii(re.sub(r"\s+", " ", line).strip()).upper()
        m = re.match(r"^QUESTAO\s+(\d{1,3})\b", norm)
        if m:
            markers.append((i, int(m.group(1))))
    out = {}
    for j, (start, number) in enumerate(markers):
        end = markers[j + 1][0] if j + 1 < len(markers) else len(lines)
        parsed = _parse_block(lines[start + 1:end])
        if parsed:
            # Student-perception questionnaires at the back of some booklets
            # restart numbering at QUESTAO 01. The actual exam appears first.
            out.setdefault(number, parsed)
    return out


def _best_asset(label: str, assets: dict[str, dict[str, str]]):
    target = _slug(label)
    if not target or not assets:
        return None, 0.0, 0.0
    ranked = sorted(
        ((difflib.SequenceMatcher(None, target, slug).ratio(), slug)
         for slug in assets),
        reverse=True)
    score, best_slug = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    return best_slug, score, score - runner_up


def _distribution(counter: Counter):
    total = sum(counter[c] for c in LETTERS)
    return ([counter[c] / total for c in LETTERS], total) if total else ([], 0)


def _section_numbers(
        parsed: dict[int, tuple[str, list[str]]],
        key_answers: dict[int, str], n_general: int, n_specific: int):
    # The definitive key lists only objective questions. Use its ordering to
    # skip interleaved discursive questions without hard-coding year formats.
    objective = ([
        q for q in sorted(key_answers)
        if key_answers[q] in LETTERS + "9"
    ] if key_answers else sorted(parsed))
    need = n_general + n_specific
    return objective[:n_general], objective[n_general:need]


def build_year(
        year: int, raw_dir: Path, microdata_url: str,
        min_responses: int, rebuild: bool = False,
        download_workers: int = 4) -> list[Item]:
    year_dir = raw_dir / "enade" / str(year)
    cache = year_dir / f"items_v{PIPELINE_VERSION}.jsonl"
    qc_path = year_dir / f"qc_v{PIPELINE_VERSION}.json"
    if cache.is_file() and qc_path.is_file() and not rebuild:
        with open(qc_path, encoding="utf-8") as f:
            prior_qc = json.load(f)
        if prior_qc.get("complete"):
            print(f"[{year}] using cached {cache}", flush=True)
            return read_items(str(cache))

    suffix = Path(urlparse(microdata_url).path).suffix or ".archive"
    archive = download(microdata_url, year_dir / f"microdata{suffix}")
    archive = normalized_zip(archive)
    response = read_enade_responses(archive)
    print(f"[{year}] response groups={len(response.groups)}", flush=True)
    assets = _discover_exam_assets(year)
    if not assets:
        raise RuntimeError(f"{year}: no exam assets discovered")

    items: list[Item] = []
    general_done = False
    used_assets = set()
    parsed_cache = {}
    key_cache = {}
    qc = {
        "year": year, "groups_in_responses": len(response.groups),
        "assets_discovered": len(assets), "groups_matched": 0,
        "groups_skipped": [], "items": 0,
    }
    matches = {}
    claimed_assets = set()
    urls_to_paths = {}
    path_owners = {}
    for group_id in sorted(response.groups):
        label = response.labels.get(group_id, f"group {group_id}")
        asset_slug, match, margin = _best_asset(label, assets)
        if (not asset_slug or match < 0.55 or margin < 0.03 or
                asset_slug in claimed_assets):
            qc["groups_skipped"].append({
                "group": group_id, "label": label, "reason": "no PDF match",
                "match": round(match, 3), "runner_up_margin": round(margin, 3)})
            continue
        asset = assets[asset_slug]
        if "key" not in asset:
            qc["groups_skipped"].append({
                "group": group_id, "label": label,
                "reason": "definitive answer key unavailable"})
            continue
        matches[group_id] = (label, asset_slug)
        claimed_assets.add(asset_slug)
        for kind in ("exam", "key"):
            if kind not in asset:
                continue
            url = asset[kind]
            filename = unquote(Path(urlparse(url).path).name)
            candidate = year_dir / "pdf" / filename
            if candidate in path_owners:
                prior_url, prior_kind = path_owners[candidate]
                urls_to_paths[prior_url] = candidate.with_name(
                    f"{prior_kind}_{filename}")
                urls_to_paths[url] = candidate.with_name(f"{kind}_{filename}")
            else:
                path_owners[candidate] = (url, kind)
                urls_to_paths[url] = candidate

    # INEP's file server has high per-request latency. A small bounded pool
    # cuts wall time while remaining polite and keeps every .part resumable.
    downloaded = {}
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as pool:
        futures = {
            pool.submit(download, url, path): url
            for url, path in urls_to_paths.items()
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                downloaded[url] = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] [{year}] asset failed: {url}: {exc}", flush=True)

    group_rows = sorted(matches.items())
    group_progress = Progress(
        f"ENADE {year} exam parsing", len(group_rows), unit="exam groups")
    for group_index, (group_id, (label, asset_slug)) in enumerate(
            group_rows, start=1):
        group_progress.update(
            group_index - 1, part=label, force=group_index == 1)
        group = response.groups[group_id]
        if group.key_consistency() < 0.999:
            qc["groups_skipped"].append({
                "group": group_id, "label": label,
                "reason": (
                    f"inconsistent microdata keys "
                    f"({group.key_consistency():.3%} dominant)")})
            continue
        asset = assets[asset_slug]
        exam_path = downloaded.get(asset["exam"])
        key_path = downloaded.get(asset.get("key", ""))
        if not exam_path:
            qc["groups_skipped"].append({
                "group": group_id, "label": label,
                "reason": "booklet download failed"})
            continue
        try:
            if exam_path not in parsed_cache:
                parsed_cache[exam_path] = parse_exam_pdf(exam_path)
            if key_path and key_path not in key_cache:
                key_cache[key_path] = parse_key_pdf(key_path)
            parsed = parsed_cache[exam_path]
            key_answers = key_cache[key_path] if key_path else {}
        except Exception as exc:  # noqa: BLE001
            qc["groups_skipped"].append({
                "group": group_id, "label": label,
                "reason": f"PDF parse: {exc}"})
            continue

        key_specific = group.key()
        key_general = response.general.key()
        general_nums, specific_nums = _section_numbers(
            parsed, key_answers, len(key_general), len(key_specific))
        if len(specific_nums) < len(key_specific):
            qc["groups_skipped"].append({
                "group": group_id, "label": label,
                "reason": (f"only {len(specific_nums)}/{len(key_specific)} "
                           "specific MCQs parsed")})
            continue
        if key_answers:
            observed = "".join(key_answers[q] for q in specific_nums)
            comparable = [
                (a, b) for a, b in zip(observed, key_specific)
                if a in LETTERS and b in LETTERS
            ]
            agreement = (
                sum(a == b for a, b in comparable) / len(comparable)
                if comparable else 0.0)
            coverage = len(comparable) / max(1, len(key_specific))
            if len(comparable) < 20 or agreement < 0.95:
                qc["groups_skipped"].append({
                    "group": group_id, "label": label,
                    "reason": (
                        f"answer-key mismatch ({agreement:.1%} agreement, "
                        f"{coverage:.1%} coverage)")})
                continue
        else:
            qc["groups_skipped"].append({
                "group": group_id, "label": label,
                "reason": "definitive answer key unavailable"})
            continue

        qc["groups_matched"] += 1
        used_assets.add(asset_slug)
        if (not general_done and response.general.key_consistency() >= 0.999
                and len(general_nums) == len(key_general)):
            general_added = 0
            for pos, qnum in enumerate(general_nums):
                correct = key_general[pos] if pos < len(key_general) else ""
                if (correct not in LETTERS or qnum not in parsed or
                        pos >= len(response.general.position_counts) or
                        key_answers.get(qnum) != correct):
                    continue
                dist, n = _distribution(response.general.position_counts[pos])
                if n < min_responses:
                    continue
                question, options = parsed[qnum]
                items.append(Item(
                    id=f"enade:{year}:general:{qnum}",
                    question=question, options=options, dist=dist,
                    correct_index=LETTERS.index(correct), n_responses=n,
                    skill="general education", source=f"ENADE {year}",
                    population=(
                        f"Brazilian graduating university students taking ENADE {year}"),
                    language="pt"))
                general_added += 1
            general_done = general_added > 0

        for pos, qnum in enumerate(specific_nums):
            correct = key_specific[pos] if pos < len(key_specific) else ""
            if (correct not in LETTERS or qnum not in parsed or
                    pos >= len(group.position_counts) or
                    key_answers.get(qnum) != correct):
                continue
            dist, n = _distribution(group.position_counts[pos])
            if n < min_responses:
                continue
            question, options = parsed[qnum]
            items.append(Item(
                id=f"enade:{year}:{group_id}:{qnum}",
                question=question, options=options, dist=dist,
                correct_index=LETTERS.index(correct), n_responses=n,
                skill=label, source=f"ENADE {year}",
                population=(
                    f"Brazilian graduating university students in {label} "
                    f"taking ENADE {year}"),
                language="pt"))

    # De-duplicate content (some areas/booklets are linked more than once).
    dedup = {}
    for item in items:
        digest = _content_hash(item)
        if digest not in dedup or item.n_responses > dedup[digest].n_responses:
            dedup[digest] = item
    items = list(dedup.values())
    group_progress.update(
        group_progress.total, part="exam groups complete", force=True)
    qc["items"] = len(items)
    qc["assets_used"] = len(used_assets)
    qc["complete"] = not any(
        "download failed" in x.get("reason", "")
        for x in qc["groups_skipped"])
    year_dir.mkdir(parents=True, exist_ok=True)
    write_items(str(cache), items)
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)
    print(f"[{year}] built {len(items)} items", flush=True)
    return items


def _content_hash(item: Item) -> str:
    body = json.dumps(
        [_ascii(item.question).lower(),
         [_ascii(x).lower() for x in item.options]],
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def _balanced_sample(items: list[Item], n: int, seed: int) -> list[Item]:
    if len(items) < n:
        raise ValueError(f"need {n} items, only {len(items)} available")
    buckets = defaultdict(list)
    for item in items:
        buckets[(item.source, item.skill)].append(item)
    rng = random.Random(seed)
    for rows in buckets.values():
        rows.sort(key=lambda x: x.id)
        rng.shuffle(rows)
    keys = list(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < n:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
    return selected


def _token_filter(items: list[Item], tokenizer_name: str, max_len: int):
    if not tokenizer_name or not max_len:
        return items, 0
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    kept = []
    progress = Progress("token-length filtering", len(items), unit="items")
    for index, item in enumerate(items, start=1):
        if len(lp.prompt_ids(tok, item)) <= max_len:
            kept.append(item)
        progress.update(index, part=item.id)
    return kept, len(items) - len(kept)


def _parse_years(spec: str) -> list[int]:
    years = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            years.update(range(a, b + 1))
        else:
            years.add(int(part))
    return sorted(years)


def _dry_run(years: list[int], enem_years: list[int]):
    micro = _discover_microdata_urls()
    total_assets = 0
    usable_years = 0
    for year in sorted(years, reverse=True):
        if year not in micro:
            print(f"{year}: no microdata URL")
            continue
        try:
            assets = _discover_exam_assets(year)
            total_assets += len(assets)
            usable_years += 1
            with_keys = sum("key" in x for x in assets.values())
            print(
                f"{year}: microdata=yes, exam areas={len(assets)}, "
                f"paired keys={with_keys}")
        except Exception as exc:  # noqa: BLE001
            print(f"{year}: exam discovery failed: {exc}")
    conservative = total_assets * 27 + usable_years * 8
    print(f"dry run: {len(years)} requested years, {total_assets} exam-area PDFs")
    print(
        f"conservative pre-filter capacity: about {conservative:,} objective items "
        "(actual retention depends on PDF text and token length)")
    if enem_years:
        try:
            questions = _enem_questions(Path("data/brazil8k_raw"))
            n_enem = sum(len(questions.get(year, [])) for year in enem_years)
            print(
                f"ENEM Challenge text-only reserve ({enem_years}): "
                f"{n_enem:,} structured questions")
        except Exception as exc:  # noqa: BLE001
            print(f"ENEM reserve discovery failed: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2004-2019,2021-2023")
    ap.add_argument("--enem-years", default="2009-2023")
    ap.add_argument("--skip-enem", action="store_true")
    ap.add_argument("--raw-dir", default="data/brazil8k_raw")
    ap.add_argument("--out-prefix", default="data/brazil8k")
    ap.add_argument("--train-items", type=int, default=8000)
    ap.add_argument("--val-items", type=int, default=200)
    ap.add_argument("--test-items", type=int, default=400)
    ap.add_argument(
        "--allow-val-year-train", action="store_true",
        help="after selecting item-disjoint validation rows, allow remaining "
             "items from that year into training; test remains strictly newer")
    ap.add_argument("--min-responses", type=int, default=100)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--download-workers", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    years = [y for y in _parse_years(args.years) if y in DEFAULT_YEARS]
    enem_years = [
        y for y in _parse_years(args.enem_years)
        if y in DEFAULT_ENEM_YEARS]
    if args.dry_run:
        _dry_run(years, enem_years)
        return

    micro = _discover_microdata_urls()
    missing = [y for y in years if y not in micro]
    if missing:
        print(f"[warn] no microdata links for years: {missing}")
    all_items = []
    # Newest years are needed for temporal val/test; then walk backward until
    # the older-year train pool has enough headroom for token filtering.
    built_years = []
    enade_plan = sorted((y for y in years if y in micro), reverse=True)
    enade_progress = Progress(
        "ENADE source build", len(enade_plan), unit="years")
    for year_index, year in enumerate(enade_plan, start=1):
        enade_progress.update(
            year_index - 1, part=f"ENADE {year}", force=year_index == 1)
        try:
            rows = build_year(
                year, Path(args.raw_dir), micro[year],
                args.min_responses, rebuild=args.rebuild,
                download_workers=args.download_workers)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {year} skipped: {exc}", flush=True)
            continue
        all_items.extend(rows)
        built_years.append(year)
        if len(built_years) >= 3:
            newest = sorted(built_years, reverse=True)
            train_pool = [x for x in all_items
                          if int(x.source.rsplit(" ", 1)[-1]) < newest[1]]
            if len(train_pool) >= int(args.train_items * 1.25):
                break
    enade_progress.update(
        enade_progress.total, part="enough source items collected", force=True)

    if not args.skip_enem and enem_years:
        structured = _enem_questions(Path(args.raw_dir))
        enem_plan = sorted(enem_years, reverse=True)
        enem_progress = Progress("ENEM source build", len(enem_plan), unit="years")
        for year_index, year in enumerate(enem_plan, start=1):
            enem_progress.update(
                year_index - 1, part=f"ENEM {year}", force=year_index == 1)
            if year not in structured:
                print(f"[warn] no structured ENEM questions for {year}")
                continue
            try:
                all_items.extend(build_enem_year(
                    year, Path(args.raw_dir), structured[year],
                    args.min_responses, rebuild=args.rebuild))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] ENEM {year} skipped: {exc}", flush=True)
        enem_progress.update(
            enem_progress.total, part="ENEM years complete", force=True)

    if len(built_years) < 3:
        raise SystemExit("fewer than three years built; cannot make temporal splits")
    # Global content de-duplication before splitting.
    unique = {}
    for item in all_items:
        unique.setdefault(_content_hash(item), item)
    all_items = list(unique.values())
    all_items, token_dropped = _token_filter(
        all_items, args.tokenizer, args.max_len)

    years_desc = sorted({
        int(x.source.rsplit(" ", 1)[-1]) for x in all_items}, reverse=True)
    test_year, val_year = years_desc[0], years_desc[1]
    test_pool = [x for x in all_items if x.source == f"ENADE {test_year}"]
    val_pool = [x for x in all_items if x.source == f"ENADE {val_year}"]
    train_pool = [x for x in all_items
                  if int(x.source.rsplit(" ", 1)[-1]) < val_year]

    val = _balanced_sample(val_pool, args.val_items, args.seed + 1)
    if args.allow_val_year_train:
        val_ids = {item.id for item in val}
        train_pool.extend(item for item in val_pool if item.id not in val_ids)
    train = _balanced_sample(train_pool, args.train_items, args.seed)
    test = _balanced_sample(test_pool, args.test_items, args.seed + 2)
    for split, rows in (("train", train), ("val", val), ("test", test)):
        for item in rows:
            item.split = split
        write_items(f"{args.out_prefix}_{split}.jsonl", rows)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "non_synthetic": True,
        "source": (
            "official INEP ENADE/ENEM microdata, released exams, and "
            "ENEM Challenge structured text"),
        "built_years": sorted(built_years),
        "train_years": sorted({
            int(x.source.rsplit(" ", 1)[-1]) for x in train}),
        "val_year": val_year,
        "test_year": test_year,
        "split_policy": (
            "newer-year test; item-disjoint validation with remaining "
            "validation-year items eligible for training"
            if args.allow_val_year_train else "strict temporal"),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "min_responses": args.min_responses,
        "max_len": args.max_len,
        "tokenizer": args.tokenizer,
        "token_length_dropped": token_dropped,
        "seed": args.seed,
        "content_sha256": {
            split: hashlib.sha256(Path(
                f"{args.out_prefix}_{split}.jsonl").read_bytes()).hexdigest()
            for split in ("train", "val", "test")
        },
    }
    with open(f"{args.out_prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"manifest -> {args.out_prefix}_manifest.json")


if __name__ == "__main__":
    main()
