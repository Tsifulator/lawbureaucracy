"""Decision numbers: one place that knows how ΕΑΔΗΣΥ numbers its decisions.

The PDF filename is the only reliable source for a decision's number/year, and
it comes in five shapes:

    Apofasi-1614-2025.pdf   11384   ordinary (ΟΡΙΣΤΙΚΗ κ.λπ.) -> 1614/2025
    Apofasi-A831-2025.pdf    4199   suspension (ΑΝΑΣΤΟΛΗΣ)    -> A831/2025
    Apofasi-E11-2021.pdf       53                             -> E11/2021
    Apofasi-AA55-2017.pdf      32                             -> AA55/2017
    Apofasi-Α873-2025.pdf      22   GREEK capital alpha (!)   -> A873/2025

That last shape is the trap: "A831" and "Α873" look identical on screen but are
different codepoints (U+0041 LATIN A vs U+0391 GREEK ALPHA). Everything here
normalizes to Latin uppercase, so a user who types either spelling — or copies
one out of a Greek PDF — still lands on the right decision.
"""
import re

_RE = re.compile(r"Apofasi[-_]([A-Za-zΑ-Ωα-ω]{0,2})(\d{1,5})[-_](\d{4})\.pdf", re.I)

# Greek capitals that are visually identical to a Latin letter. Only these can
# ever be confused, so only these get folded.
_HOMOGLYPHS = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})


def normalize(num: str) -> str:
    """Canonical form of a decision number: trimmed, Latin, uppercase."""
    return (num or "").strip().upper().translate(_HOMOGLYPHS)


def from_pdf_url(url: str) -> tuple[str, str]:
    """(number, year) parsed out of a decision PDF url. ('', '') if unparseable."""
    m = _RE.search(url or "")
    if not m:
        return "", ""
    prefix, digits_, year = m.groups()
    return normalize(prefix + digits_), year


def digits(num: str) -> str:
    """The numeric part alone — 'A831' -> '831'. Used by the loose keyword boost
    so that typing a bare "831" still nudges Α831/2025 up the ranking."""
    return re.sub(r"\D", "", num or "")
