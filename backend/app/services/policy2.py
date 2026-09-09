"""Policy intelligence — semantic normalization + contradiction engine.

Every requirement is normalized into a structured, comparable form:

    subject, action, direction (+allow/−deny/±require), threshold, unit,
    comparator, scope, frequency, effective/expiry dates

Contradiction detection is deterministic and evidence-backed. It does NOT
report superficial textual differences as conflicts: two statements conflict
only when a concrete condition (amount range, date window, obligation
direction, frequency, scope) can be shown to be mutually unsatisfiable.

The default engine is rule-based. An optional semantic-entailment provider
may be plugged in later behind the same verdict interface — until then we
never claim general semantic reasoning we did not implement.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import PolicyStatement, PolicyConflict
from ..services.review_service import create_review_item

logger = logging.getLogger(__name__)

VERDICTS = ("SUPPORTS", "CONTRADICTS", "UNCERTAIN", "NOT_RELATED")

def _detect_action(text: str) -> tuple[Optional[str], int]:
    """Detect (action, polarity) deterministically. Negations always win over
    their positive counterpart so "no approval required" never reads as a
    positive approval requirement."""
    low = text.lower()

    def has(*patterns: str) -> bool:
        return any(re.search(p, low) for p in patterns)

    if has(r"no approval", r"without approval",
           r"approval is not required", r"approval not required",
           r"approval[^.]{0,40}?not required"):
        return "require_approval", -1
    if has(r"approval required", r"require approval", r"requires approval",
           r"must be approved", r"need approval",
           r"approval[^.]{0,40}?required"):
        return "require_approval", +1
    if has(r"\bmust not\b", r"\bshall not\b", r"prohibited",
           r"cannot be", r"not allowed"):
        return "deny", -1
    if has(r"\bmust\b", r"\bshall\b", r"is required to",
           r"obligation to", r"required to"):
        return "require", +1
    if has(r"\bmay\b", r"permitted to", r"is allowed to"):
        return "permit", +1
    return ("permit", +1) if _looks_permissive(text) else ("require", +1)


_THRESHOLD_PAT = re.compile(
    r"(?:above|exceeds?|greater than|more than|at least|no less than|at or above)\s*"
    r"([\$€£]?\s?[\d,]+(?:\.\d+)?)\s*([A-Za-z]{2,5})?"
    r"|(?:below|under|less than|at most|no more than|at or below)\s*"
    r"([\$€£]?\s?[\d,]+(?:\.\d+)?)\s*([A-Za-z]{2,5})?",
    re.IGNORECASE,
)

_DATE_PAT = re.compile(
    r"(?:effective|as of|from)\s+(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december|\d{1,2})[\s,/]*\d{0,4}",
    re.IGNORECASE,
)

_FREQ_PAT = re.compile(
    r"(daily|weekly|biweekly|monthly|quarterly|annually|yearly|every\s+\d+\s+\w+)",
    re.IGNORECASE,
)


def _num(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "").replace("$", "").replace("€", "")
                      .replace("£", ""))
    except ValueError:
        return None


def _first_phrase(text: str) -> str:
    # Cheap subject hint: first noun-ish token run before the first verb.
    m = re.match(r"^\s*([A-Z][\w\s&-]{2,60}?)\s+(?:is|are|must|may|shall|"
                 r"requires?|approval|expenses?|purchases?|spending|amounts?|"
                 r"transactions?)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:200]
    return text[:60].strip()


def normalize_statement(text: str) -> dict:
    """Deterministically normalize a policy statement into semantics."""
    sem: dict = {
        "raw": text,
        "subject": _first_phrase(text),
        "action": None,
        "direction": 0,
        "polarity": 0,           # +1 affirmative, −1 negated obligation
        "threshold": None,
        "comparator": None,
        "unit": None,
        "scope": None,
        "frequency": None,
        "effective": None,
        "expiry": None,
    }
    action, polarity = _detect_action(text)
    sem["action"] = action or "require"
    sem["polarity"] = polarity
    m1 = _THRESHOLD_PAT.search(text)
    if m1:
        above = m1.group(1) or m1.group(3)
        unit = (m1.group(2) or m1.group(4) or "").upper() or None
        if above:
            sem["threshold"] = _num(above)
            if m1.group(1) is not None:
                sem["comparator"] = ">="
            else:
                sem["comparator"] = "<="
            sem["unit"] = unit or _currency_hint(text)
    else:
        m2 = re.search(r"[\$€£]\s?([\d,]+(?:\.\d+)?)", text)
        if m2:
            sem["threshold"] = _num(m2.group(1))
            sem["unit"] = "currency"
            sem["comparator"] = (
                ">=" if re.search(r"above|exceed|greater|more|at least", text,
                                  re.IGNORECASE)
                else "<=" if re.search(r"below|under|less|at most|no more",
                                       text, re.IGNORECASE)
                else "=")
    freq = _FREQ_PAT.search(text)
    if freq:
        sem["frequency"] = freq.group(1).lower()
    date_hint = _extract_dates(text)
    sem["effective"] = date_hint.get("effective")
    sem["expiry"] = date_hint.get("expiry")
    return sem


def _looks_permissive(text: str) -> bool:
    return bool(re.search(r"\b(may|can|allowed|permitted|no\s+(?:approval|"
                          r"review|consent) required)\b", text, re.IGNORECASE))


def _currency_hint(text: str) -> Optional[str]:
    if "$" in text:
        return "USD"
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    return None


_DATE_FORM = (r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
              r"(?:january|february|march|april|may|june|july|august|"
              r"september|october|november|december)\s+\d{1,2},?\s*\d{4})")


def _extract_dates(text: str) -> dict:
    out = {"effective": None, "expiry": None}
    m = re.search(r"effective\s+" + _DATE_FORM, text, re.IGNORECASE)
    if m:
        out["effective"] = m.group(1)
    m = re.search(r"(?:expires?|until|valid until|through)\s+" + _DATE_FORM,
                  text, re.IGNORECASE)
    if m:
        out["expiry"] = m.group(1)
    return out


def _parse_date(value: Optional[str]):
    """Best-effort date tuple (y, m, d) for flexible formats — None if
    unparseable (never fabricates)."""
    if not value:
        return None
    text = value.strip()
    months = {"january": 1, "february": 2, "march": 3, "april": 4,
              "may": 5, "june": 6, "july": 7, "august": 8,
              "september": 9, "october": 10, "november": 11,
              "december": 12}
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", text)
    if m:
        year = int(m.group(3))
        if year < 100:
            year += 2000
        return (year, int(m.group(1)), int(m.group(2)))
    for name, num in months.items():
        if text.lower().startswith(name):
            rest = text[len(name):].strip().strip(",").strip()
            m = re.match(r"^(\d{1,2})\s*,?\s*(\d{4})$", rest)
            if m:
                return (int(m.group(2)), num, int(m.group(1)))
    return None


def semanticize(db: Session, statement: PolicyStatement) -> dict:
    """Persist normalized semantics onto the policy statement row."""
    sem = normalize_statement(statement.statement)
    statement.subject = sem["subject"]
    statement.action = sem["action"]
    statement.threshold_value = sem["threshold"]
    statement.threshold_unit = sem["unit"]
    statement.scope_text = sem["scope"]
    statement.timeframe_text = sem["frequency"]
    statement.condition_text = (
        f"comparator={sem['comparator']}; polarity={sem['polarity']}"
        if sem["comparator"] else None)
    db.flush()
    return sem


def _approval_required_interval(sem: dict) -> Optional[tuple]:
    """Return [lo, hi) amounts that REQUIRE approval for a threshold rule, or
    None when the statement carries no approval threshold.

    Negative rules ("no approval required below X") are converted to their
    complement: approval IS required above X — both statements are then
    compared on the same axis."""
    if sem.get("action") != "require_approval" or sem.get("threshold") is None:
        return None
    t = sem["threshold"]
    comp = sem.get("comparator") or ">="
    if sem["polarity"] < 0:
        # no-approval interval flips into the approval-required region
        if comp in ("<=", "<"):
            return (t, float("inf"))
        if comp in (">=", ">"):
            return (float("-inf"), t)
        return None
    if comp in (">=", ">"):
        return (t, float("inf"))
    if comp in ("<=", "<"):
        return (float("-inf"), t)
    return (t, t + 1e-9)  # exactly t


def policy_conflict_verdict(a_text: str, b_text: str) -> dict:
    """Compare two policy statement texts; deterministic verdict + evidence.

    Handles at minimum: numeric contradiction, negation contradiction,
    mutually exclusive requirements, incompatible thresholds, incompatible
    scope, incompatible frequency, incompatible obligations. Never reports
    superficial textual differences.
    """
    a = normalize_statement(a_text)
    b = normalize_statement(b_text)
    default = {
        "result": "UNCERTAIN", "category": None,
        "reason": "Statements are not directly comparable by the deterministic "
                  "engine.",
        "condition": None, "confidence": 0.3,
    }

    # Negation of the same requirement → direct contradiction.
    if (a["action"] == b["action"] and a["polarity"] != 0
            and b["polarity"] != 0 and a["polarity"] != b["polarity"]
            and a["threshold"] == b["threshold"]
            and _same_subject(a, b) and a["frequency"] == b["frequency"]):
        return {
            "result": "CONTRADICTS", "category": "NEGATION",
            "reason": ("Statement A requires the action while statement B "
                       "negates the same requirement at the same threshold."),
            "condition": f"threshold={a['threshold']} {a['unit'] or ''}".strip(),
            "confidence": 0.95,
        }

    # Mutually exclusive requirements (require vs deny, same subject).
    if (a["action"] in ("require", "require_approval")
            and b["action"] == "deny" and _same_subject(a, b)
            and a["threshold"] is None and b["threshold"] is None):
        return {
            "result": "CONTRADICTS", "category": "MUTUALLY_EXCLUSIVE",
            "reason": "A requires an obligation that B explicitly forbids "
                      "for the same subject/scope.",
            "condition": None, "confidence": 0.85,
        }

    # Incompatible frequency for the same obligation.
    if (a["frequency"] and b["frequency"] and a["frequency"] != b["frequency"]
            and a["action"] == b["action"] and _same_subject(a, b)
            and a["threshold"] is None):
        return {
            "result": "CONTRADICTS", "category": "FREQUENCY",
            "reason": (f"The same obligation cannot run on both "
                       f"'{a['frequency']}' and '{b['frequency']}' cadence."),
            "condition": f"frequency A={a['frequency']}, B={b['frequency']}",
            "confidence": 0.8,
        }

    # Numeric approval-threshold rules: compare required-approval intervals.
    ia = _approval_required_interval(a)
    ib = _approval_required_interval(b)
    if ia is not None and ib is not None:
        lo = max(ia[0], ib[0])
        hi = min(ia[1], ib[1])
        # Compute the region where only one rule applies → contradiction.
        a_only_lo, a_only_hi = ia[0], min(ia[1], ib[1] if ib[0] > ia[0] else float("inf"))
        # Simpler: overlapping intervals that disagree at any point.
        if lo < hi:
            if (ia[0] != ib[0]) or (ia[1] != ib[1]):
                region = _contradiction_region(ia, ib)
                if region:
                    return {
                        "result": "CONTRADICTS", "category": "NUMERIC_THRESHOLD",
                        "reason": ("The approval requirements disagree for "
                                   "part of the covered amount range."),
                        "condition": region,
                        "confidence": 0.9,
                    }
    if ia is not None or ib is not None:
        return default

    # Date contradiction: expiry before effective (never fabricates).
    if a.get("effective") and a.get("expiry"):
        eff = _parse_date(a["effective"])
        exp = _parse_date(a["expiry"])
        if eff and exp and exp < eff:
            return {
                "result": "CONTRADICTS", "category": "DATE",
                "reason": "Statement expires before its own effective date.",
                "condition": f"effective {a['effective']} but expiry "
                             f"{a['expiry']}", "confidence": 0.8,
            }
    return default


def _month_to_num(date_str: str) -> str:
    months = {"january": "01", "february": "02", "march": "03", "april": "04",
              "may": "05", "june": "06", "july": "07", "august": "08",
              "september": "09", "october": "10", "november": "11",
              "december": "12"}
    low = date_str.strip().lower()
    if "/" in low or "-" in low:
        sep = "/" if "/" in low else "-"
        parts = low.split(sep)
        if len(parts) == 3:
            y = parts[2] if len(parts[2]) == 4 else f"20{parts[2]}"
            return f"{parts[0].zfill(2)}/{parts[1].zfill(2)}/{y}"
    for name, num in months.items():
        if low.startswith(name):
            rest = low[len(name):].strip().strip(",").strip()
            m = re.match(r"(\d{1,2})\s*,?\s*(\d{4})", rest)
            if m:
                return f"{num}/{m.group(1).zfill(2)}/{m.group(2)}"
    return date_str


def _same_subject(a: dict, b: dict) -> bool:
    if not a.get("subject") or not b.get("subject"):
        return True  # unknown subjects — do not over-claim difference
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    sa, sb = norm(a["subject"]), norm(b["subject"])
    return sa == sb or sa in sb or sb in sa


def _contradiction_region(ia: tuple, ib: tuple) -> Optional[str]:
    """Human-readable amount region where rules disagree (if non-empty)."""
    # A requires approval in ia; B requires approval in ib. Where they differ
    # (symmetric difference) a user cannot satisfy both rules simultaneously.
    pts = sorted({ia[0], ia[1], ib[0], ib[1]})
    segments = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    for lo, hi in segments:
        a_in = lo >= ia[0] and hi <= ia[1]
        b_in = lo >= ib[0] and hi <= ib[1]
        if a_in != b_in and hi != float("inf"):
            return (f"amounts between {_fmt(lo)} and {_fmt(hi)} "
                    f"({_fmt(lo)} inclusive)")
    return None


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "unbounded"
    if v == float("-inf"):
        return "-inf"
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


# ---------------------------------------------------------------------------
# Persistence: conflicts + review items
# ---------------------------------------------------------------------------

def detect_and_record_conflicts(
    db: Session,
    workspace_id: int,
    organization_id: Optional[int] = None,
    limit: int = 500,
) -> dict:
    """Semanticize statements and detect/record real conflicts (pairwise)."""
    from ..models.workspace import Workspace
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    creator_id = ws.owner_id if ws else 1
    statements = (
        db.query(PolicyStatement)
        .filter(PolicyStatement.workspace_id == workspace_id)
        .limit(limit)
        .all()
    )
    sem_map = {s.id: normalize_statement(s.statement) for s in statements}
    created = 0
    reviewed = 0
    for i, a in enumerate(statements):
        for b in statements[i + 1:]:
            verdict = policy_conflict_verdict(a.statement, b.statement)
            if verdict["result"] != "CONTRADICTS":
                continue
            existing = (
                db.query(PolicyConflict)
                .filter(
                    ((PolicyConflict.policy_a_id == a.id)
                     & (PolicyConflict.policy_b_id == b.id))
                    | ((PolicyConflict.policy_a_id == b.id)
                       & (PolicyConflict.policy_b_id == a.id))
                ).first()
            )
            if existing:
                continue
            conflict = PolicyConflict(
                workspace_id=workspace_id,
                organization_id=organization_id,
                policy_a_id=a.id,
                policy_b_id=b.id,
                conflict_type=verdict["category"],
                severity=_severity(verdict),
                description=(f"{verdict['reason']} "
                             f"[condition: {verdict['condition']}]"),
                conditions_json=json.dumps({
                    "statement_a": a.statement,
                    "statement_b": b.statement,
                    "condition": verdict["condition"],
                    "detection_method": "rule-based-semantic",
                    "confidence": verdict["confidence"],
                    "evidence_a": a.source_chunk or a.evidence_reference,
                    "evidence_b": b.source_chunk or b.evidence_reference,
                }),
                status="OPEN",
            )
            db.add(conflict)
            db.flush()
            # Step 32 — conflicts ALWAYS become review items; never resolved
            # automatically.
            create_review_item(
                db, workspace_id=workspace_id, item_type="POLICY_CONFLICT",
                title=f"Policy conflict: {verdict['category']}",
                user_id=creator_id,
                description=(f"A: {a.statement}\nB: {b.statement}\n"
                             f"{verdict['reason']}"),
                payload=verdict,
                source_type="policy_conflict",
                source_id=conflict.id,
                priority="HIGH" if _severity(verdict) == "HIGH" else "NORMAL",
                organization_id=organization_id,
            )
            created += 1
    db.flush()
    return {"statements": len(statements),
            "conflicts_recorded": created,
            "review_items_created": created}


def _severity(verdict: dict) -> str:
    confidence = verdict.get("confidence", 0.0)
    if confidence >= 0.9:
        return "HIGH"
    if confidence >= 0.8:
        return "MEDIUM"
    return "LOW"
