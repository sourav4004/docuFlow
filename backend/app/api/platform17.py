"""Phase 17 platform API — governance (policy hierarchy, retention, legal
holds), operational alerts, cost anomaly/forecast/export, and search planning.
Org-scoped routes require org-admin; workspace routes require membership."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..models.organization import Organization
from ..services.permission_service import (
    require_workspace_membership, get_organization_role,
)
from ..services import governance2 as gov, alerts as alert_svc, cost2, search3

router = APIRouter(tags=["platform17"])


def _ws(db: Session, workspace_id: int, user_id: int) -> Workspace:
    return require_workspace_membership(db, workspace_id, user_id)


def _admin_gate(db: Session, principal: AuthPrincipal,
                workspace_id: Optional[int] = None) -> None:
    if workspace_id is None:
        raise HTTPException(status_code=403,
                            detail="Workspace scope required")
    ws = _ws(db, workspace_id, principal.user.id)
    if ws.organization_id is not None:
        role = get_organization_role(db, ws.organization_id,
                                     principal.user.id)
        if role in ("OWNER", "ADMIN"):
            return
        raise HTTPException(status_code=403,
                            detail="Organization admin required")
    if ws.owner_id == principal.user.id:
        return
    raise HTTPException(status_code=403, detail="Workspace owner required")


# ---------------------------------------------------------------------------
# Governance — AI policy rules
# ---------------------------------------------------------------------------

class PolicyRuleBody(BaseModel):
    organization_id: Optional[int] = None
    workspace_id: Optional[int] = None
    feature: Optional[str] = None
    rule_type: str
    allowlist: Optional[list[str]] = None
    deny: Optional[list[str]] = None
    sensitivity_max: Optional[str] = None
    budget_max_usd: Optional[float] = None


@router.post("/governance/rules")
def create_policy_rule(body: PolicyRuleBody,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    if body.organization_id is not None:
        role = get_organization_role(db, body.organization_id,
                                     principal.user.id)
        if role not in ("OWNER", "ADMIN"):
            raise HTTPException(status_code=403,
                                detail="Organization admin required")
    if body.workspace_id is not None and body.organization_id is None:
        _admin_gate(db, principal, body.workspace_id)
    try:
        rule = gov.create_rule(
            db, organization_id=body.organization_id,
            workspace_id=body.workspace_id, feature=body.feature,
            rule_type=body.rule_type, allowlist=body.allowlist,
            deny=body.deny, sensitivity_max=body.sensitivity_max,
            budget_max_usd=body.budget_max_usd,
            created_by=principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"rule_id": rule.id, "rule_type": rule.rule_type}


@router.get("/governance/rules")
def list_policy_rules(workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    return gov.list_rules(db, organization_id=ws.organization_id,
                          workspace_id=workspace_id)


@router.get("/governance/check-model")
def check_model(workspace_id: int, feature: str = "rag", model: str = "",
                sensitivity: str = "INTERNAL",
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    ok, reason = gov.check_model_allowed(
        db, organization_id=ws.organization_id, workspace_id=workspace_id,
        feature=feature, model=model, sensitivity=sensitivity)
    if not ok:
        raise HTTPException(status_code=403, detail=reason)
    return {"allowed": True, "model": model, "reason": reason}


# ---------------------------------------------------------------------------
# Retention + legal holds
# ---------------------------------------------------------------------------

class RetentionBody(BaseModel):
    entity_type: str
    retention_days: int


@router.post("/governance/retention")
def set_retention(workspace_id: int, body: RetentionBody,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    try:
        assignment = gov.set_retention(
            db, entity_type=body.entity_type,
            retention_days=body.retention_days,
            organization_id=ws.organization_id,
            workspace_id=workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"assignment_id": assignment.id,
            "retention_days": assignment.retention_days}


@router.post("/governance/cleanup")
def run_cleanup(workspace_id: int, entity_type: str,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    result = gov.run_cleanup(db, entity_type=entity_type,
                             workspace_id=workspace_id,
                             organization_id=ws.organization_id)
    db.commit()
    return result


class HoldBody(BaseModel):
    name: str
    reason: str = ""
    entity_type: str = "document"
    entity_ids: Optional[list[int]] = None


@router.post("/legal-holds")
def place_hold(workspace_id: int, body: HoldBody,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    hold = gov.place_hold(db, workspace_id=workspace_id,
                          organization_id=ws.organization_id,
                          name=body.name, reason=body.reason,
                          entity_type=body.entity_type,
                          entity_ids=body.entity_ids,
                          started_by=principal.user.id)
    db.commit()
    return {"hold_id": hold.id, "status": hold.status}


@router.post("/legal-holds/{hold_id}/release")
def release_hold(hold_id: int, workspace_id: int,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    try:
        result = gov.release_hold(db, hold_id, workspace_id,
                                  principal.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return result


@router.get("/legal-holds")
def list_holds(workspace_id: int,
               principal: AuthPrincipal = Depends(get_current_principal),
               db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    from ..models.phase17 import LegalHold
    rows = db.query(LegalHold).filter(
        LegalHold.workspace_id == workspace_id).order_by(
        LegalHold.id.desc()).limit(100).all()
    return {"items": [
        {"id": h.id, "name": h.name, "status": h.status,
         "started_at": h.started_at, "released_at": h.released_at}
        for h in rows], "total": len(rows)}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertRuleBody(BaseModel):
    name: str
    metric: str
    operator: str = ">"
    threshold: float
    severity: str = "WARNING"
    cooldown_minutes: int = 30


@router.post("/alerts/rules")
def create_alert_rule(workspace_id: int, body: AlertRuleBody,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    ws = _ws(db, workspace_id, principal.user.id)
    try:
        rule = alert_svc.create_rule(
            db, name=body.name, metric=body.metric, operator=body.operator,
            threshold=body.threshold, severity=body.severity,
            cooldown_minutes=body.cooldown_minutes,
            workspace_id=workspace_id,
            organization_id=ws.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"rule_id": rule.id}


@router.get("/alerts/rules")
def list_alert_rules(workspace_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    return alert_svc.list_rules(db, workspace_id=workspace_id)


class EvaluateBody(BaseModel):
    metrics: dict


@router.post("/alerts/evaluate")
def evaluate_alerts(workspace_id: int, body: EvaluateBody,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    fired = alert_svc.evaluate(db, body.metrics)
    db.commit()
    return {"fired": fired}


@router.get("/alerts/events")
def list_alert_events(workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    return alert_svc.list_events(db, workspace_id=workspace_id)


# ---------------------------------------------------------------------------
# Cost + usage
# ---------------------------------------------------------------------------

@router.get("/cost/anomalies")
def cost_anomalies(workspace_id: Optional[int] = None,
                   organization_id: Optional[int] = None,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    if organization_id is not None:
        _org_admin(principal, organization_id, db)
    elif workspace_id is not None:
        _ws(db, workspace_id, principal.user.id)
    anomalies = cost2.detect_anomalies(
        db, organization_id=organization_id, workspace_id=workspace_id)
    return {"items": anomalies, "labeled": "estimates — verify before action"}


@router.get("/cost/forecast")
def cost_forecast(workspace_id: Optional[int] = None,
                  organization_id: Optional[int] = None,
                  principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    if organization_id is not None:
        _org_admin(principal, organization_id, db)
    elif workspace_id is not None:
        _ws(db, workspace_id, principal.user.id)
    return cost2.forecast(db, organization_id=organization_id,
                          workspace_id=workspace_id)


@router.get("/usage/export.csv")
def usage_export_csv(organization_id: int, workspace_id: Optional[int] = None,
                     days: int = 90,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    from fastapi.responses import PlainTextResponse
    _org_admin(principal, organization_id, db)
    csv_text = cost2.export_usage_csv(
        db, organization_id=organization_id, workspace_id=workspace_id,
        days=days)
    return PlainTextResponse(csv_text, media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=usage.csv"})


def _org_admin(principal: AuthPrincipal, org_id: int,
               db: Session) -> None:
    role = get_organization_role(db, org_id, principal.user.id)
    if role not in ("OWNER", "ADMIN"):
        raise HTTPException(status_code=403,
                            detail="Organization admin required")


# ---------------------------------------------------------------------------
# Search planning + explainability
# ---------------------------------------------------------------------------

@router.get("/search/plan")
def search_plan(query: str,
                principal: AuthPrincipal = Depends(get_current_principal),
                db: Session = Depends(get_db)):
    if not principal:
        raise HTTPException(status_code=401, detail="Authentication required")
    return search3.plan_query(query)


@router.post("/search/saved/{saved_search_id}/check")
def saved_search_check(saved_search_id: int, workspace_id: int,
                       principal: AuthPrincipal = Depends(get_current_principal),
                       db: Session = Depends(get_db)):
    _ws(db, workspace_id, principal.user.id)
    def evaluator(db2, ws_id, query):
        from ..models.search_intel import SavedSearch
        from ..models.document import Document
        rec = db2.query(SavedSearch).filter(
            SavedSearch.id == saved_search_id).first()
        if rec is None:
            return []
        # Deterministic: any matching-ish documents count as a change signal.
        q = (query or "").lower()
        docs = db2.query(Document.id).filter(
            Document.workspace_id == ws_id,
            Document.original_filename.ilike(f"%{q[:80]}%")).limit(10).all()
        return [d[0] for d in docs]
    try:
        result = search3.run_saved_search_check(
            db, saved_search_id=saved_search_id,
            workspace_id=workspace_id, alert_evaluator=evaluator)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return result
