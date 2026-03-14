import csv
import io
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import PortfolioHoldingCreate
from app.services.portfolio_service import portfolio_service
from app.auth import get_current_active_user

router = APIRouter()


@router.get("")
def list_holdings(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    return portfolio_service.get_holdings(db, user_id=user.id, limit=limit, offset=offset)


@router.get("/summary")
def get_summary(db: Session = Depends(get_db), user=Depends(get_current_active_user)):
    return portfolio_service.get_summary(db, user_id=user.id)


@router.get("/analytics")
def get_analytics(db: Session = Depends(get_db), user=Depends(get_current_active_user)):
    return portfolio_service.get_analytics(db, user_id=user.id)


@router.get("/export/csv")
def export_csv(
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    holdings = portfolio_service.get_holdings(db, user_id=user.id, limit=500)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Symbol", "Quantity", "Buy Price", "Current Price", "Invested", "Current Value", "P&L", "P&L %"])
    for h in holdings:
        invested = h["quantity"] * h["buy_price"]
        current_val = h["quantity"] * h["current_price"] if h["current_price"] else 0
        writer.writerow([
            h["symbol"],
            h["quantity"],
            f'{h["buy_price"]:.2f}',
            f'{h["current_price"]:.2f}' if h["current_price"] else "",
            f"{invested:.2f}",
            f"{current_val:.2f}",
            f'{h["pnl"]:.2f}' if h["pnl"] is not None else "",
            f'{h["pnl_pct"]:.2f}' if h["pnl_pct"] is not None else "",
        ])
    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="portfolio_{timestamp}.csv"'},
    )


@router.get("/export/html")
def export_html(
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    holdings = portfolio_service.get_holdings(db, user_id=user.id, limit=500)
    summary = portfolio_service.get_summary(db, user_id=user.id)
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    rows = ""
    for h in holdings:
        invested = h["quantity"] * h["buy_price"]
        current_val = h["quantity"] * h["current_price"] if h["current_price"] else 0
        pnl = h["pnl"] if h["pnl"] is not None else 0
        pnl_pct = h["pnl_pct"] if h["pnl_pct"] is not None else 0
        pnl_color = "green" if pnl >= 0 else "red"
        rows += f"""<tr>
            <td>{h["symbol"]}</td>
            <td class="num">{h["quantity"]}</td>
            <td class="num">{h["buy_price"]:.2f}</td>
            <td class="num">{h["current_price"]:.2f if h["current_price"] else "-"}</td>
            <td class="num">{invested:,.2f}</td>
            <td class="num">{current_val:,.2f}</td>
            <td class="num" style="color:{pnl_color}">{pnl:+,.2f}</td>
            <td class="num" style="color:{pnl_color}">{pnl_pct:+.2f}%</td>
        </tr>"""

    pnl_color = "green" if summary["total_pnl"] >= 0 else "red"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Portfolio Report - {timestamp}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #1a1a2e; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .summary {{ display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }}
  .summary-card {{ background: #f5f5f5; border-radius: 8px; padding: 12px 18px; }}
  .summary-card .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
  .summary-card .value {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #ddd; font-size: 11px; text-transform: uppercase; color: #888; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  th.num {{ text-align: right; }}
  @media print {{ body {{ padding: 0; }} }}
</style></head><body>
<h1>Portfolio Report</h1>
<div class="subtitle">Generated on {timestamp}</div>
<div class="summary">
  <div class="summary-card"><div class="label">Total Invested</div><div class="value">&#8377;{summary["total_invested"]:,.2f}</div></div>
  <div class="summary-card"><div class="label">Current Value</div><div class="value">&#8377;{summary["current_value"]:,.2f}</div></div>
  <div class="summary-card"><div class="label">Total P&amp;L</div><div class="value" style="color:{pnl_color}">{summary["total_pnl"]:+,.2f} ({summary["total_pnl_pct"]:+.2f}%)</div></div>
  <div class="summary-card"><div class="label">Holdings</div><div class="value">{summary["holdings_count"]}</div></div>
</div>
<table>
  <thead><tr>
    <th>Symbol</th><th class="num">Qty</th><th class="num">Buy Price</th><th class="num">CMP</th>
    <th class="num">Invested</th><th class="num">Current</th><th class="num">P&amp;L</th><th class="num">P&amp;L %</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</body></html>"""
    return HTMLResponse(content=html)


@router.post("")
def add_holding(
    data: PortfolioHoldingCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    holding = portfolio_service.add_holding(db, data.model_dump(), user_id=user.id)
    return {"id": holding.id, "message": "Holding added successfully"}


@router.delete("/{holding_id}")
def delete_holding(
    holding_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    success = portfolio_service.delete_holding(db, holding_id, user_id=user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Holding not found")
    return {"message": "Holding deleted"}
