from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import func
from app.database import SessionLocal
from app.models import SharedStrategy, StrategyFollow
from app.schemas import StrategyCreate
from app.auth import get_current_active_user, get_optional_user

router = APIRouter()


def _strategy_to_dict(s, follower_count=None):
    return {
        "id": s.id,
        "user_id": s.user_id,
        "name": s.name,
        "description": s.description,
        "symbols": s.symbols,
        "timeframe": s.timeframe,
        "entry_rules": s.entry_rules,
        "exit_rules": s.exit_rules,
        "is_public": s.is_public,
        "upvotes": s.upvotes,
        "total_trades": s.total_trades,
        "win_rate": s.win_rate,
        "avg_return_pct": s.avg_return_pct,
        "sharpe_ratio": s.sharpe_ratio,
        "follower_count": follower_count if follower_count is not None else 0,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


@router.get("/my")
async def list_my_strategies(
    current_user=Depends(get_current_active_user),
):
    """List strategies created by the current user."""
    db = SessionLocal()
    try:
        strategies = (
            db.query(SharedStrategy)
            .filter(SharedStrategy.user_id == current_user.id)
            .order_by(SharedStrategy.created_at.desc())
            .all()
        )
        result = []
        for s in strategies:
            fc = db.query(func.count(StrategyFollow.id)).filter(StrategyFollow.strategy_id == s.id).scalar() or 0
            result.append(_strategy_to_dict(s, fc))
        return result
    finally:
        db.close()


@router.get("/leaderboard")
def leaderboard():
    """Top 20 strategies by win_rate (min 10 trades)."""
    db = SessionLocal()
    try:
        strategies = (
            db.query(SharedStrategy)
            .filter(
                SharedStrategy.is_public == True,
                SharedStrategy.total_trades >= 10,
                SharedStrategy.win_rate.isnot(None),
            )
            .order_by(SharedStrategy.win_rate.desc())
            .limit(20)
            .all()
        )
        result = []
        for s in strategies:
            fc = db.query(func.count(StrategyFollow.id)).filter(StrategyFollow.strategy_id == s.id).scalar() or 0
            result.append(_strategy_to_dict(s, fc))
        return result
    finally:
        db.close()


@router.get("/")
def list_strategies(
    sort: str = Query("newest", regex="^(newest|upvotes|win_rate)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List public strategies with pagination and sorting."""
    db = SessionLocal()
    try:
        q = db.query(SharedStrategy).filter(SharedStrategy.is_public == True)

        if sort == "upvotes":
            q = q.order_by(SharedStrategy.upvotes.desc())
        elif sort == "win_rate":
            q = q.order_by(SharedStrategy.win_rate.desc().nullslast())
        else:
            q = q.order_by(SharedStrategy.created_at.desc())

        strategies = q.offset(offset).limit(limit).all()
        result = []
        for s in strategies:
            fc = db.query(func.count(StrategyFollow.id)).filter(StrategyFollow.strategy_id == s.id).scalar() or 0
            result.append(_strategy_to_dict(s, fc))
        return result
    finally:
        db.close()


@router.post("/")
async def create_strategy(
    data: StrategyCreate,
    current_user=Depends(get_current_active_user),
):
    """Share a new strategy."""
    db = SessionLocal()
    try:
        strategy = SharedStrategy(
            user_id=current_user.id,
            name=data.name,
            description=data.description,
            symbols=data.symbols,
            timeframe=data.timeframe,
            entry_rules=data.entry_rules,
            exit_rules=data.exit_rules,
            is_public=data.is_public,
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)
        return _strategy_to_dict(strategy, 0)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/{strategy_id}")
def get_strategy(strategy_id: int):
    """Get strategy details."""
    db = SessionLocal()
    try:
        s = db.query(SharedStrategy).filter(SharedStrategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if not s.is_public:
            raise HTTPException(status_code=403, detail="Strategy is private")
        fc = db.query(func.count(StrategyFollow.id)).filter(StrategyFollow.strategy_id == s.id).scalar() or 0
        return _strategy_to_dict(s, fc)
    finally:
        db.close()


@router.post("/{strategy_id}/upvote")
async def upvote_strategy(
    strategy_id: int,
    current_user=Depends(get_current_active_user),
):
    """Upvote a strategy."""
    db = SessionLocal()
    try:
        s = db.query(SharedStrategy).filter(SharedStrategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="Strategy not found")
        s.upvotes = (s.upvotes or 0) + 1
        db.commit()
        return {"ok": True, "upvotes": s.upvotes}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/{strategy_id}/follow")
async def follow_strategy(
    strategy_id: int,
    current_user=Depends(get_current_active_user),
):
    """Follow/copy a strategy. Toggles follow status."""
    db = SessionLocal()
    try:
        s = db.query(SharedStrategy).filter(SharedStrategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="Strategy not found")

        existing = (
            db.query(StrategyFollow)
            .filter(StrategyFollow.user_id == current_user.id, StrategyFollow.strategy_id == strategy_id)
            .first()
        )
        if existing:
            db.delete(existing)
            db.commit()
            return {"ok": True, "followed": False}
        else:
            follow = StrategyFollow(user_id=current_user.id, strategy_id=strategy_id)
            db.add(follow)
            db.commit()
            return {"ok": True, "followed": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: int,
    current_user=Depends(get_current_active_user),
):
    """Delete own strategy."""
    db = SessionLocal()
    try:
        s = db.query(SharedStrategy).filter(SharedStrategy.id == strategy_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if s.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not your strategy")
        # Delete follows first
        db.query(StrategyFollow).filter(StrategyFollow.strategy_id == strategy_id).delete()
        db.delete(s)
        db.commit()
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
