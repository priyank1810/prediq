import logging
import time

logger = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_OC_URL = NSE_BASE + "/api/option-chain-equities?symbol={symbol}"
CACHE_TTL = 300  # 5 minutes


class OIService:
    def __init__(self):
        self._session = None
        self._session_time = 0
        self._cache = {}  # symbol -> (timestamp, result)

    def _create_session(self):
        """Create a curl_cffi session with NSE cookies."""
        try:
            from curl_cffi import requests as cffi_requests
            session = cffi_requests.Session(impersonate="chrome110")
            resp = session.get(NSE_BASE, timeout=15)
            if resp.status_code == 200:
                self._session = session
                self._session_time = time.time()
                return True
        except ImportError:
            logger.warning("OIService: curl_cffi not installed")
        except Exception as e:
            logger.warning(f"OIService: session init failed: {e}")
        return False

    def _ensure_session(self):
        """Ensure session is alive (refresh every 4 minutes)."""
        if self._session and (time.time() - self._session_time) < 240:
            return True
        return self._create_session()

    def get_oi_analysis(self, symbol: str) -> dict:
        """Fetch NSE option chain and compute OI analysis."""
        default = {"score": 0, "available": False}

        # Check cache
        if symbol in self._cache:
            ts, result = self._cache[symbol]
            if time.time() - ts < CACHE_TTL:
                return result

        if not self._ensure_session():
            return default

        try:
            url = NSE_OC_URL.format(symbol=symbol)
            resp = self._session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"OI fetch failed for {symbol}: HTTP {resp.status_code}")
                return default

            data = resp.json()
            records = data.get("records", {})
            oc_data = records.get("data", [])

            if not oc_data:
                return default

            # Compute PCR, Max Pain, OI change
            total_call_oi = 0
            total_put_oi = 0
            total_call_oi_change = 0
            total_put_oi_change = 0
            strike_pain = {}  # strike -> total pain

            underlying = records.get("underlyingValue", 0)

            for row in oc_data:
                strike = row.get("strikePrice", 0)
                ce = row.get("CE", {})
                pe = row.get("PE", {})

                ce_oi = ce.get("openInterest", 0) or 0
                pe_oi = pe.get("openInterest", 0) or 0
                ce_oi_chg = ce.get("changeinOpenInterest", 0) or 0
                pe_oi_chg = pe.get("changeinOpenInterest", 0) or 0

                total_call_oi += ce_oi
                total_put_oi += pe_oi
                total_call_oi_change += ce_oi_chg
                total_put_oi_change += pe_oi_chg

                # Max Pain calculation: for each strike, sum ITM option buyer losses
                # At expiry price = strike: calls above strike expire worthless, puts below strike expire worthless
                strike_pain[strike] = 0
                for inner_row in oc_data:
                    inner_strike = inner_row.get("strikePrice", 0)
                    inner_ce_oi = (inner_row.get("CE", {}).get("openInterest", 0) or 0)
                    inner_pe_oi = (inner_row.get("PE", {}).get("openInterest", 0) or 0)
                    # Call buyer loss if expiry < their strike (they lose premium, approx as OI * distance)
                    if inner_strike < strike:
                        strike_pain[strike] += inner_pe_oi * (strike - inner_strike)
                    elif inner_strike > strike:
                        strike_pain[strike] += inner_ce_oi * (inner_strike - strike)

            # PCR
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

            # Max Pain: strike with maximum total pain to option buyers
            max_pain = max(strike_pain, key=strike_pain.get) if strike_pain else underlying

            # PCR scoring
            if pcr > 1.5:
                pcr_score = 50  # Strongly bullish
            elif pcr > 1.0:
                pcr_score = 20  # Mildly bullish
            elif pcr > 0.7:
                pcr_score = -20  # Mildly bearish
            else:
                pcr_score = -50  # Strongly bearish

            # OI change modifier
            oi_change_score = 0
            if total_put_oi_change > total_call_oi_change:
                oi_change_score = 20  # Put OI increasing more = bullish
            elif total_call_oi_change > total_put_oi_change:
                oi_change_score = -20  # Call OI increasing more = bearish

            total_score = max(-100, min(100, pcr_score + oi_change_score))

            # Signal description
            if total_score > 20:
                oi_signal = "Bullish OI setup"
            elif total_score < -20:
                oi_signal = "Bearish OI setup"
            else:
                oi_signal = "Neutral OI"

            result = {
                "score": total_score,
                "available": True,
                "pcr": round(pcr, 2),
                "max_pain": round(max_pain, 2),
                "oi_signal": oi_signal,
                "call_oi_change": total_call_oi_change,
                "put_oi_change": total_put_oi_change,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
            }
            self._cache[symbol] = (time.time(), result)
            return result

        except Exception as e:
            logger.warning(f"OI analysis failed for {symbol}: {e}")
            return default


oi_service = OIService()
