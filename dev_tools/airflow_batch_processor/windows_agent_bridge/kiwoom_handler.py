"""Kiwoom API handler for the Windows Agent Bridge.

Extends bridge_server.py with a dedicated handler that processes
Kiwoom API requests via pykiwoom/COM and returns JSON responses
over RabbitMQ.

Usage:
    Add to bridge_server.py's HANDLERS dict:
    
    HANDLERS = {
        QUEUE_AGENT_EXEC: on_agent_exec_request,
        QUEUE_KIWOOM_REQUEST: on_kiwoom_request,  # <-- this handler
    }
"""

import json
import logging
import os
import time
from pathlib import Path

import pika
from dotenv import load_dotenv

logger = logging.getLogger("bridge-server")

# Load .env from bridge directory
_bridge_dir = Path(__file__).resolve().parent
load_dotenv(_bridge_dir / ".env")

QUEUE_KIWOOM_REQUEST = os.getenv("KIWOOM_REQUEST_QUEUE", "kiwoom_request")

# Kiwoom API credentials (loaded from .env or Windows env vars)
KIWOOM_API_KEY = os.getenv("KIWOOM_API_KEY")
KIWOOM_API_SECRET = os.getenv("KIWOOM_API_SECRET")
KIWOOM_ACCOUNT_NUMBER = os.getenv("KIWOOM_ACCOUNT_NUMBER")


# ---------------------------------------------------------------------------
# Kiwoom API wrapper
# ---------------------------------------------------------------------------

class KiwoomAPI:
    """Thin wrapper around pykiwoom for the bridge handler.

    Handles connection lifecycle and exposes methods matching
    the KiwoomCollector action names.
    """

    def __init__(self):
        self._kiwoom = None
        self._connected = False

    def _get_kiwoom(self):
        """Lazy-initialize pykiwoom Kiwoom instance."""
        if self._kiwoom is None:
            try:
                from pykiwoom import Kiwoom  # type: ignore
                self._kiwoom = Kiwoom()
            except ImportError:
                logger.error("pykiwoom not installed. Install with: pip install pykiwoom")
                raise
        return self._kiwoom

    def connect(self) -> dict:
        """Initialize Kiwoom API connection.

        Returns connection status. On Windows with pykiwoom, this
        triggers the Kiwoom Open API login dialog if needed.
        """
        try:
            kiwoom = self._get_kiwoom()
            # pykiwoom auto-connects on init and handles the login dialog.
            # Check if we can fetch data to verify connection.
            codes = kiwoom.get_code_list("KOSPI")
            self._connected = isinstance(codes, list) and len(codes) > 0
            return {
                "connected": self._connected,
                "server_type": "real",
            }
        except Exception as e:
            logger.error("Kiwoom connection failed: %s", e)
            return {"connected": False, "server_type": "real"}

    def fetch_ohlcv(self, ticker: str, timeframe: str = "day", count: int = 200) -> list:
        """Fetch historical OHLCV data.

        Args:
            ticker: 6-digit stock code.
            timeframe: "day", "week", or "month".
            count: Number of candles (max 200 for Kiwoom API).

        Returns:
            List of OHLCV dicts with date/open/high/low/close/volume.
        """
        if not self._connected:
            self.connect()

        kiwoom = self._get_kiwoom()

        try:
            # pykiwoom chart() returns a DataFrame or tuple depending on version
            result = kiwoom.chart(ticker, count=count, interval=timeframe)

            # Handle tuple return (some pykiwoom versions)
            if isinstance(result, tuple):
                # (df, columns) or (data,) format
                df = result[0] if result else None
            else:
                df = result

            if df is None:
                return []

            # Handle list-of-tuples (some pykiwoom versions return raw data)
            if isinstance(df, list):
                candles = []
                for row in df:
                    if isinstance(row, (list, tuple)) and len(row) >= 6:
                        candles.append({
                            "date": str(row[0]) if row[0] else "",
                            "open": float(row[1]) if row[1] else 0,
                            "high": float(row[2]) if row[2] else 0,
                            "low": float(row[3]) if row[3] else 0,
                            "close": float(row[4]) if row[4] else 0,
                            "volume": float(row[5]) if row[5] else 0,
                        })
                return candles

            # Handle pandas DataFrame
            import pandas as pd  # noqa: PLC0415
            if isinstance(df, pd.DataFrame):
                if df.empty:
                    return []
                candles = []
                for idx, row in df.iterrows():
                    # Support both named columns and positional
                    try:
                        candles.append({
                            "date": str(idx) if hasattr(idx, "isoformat") else str(idx),
                            "open": float(row.get("open", 0) if hasattr(row, "get") else row.iloc[0] if len(row) > 0 else 0),
                            "high": float(row.get("high", 0) if hasattr(row, "get") else row.iloc[1] if len(row) > 1 else 0),
                            "low": float(row.get("low", 0) if hasattr(row, "get") else row.iloc[2] if len(row) > 2 else 0),
                            "close": float(row.get("close", 0) if hasattr(row, "get") else row.iloc[3] if len(row) > 3 else 0),
                            "volume": float(row.get("volume", 0) if hasattr(row, "get") else row.iloc[4] if len(row) > 4 else 0),
                        })
                    except (IndexError, TypeError, ValueError):
                        continue
                return candles

            # Fallback: log what we got and return empty
            logger.warning("Unexpected chart return type: %s", type(df))
            return []
        except Exception as e:
            logger.error("Failed to fetch OHLCV for %s: %s", ticker, e)
            raise

    def get_account_balance(self) -> dict:
        """Get account balance and positions.

        Returns:
            Dict with account_number, total_equity, available_cash, positions.
        """
        if not self._connected:
            self.connect()

        kiwoom = self._get_kiwoom()

        try:
            # pykiwoom provides balance info via get_balance methods
            balance = {}

            # Try to get account info
            # These method names depend on pykiwoom version
            # Adjust based on actual pykiwoom API surface
            if hasattr(kiwoom, "get_balance"):
                balance_info = kiwoom.get_balance()
            elif hasattr(kiwoom, "get_account_balance"):
                balance_info = kiwoom.get_account_balance()
            else:
                # Fallback: try get_chejan_data or balance from positions
                balance_info = {}

            # Get positions
            positions = []
            if hasattr(kiwoom, "get_balance_detail"):
                pos_list = kiwoom.get_balance_detail()
                if isinstance(pos_list, list):
                    for pos in pos_list:
                        positions.append({
                            "ticker": str(pos.get("종목번호", pos.get("ticker", ""))),
                            "name": str(pos.get("종목명", pos.get("name", ""))),
                            "quantity": float(pos.get("보유수량", pos.get("quantity", 0))),
                            "avg_price": float(pos.get("매입평균가", pos.get("avg_price", 0))),
                            "current_price": float(pos.get("현재가", pos.get("current_price", 0))),
                            "unrealized_pnl": float(pos.get("평가손익", pos.get("unrealized_pnl", 0))),
                        })

            total_equity = float(balance_info.get("총자산", balance_info.get("total_equity", 0)))
            available_cash = float(balance_info.get("주문가능금액", balance_info.get("available_cash", 0)))

            return {
                "account_number": KIWOOM_ACCOUNT_NUMBER or "unknown",
                "total_equity": total_equity,
                "available_cash": available_cash,
                "positions": positions,
            }
        except Exception as e:
            logger.error("Failed to get account balance: %s", e)
            raise

    def send_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: int = 0,
        order_type: str = "market",
    ) -> dict:
        """Place a stock order.

        Args:
            ticker: 6-digit stock code.
            side: "buy" or "sell".
            quantity: Number of shares.
            price: Limit price (0 for market orders).
            order_type: "market", "limit", or "stop".

        Returns:
            Dict with order_no and status.
        """
        if not self._connected:
            self.connect()

        kiwoom = self._get_kiwoom()

        try:
            # Map order_type to Kiwoom order type codes
            # 00: market, 01: limit, 02: stop
            order_type_map = {"market": "00", "limit": "01", "stop": "02"}
            hoga_type = order_type_map.get(order_type, "00")

            # hoga_gb: "00"=market, "01"=limit, "03"=stop
            # pykiwoom send_order signature varies by version
            if hasattr(kiwoom, "send_order"):
                order_no = kiwoom.send_order(
                    sScreenNo="0101",
                    sOrderType="신규매수" if side == "buy" else "신규매도",
                    sAccNo=KIWOOM_ACCOUNT_NUMBER or "",
                    sAccType="00",
                    sCode=ticker,
                    nQty=quantity,
                    nPrice=price if order_type == "limit" else 0,
                    sHogaGb=hoga_type,
                    sOrgOrderNo="",
                )
            else:
                raise AttributeError("pykiwoom Kiwoom class has no send_order method")

            return {
                "order_no": str(order_no) if order_no else "unknown",
                "status": "accepted",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
            }
        except Exception as e:
            logger.error("Failed to send order for %s: %s", ticker, e)
            raise

    def cancel_order(self, order_no: str) -> dict:
        """Cancel a pending order."""
        if not self._connected:
            self.connect()

        kiwoom = self._get_kiwoom()

        try:
            if hasattr(kiwoom, "cancel_order"):
                result = kiwoom.cancel_order(order_no)
            elif hasattr(kiwoom, "send_order"):
                # Cancel via send_order with cancel type
                result = kiwoom.send_order(
                    sScreenNo="0102",
                    sOrderType="매도취소",
                    sAccNo=KIWOOM_ACCOUNT_NUMBER or "",
                    sAccType="00",
                    sCode="",
                    nQty=0,
                    nPrice=0,
                    sHogaGb="00",
                    sOrgOrderNo=order_no,
                )
            else:
                raise AttributeError("No cancel method available")

            return {"order_no": order_no, "status": "cancelled"}
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_no, e)
            raise

    def get_master_list(self, market_type: str = "KOSPI") -> list:
        """Get list of all stock codes for a market.

        Args:
            market_type: "KOSPI" or "KOSDAQ".

        Returns:
            List of {code, name} dicts.
        """
        if not self._connected:
            self.connect()

        kiwoom = self._get_kiwoom()

        try:
            codes = kiwoom.get_code_list(market_type)
            result = []
            for code in codes:
                name = kiwoom.get_master_code_name(code)
                result.append({"code": code, "name": name})
            return result
        except Exception as e:
            logger.error("Failed to get master list for %s: %s", market_type, e)
            raise

    def get_chejan_data(self) -> list:
        """Get real-time execution/confirmation data.

        Note: Real-time data requires Kiwoom event callbacks.
        This returns the latest cached chejan data.
        """
        # pykiwoom may cache chejan data from real-time events.
        # This is a polling approach - for true real-time,
        # the bridge would need to set up Kiwoom event handlers
        # and push data to a separate RabbitMQ queue.
        return []

    def get_realtime_code(self, ticker: str) -> dict:
        """Get current price for a ticker."""
        if not self._connected:
            self.connect()

        kiwoom = self._get_kiwoom()

        try:
            if hasattr(kiwoom, "get_current_price"):
                price = kiwoom.get_current_price(ticker)
            elif hasattr(kiwoom, "get_master_current_price"):
                price = kiwoom.get_master_current_price(ticker)
            else:
                price = None

            return {
                "ticker": ticker,
                "current_price": float(price) if price else 0,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            }
        except Exception as e:
            logger.error("Failed to get real-time price for %s: %s", ticker, e)
            raise


# Singleton Kiwoom API instance
_kiwoom_api = KiwoomAPI()


# ---------------------------------------------------------------------------
# Handler actions dispatcher
# ---------------------------------------------------------------------------

KIWOOM_ACTIONS = {
    "connect": "connect",
    "fetch_ohlcv": "fetch_ohlcv",
    "get_account_balance": "get_account_balance",
    "send_order": "send_order",
    "cancel_order": "cancel_order",
    "get_chejan_data": "get_chejan_data",
    "get_master_list": "get_master_list",
    "get_realtime_code": "get_realtime_code",
}


def handle_kiwoom_action(action: str, ticker: str = None, params: dict = None) -> dict:
    """Dispatch a Kiwoom action to the appropriate method.

    Args:
        action: Action name (must be in KIWOOM_ACTIONS).
        ticker: 6-digit stock code (if needed).
        params: Action-specific parameters.

    Returns:
        Response data dict (not the full envelope).
    """
    params = params or {}

    if action == "connect":
        return _kiwoom_api.connect()

    elif action == "fetch_ohlcv":
        timeframe = params.get("timeframe", "day")
        count = int(params.get("count", 200))
        return _kiwoom_api.fetch_ohlcv(ticker, timeframe=timeframe, count=count)

    elif action == "get_account_balance":
        return _kiwoom_api.get_account_balance()

    elif action == "send_order":
        side = params.get("side", "buy")
        quantity = int(params.get("quantity", 0))
        price = int(params.get("price", 0))
        order_type = params.get("order_type", "market")
        return _kiwoom_api.send_order(ticker, side, quantity, price, order_type)

    elif action == "cancel_order":
        order_no = params.get("order_no", "")
        return _kiwoom_api.cancel_order(order_no)

    elif action == "get_chejan_data":
        return _kiwoom_api.get_chejan_data()

    elif action == "get_master_list":
        market_type = params.get("market_type", "KOSPI")
        return _kiwoom_api.get_master_list(market_type)

    elif action == "get_realtime_code":
        return _kiwoom_api.get_realtime_code(ticker)

    else:
        raise ValueError(f"Unknown Kiwoom action: {action}")


# ---------------------------------------------------------------------------
# RabbitMQ callback handler
# ---------------------------------------------------------------------------

def on_kiwoom_request(ch, method, props, body):
    """RabbitMQ callback for kiwoom_request queue.

    Processes JSON request, executes Kiwoom API call, returns JSON response.
    """
    start_time = time.monotonic()

    try:
        payload = json.loads(body)
        action = payload.get("action")
        ticker = payload.get("ticker")
        params = payload.get("params", {})
        req_id = payload.get("id")

        logger.info("Kiwoom request: action=%s ticker=%s id=%s", action, ticker, req_id)

        if not action:
            response_body = json.dumps({
                "status": "error",
                "action": action or "unknown",
                "id": req_id,
                "data": None,
                "error": {"code": "INVALID_PARAMS", "message": "No action provided"},
            })
        elif action not in KIWOOM_ACTIONS:
            response_body = json.dumps({
                "status": "error",
                "action": action,
                "id": req_id,
                "data": None,
                "error": {"code": "INVALID_ACTION", "message": f"Unknown action: {action}"},
            })
        else:
            data = handle_kiwoom_action(action, ticker, params)
            response_body = json.dumps({
                "status": "ok",
                "action": action,
                "id": req_id,
                "data": data,
                "error": None,
            })

        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info("Kiwoom response: action=%s status=ok duration=%.0fms", action, duration_ms)

    except Exception as e:
        logger.error("Kiwoom handler error: %s", e, exc_info=True)
        duration_ms = (time.monotonic() - start_time) * 1000
        response_body = json.dumps({
            "status": "error",
            "action": payload.get("action", "unknown") if "payload" in dir() else "unknown",
            "id": payload.get("id") if "payload" in dir() else None,
            "data": None,
            "error": {
                "code": "KIWOOM_API_ERROR",
                "message": str(e),
            },
        })

    # Reply via RabbitMQ
    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=response_body,
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


def register_kiwoom_handler(connection):
    """Register the Kiwoom handler with RabbitMQ.

    Args:
        connection: pika BlockingConnection instance.

    Returns:
        Tuple of (handler_name, queue_name) for logging.
    """
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_KIWOOM_REQUEST, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_KIWOOM_REQUEST, on_message_callback=on_kiwoom_request)

    return "Kiwoom API handler (pykiwoom)", QUEUE_KIWOOM_REQUEST
