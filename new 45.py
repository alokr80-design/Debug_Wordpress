#!/usr/bin/env python3
"""
Enhanced OpenAlgo Options Trading Strategy
- PCR and OI-based entry signals with thresholds
- Live futures price and VWAP data
- Multi-condition exit logic with P&L targets
- Daily loss limits and trading time restrictions
- Direct OpenAlgo REST API integration (no external library dependency)
- Clean, maintainable code structure
"""

import os
import time
import json
import logging
import requests
import mysql.connector
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# ======================
# CONFIGURATION
# ======================

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APIKEY_FILE = os.path.join(BASE_DIR, "openalgo_apikey.txt")
    STATE_FILE = os.path.join(BASE_DIR, "strategy_state.json")
    
    # OpenAlgo settings
    OPENALGO_BASE = os.getenv("OPENALGO_BASE", "https://pcrtrade.ddns.net:8443")
    
    # Futures API
    FUTURES_API_URL = "https://pcrtrade.ddns.net/api/futures/live"
    
    # Database settings
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "34.10.22.147"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "wp_user"),
        "password": os.getenv("DB_PASSWORD", "Syntel789*"),
        "database": os.getenv("DB_NAME", "wordpress_db"),
        "autocommit": True,
    }
    
    # Trading parameters
    UNDERLYING = "NIFTY"
    QTY = 75
    ENTRY_START = "11:00"
    EXIT_END = "15:15"
    VWAP_BAND = 10.0
    PCR_CALL_MIN = 1.2
    PCR_PUT_MAX = 1.0
    PCR_CHANGE_THRESHOLD = 0.05  # 5%
    OI_CHANGE_THRESHOLD = 0.10   # 10%
    TARGET_PNL = 1400.0
    STOP_PNL = -1000.0
    MAX_DAILY_LOSS = -1000.0
    POLL_INTERVAL = 60
    
    # Timezone
    IST = ZoneInfo("Asia/Kolkata")

# ======================
# LOGGING
# ======================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# DATA STRUCTURES
# ======================

@dataclass
class MarketData:
    timestamp: datetime
    price: float
    vwap: float
    pcr: float
    oi_diff: float
    prev_pcr: Optional[float] = None
    prev_oi_diff: Optional[float] = None
    
    @property
    def pcr_change(self) -> Optional[float]:
        if self.prev_pcr is None or self.prev_pcr == 0:
            return None
        return (self.pcr - self.prev_pcr) / abs(self.prev_pcr)
    
    @property
    def oi_change(self) -> Optional[float]:
        if self.prev_oi_diff is None or self.prev_oi_diff == 0:
            return None
        return (self.oi_diff - self.prev_oi_diff) / abs(self.prev_oi_diff)

@dataclass
class TradePosition:
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    entry_pcr: float
    entry_oi_diff: float
    order_id: Optional[str] = None

# ======================
# UTILITY FUNCTIONS
# ======================

def now_ist() -> datetime:
    return datetime.now(tz=Config.IST)

def is_trading_time() -> bool:
    now = now_ist()
    if now.weekday() >= 5:  # Weekend
        return False
    
    start_hour, start_min = map(int, Config.ENTRY_START.split(':'))
    end_hour, end_min = map(int, Config.EXIT_END.split(':'))
    
    current_minutes = now.hour * 60 + now.minute
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min
    
    return start_minutes <= current_minutes <= end_minutes

def load_api_key() -> str:
    with open(Config.APIKEY_FILE, "r") as f:
        return f.read().strip()

def round_to_strike(price: float, step: int = 50) -> int:
    return int(round(price / step) * step)

def get_nearest_tuesday_expiry() -> str:
    today = now_ist().date()
    days_until_tuesday = (1 - today.weekday()) % 7
    
    if days_until_tuesday == 0 and now_ist().hour >= 15:
        days_until_tuesday = 7
    
    next_tuesday = today + timedelta(days=days_until_tuesday)
    return next_tuesday.strftime('%Y-%m-%d')

def make_option_symbol(strike: int, option_type: str) -> str:
    """Generate option symbol for nearest Tuesday expiry"""
    expiry_date = get_nearest_tuesday_expiry()
    dt = datetime.fromisoformat(expiry_date)
    
    yy = str(dt.year)[-2:]
    mon = dt.strftime("%b").upper()
    dd = dt.strftime("%d")
    
    return f"{Config.UNDERLYING}{dd}{mon}{yy}{strike}{option_type}"

# ======================
# OPENALGO CLIENT
# ======================

class OpenAlgoClient:
    def __init__(self):
        self.base_url = Config.OPENALGO_BASE
        self.api_key = load_api_key()
        self.session = requests.Session()
        self.session.timeout = 30
    
    def _post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request with error handling"""
        url = f"{self.base_url}{endpoint}"
        payload["apikey"] = self.api_key
        
        try:
            logger.debug(f"API Request: {endpoint} - {payload}")
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"API Response: {result}")
            return result
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test OpenAlgo connection"""
        try:
            result = self._post_request('/api/v1/funds', {})
            return result.get('status') == 'success'
        except:
            return False
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return self._post_request('/api/v1/positionbook', {})
    
    def has_open_position(self, symbol: str = None) -> bool:
        """Check if there's an open position"""
        try:
            positions = self.get_positions()
            
            if positions.get('status') != 'success':
                logger.error(f"Position API failed: {positions.get('emsg', 'Unknown error')}")
                return False
            
            for pos in positions.get('data', []):
                # Check multiple possible quantity field names
                qty = int(pos.get('netqty', 0) or pos.get('net_qty', 0) or pos.get('quantity', 0))
                if qty != 0:
                    if symbol is None or pos.get('symbol') == symbol:
                        logger.info(f"Found open position: {pos.get('symbol')} qty={qty}")
                        return True
            
            logger.info("No open positions found in broker account")
            return False
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return False
    
    def validate_position_exists(self, symbol: str) -> bool:
        """Validate that a specific position actually exists in broker account"""
        try:
            positions = self.get_positions()
            
            if positions.get('status') != 'success':
                return False
            
            for pos in positions.get('data', []):
                if pos.get('symbol') == symbol:
                    qty = int(pos.get('netqty', 0) or pos.get('net_qty', 0) or pos.get('quantity', 0))
                    if qty != 0:
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False
    
    def place_order(self, symbol: str, action: str, qty: int) -> Dict[str, Any]:
        """Place market order"""
        payload = {
            "strategy": "Python",
            "symbol": symbol,
            "exchange": "NFO",
            "product": "NRML",
            "pricetype": "MARKET",
            "action": action,
            "quantity": str(qty),
        }
        
        logger.info(f"Placing {action} order: {symbol} x{qty}")
        return self._post_request('/api/v1/placeorder', payload)
    
    def close_position(self, symbol: str = None) -> Dict[str, Any]:
        """Close position"""
        payload = {}
        if symbol:
            payload.update({
                "symbol": symbol,
                "exchange": "NFO",
                "product": "NRML"
            })
        
        logger.info(f"Closing position: {symbol or 'ALL'}")
        return self._post_request('/api/v1/closeposition', payload)
    
    def get_quote(self, symbol: str) -> float:
        """Get LTP for symbol"""
        payload = {"symbol": symbol, "exchange": "NFO"}
        response = self._post_request('/api/v1/quotes', payload)
        
        if response.get('status') == 'success':
            data = response.get('data', {})
            ltp = data.get('ltp')
            if ltp is None:
                raise ValueError(f"No LTP data for {symbol}")
            return float(ltp)
        else:
            raise ValueError(f"Failed to get quote: {response.get('emsg')}")

# ======================
# DATABASE MANAGER
# ======================

class DatabaseManager:
    def __init__(self):
        self.config = Config.DB_CONFIG
        self.connection = None
    
    def connect(self):
        if not self.connection or not self.connection.is_connected():
            self.connection = mysql.connector.connect(**self.config)
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False):
        self.connect()
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, params or ())
        
        if query.strip().upper().startswith('SELECT'):
            result = cursor.fetchone() if fetch_one else cursor.fetchall()
            cursor.close()
            return result
        else:
            self.connection.commit()
            cursor.close()
            return None
    
    def get_pcr_data(self) -> Tuple[float, float, Optional[float], Optional[float], datetime]:
        query = f"""
        SELECT pcr, oi_diff, time_stamp 
        FROM wp_{Config.UNDERLYING.lower()}_pcr 
        ORDER BY id DESC LIMIT 2
        """
        rows = self.execute_query(query)
        
        if not rows:
            raise RuntimeError("No PCR data found")
        
        current = rows[0]
        previous = rows[1] if len(rows) > 1 else None
        
        return (
            float(current["pcr"]),
            float(current["oi_diff"]),
            float(previous["pcr"]) if previous else None,
            float(previous["oi_diff"]) if previous else None,
            current["time_stamp"]
        )
    
    def get_daily_pnl(self) -> float:
        today = now_ist().date()
        query = """
        SELECT SUM(pnl) as total_pnl 
        FROM strategy_entries 
        WHERE DATE(entry_time) = %s AND closed = 1 AND pnl IS NOT NULL
        """
        result = self.execute_query(query, (today,), fetch_one=True)
        return float(result.get('total_pnl') or 0)
    
    def save_trade_entry(self, position: TradePosition):
        query = """
        INSERT INTO strategy_entries 
        (underlying, position, entry_symbol, entry_price, entry_pcr, 
         entry_oidiff, entry_time, entry_order_id) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        self.execute_query(query, (
            Config.UNDERLYING, position.direction, position.symbol,
            position.entry_price, position.entry_pcr, position.entry_oi_diff,
            position.entry_time, position.order_id
        ))
    
    def close_trade(self, symbol: str, close_price: float, pnl: float, reason: str):
        query = """
        UPDATE strategy_entries 
        SET closed = 1, close_time = %s, close_reason = %s, pnl = %s
        WHERE entry_symbol = %s AND closed = 0
        """
        self.execute_query(query, (now_ist(), reason, pnl, symbol))

# ======================
# DATA FETCHER
# ======================

class DataFetcher:
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def get_futures_data(self) -> Tuple[float, float]:
        """Get live futures data from API with fallback to database"""
        try:
            response = requests.get(Config.FUTURES_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.debug(f"Futures API Response: {data}")
            
            # Parse the nested JSON structure
            nifty_data = data.get(Config.UNDERLYING, {})
            if not nifty_data:
                logger.warning(f"No data for {Config.UNDERLYING} in API response. Using database fallback.")
                return self._get_futures_from_db()
            
            price = float(nifty_data.get('lastPrice', 0))
            vwap = float(nifty_data.get('vwap', 0))
            
            if price <= 0 or vwap <= 0:
                logger.warning(f"Invalid API data - price: {price}, vwap: {vwap}. Using database fallback.")
                return self._get_futures_from_db()
            
            logger.info(f"Using live futures data: price={price}, vwap={vwap}")
            return price, vwap
            
        except Exception as e:
            logger.error(f"Futures API failed: {e}. Using database fallback.")
            return self._get_futures_from_db()
    
    def _get_futures_from_db(self) -> Tuple[float, float]:
        """Get futures data from database as fallback"""
        query = f"""
        SELECT price, vwap 
        FROM wp_{Config.UNDERLYING.lower()}_futures 
        ORDER BY id DESC LIMIT 1
        """
        result = self.db.execute_query(query, fetch_one=True)
        
        if not result:
            raise RuntimeError("No futures data in database")
        
        price = float(result.get("price", 0))
        vwap = float(result.get("vwap", 0))
        
        if price <= 0 or vwap <= 0:
            raise ValueError(f"Invalid database futures data: price={price}, vwap={vwap}")
        
        logger.info(f"Using database futures data: price={price}, vwap={vwap}")
        return price, vwap
    
    def get_market_data(self) -> MarketData:
        """Get complete market data"""
        # Get PCR data
        curr_pcr, curr_oi, prev_pcr, prev_oi, pcr_time = self.db.get_pcr_data()
        
        # Get futures data
        price, vwap = self.get_futures_data()
        
        return MarketData(
            timestamp=now_ist(),
            price=price,
            vwap=vwap,
            pcr=curr_pcr,
            oi_diff=curr_oi,
            prev_pcr=prev_pcr,
            prev_oi_diff=prev_oi
        )

# ======================
# TRADING STRATEGY
# ======================

class TradingStrategy:
    def __init__(self, client: OpenAlgoClient, db: DatabaseManager):
        self.client = client
        self.db = db
        self.fetcher = DataFetcher(db)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load strategy state"""
        default = {"position": None}
        
        if os.path.exists(Config.STATE_FILE):
            try:
                with open(Config.STATE_FILE, "r") as f:
                    loaded = json.load(f)
                    # Convert position dict back to TradePosition if exists
                    if loaded.get("position"):
                        pos_data = loaded["position"]
                        loaded["position"] = TradePosition(
                            symbol=pos_data["symbol"],
                            direction=pos_data["direction"],
                            entry_price=pos_data["entry_price"],
                            entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                            entry_pcr=pos_data["entry_pcr"],
                            entry_oi_diff=pos_data["entry_oi_diff"],
                            order_id=pos_data.get("order_id")
                        )
                    return loaded
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        
        return default
    
    def _save_state(self):
        """Save strategy state"""
        try:
            # Convert TradePosition to dict for JSON serialization
            state_to_save = self.state.copy()
            if self.state.get("position"):
                state_to_save["position"] = asdict(self.state["position"])
            
            with open(Config.STATE_FILE, "w") as f:
                json.dump(state_to_save, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _validate_internal_position(self):
        """Validate internal position state against broker positions"""
        if not self.state.get("position"):
            return  # No internal position to validate
        
        position = self.state["position"]
        
        # Check if position actually exists in broker account
        if not self.client.validate_position_exists(position.symbol):
            logger.warning(f"Internal position {position.symbol} not found in broker account - clearing state")
            print(f"⚠️  Position mismatch detected!")
            print(f"   Internal state shows: {position.symbol}")
            print(f"   Broker account shows: No matching position")
            print(f"   Likely cause: Order rejection or execution failure")
            print(f"   Action: Clearing internal position state")
            
            # Clear the invalid position from state
            self.state["position"] = None
            self._save_state()
            
            # Update database to mark as closed with reason
            self.db.close_trade(position.symbol, 0.0, 0.0, "Order rejected/Position not found")
            
            print(f"✅ Internal state synchronized with broker account")
    
    def check_entry_conditions(self, market_data: MarketData) -> Optional[str]:
        """Check entry conditions and return direction if met"""
        
        # First validate any existing internal position
        self._validate_internal_position()
        
        print(f"\n📊 ENTRY CONDITIONS ANALYSIS:")
        print(f"{'='*50}")
        
        # 1. Trading time check
        trading_time_ok = is_trading_time()
        print(f"✓ Trading Time ({Config.ENTRY_START}-{Config.EXIT_END}): {'YES' if trading_time_ok else 'NO'}")
        if not trading_time_ok:
            return None
        
        # 2. Position check (both internal state and broker)
        has_internal_position = self.state.get("position") is not None
        has_broker_position = self.client.has_open_position()
        position_ok = not has_internal_position and not has_broker_position
        print(f"✓ No Internal Position: {'YES' if not has_internal_position else 'NO'}")
        print(f"✓ No Broker Position: {'YES' if not has_broker_position else 'NO'}")
        print(f"✓ Overall Position Check: {'YES' if position_ok else 'NO'}")
        if not position_ok:
            return None
        
        # 3. Daily loss limit check
        daily_pnl = self.db.get_daily_pnl()
        daily_loss_ok = daily_pnl > Config.MAX_DAILY_LOSS
        print(f"✓ Daily Loss Limit (>{Config.MAX_DAILY_LOSS}): {'YES' if daily_loss_ok else 'NO'} (Current: Rs{daily_pnl:,.2f})")
        if not daily_loss_ok:
            logger.warning("Daily loss limit reached")
            return None
        
        # 4. VWAP band check
        price_vwap_diff = abs(market_data.price - market_data.vwap)
        vwap_ok = price_vwap_diff <= Config.VWAP_BAND
        print(f"✓ Within VWAP Band (±{Config.VWAP_BAND}): {'YES' if vwap_ok else 'NO'} (Diff: {price_vwap_diff:.2f})")
        if not vwap_ok:
            return None
        
        # 5. Data availability check
        data_ok = market_data.pcr_change is not None and market_data.oi_change is not None
        print(f"✓ Historical Data Available: {'YES' if data_ok else 'NO'}")
        if not data_ok:
            return None
        
        print(f"\n🔍 DIRECTIONAL CONDITIONS:")
        print(f"{'='*50}")
        
        # CALL condition breakdown
        print(f"📈 CALL CONDITIONS:")
        call_price_above_vwap = market_data.price > market_data.vwap
        call_pcr_min = market_data.pcr >= Config.PCR_CALL_MIN
        call_pcr_change = market_data.pcr_change >= Config.PCR_CHANGE_THRESHOLD
        call_oi_change = market_data.oi_change >= Config.OI_CHANGE_THRESHOLD
        
        print(f"   Price > VWAP ({market_data.price:.2f} > {market_data.vwap:.2f}): {'YES' if call_price_above_vwap else 'NO'}")
        print(f"   PCR >= {Config.PCR_CALL_MIN} ({market_data.pcr:.4f}): {'YES' if call_pcr_min else 'NO'}")
        print(f"   PCR Change >= {Config.PCR_CHANGE_THRESHOLD:.1%} ({market_data.pcr_change:.2%}): {'YES' if call_pcr_change else 'NO'}")
        print(f"   OI Change >= {Config.OI_CHANGE_THRESHOLD:.1%} ({market_data.oi_change:.2%}): {'YES' if call_oi_change else 'NO'}")
        
        call_conditions = call_price_above_vwap and call_pcr_min and call_pcr_change and call_oi_change
        print(f"   🎯 CALL SIGNAL: {'YES' if call_conditions else 'NO'}")
        
        # PUT condition breakdown
        print(f"\n📉 PUT CONDITIONS:")
        put_price_below_vwap = market_data.price < market_data.vwap
        put_pcr_max = market_data.pcr <= Config.PCR_PUT_MAX
        put_pcr_change = market_data.pcr_change <= -Config.PCR_CHANGE_THRESHOLD
        put_oi_change = market_data.oi_change <= -Config.OI_CHANGE_THRESHOLD
        
        print(f"   Price < VWAP ({market_data.price:.2f} < {market_data.vwap:.2f}): {'YES' if put_price_below_vwap else 'NO'}")
        print(f"   PCR <= {Config.PCR_PUT_MAX} ({market_data.pcr:.4f}): {'YES' if put_pcr_max else 'NO'}")
        print(f"   PCR Change <= -{Config.PCR_CHANGE_THRESHOLD:.1%} ({market_data.pcr_change:.2%}): {'YES' if put_pcr_change else 'NO'}")
        print(f"   OI Change <= -{Config.OI_CHANGE_THRESHOLD:.1%} ({market_data.oi_change:.2%}): {'YES' if put_oi_change else 'NO'}")
        
        put_conditions = put_price_below_vwap and put_pcr_max and put_pcr_change and put_oi_change
        print(f"   🎯 PUT SIGNAL: {'YES' if put_conditions else 'NO'}")
        
        if call_conditions:
            logger.info("CALL entry conditions met")
            return "CALL"
        elif put_conditions:
            logger.info("PUT entry conditions met")
            return "PUT"
        
        return None
    
    def check_exit_conditions(self, market_data: MarketData) -> Optional[str]:
        """Check exit conditions and return reason if met"""
        if not self.state.get("position"):
            return None
        
        position = self.state["position"]
        
        print(f"\n📊 EXIT CONDITIONS ANALYSIS:")
        print(f"{'='*50}")
        
        # Get current P&L
        try:
            current_price = self.client.get_quote(position.symbol)
            current_pnl = (current_price - position.entry_price) * Config.QTY
            print(f"Current P&L: Rs{current_pnl:,.2f}")
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            print(f"❌ Unable to get current price: {e}")
            return None
        
        # 1. Target profit check
        target_hit = current_pnl >= Config.TARGET_PNL
        print(f"✓ Target Profit (>={Config.TARGET_PNL}): {'YES' if target_hit else 'NO'} (Current: Rs{current_pnl:,.2f})")
        if target_hit:
            return "Target profit reached"
        
        # 2. Stop loss check
        stop_hit = current_pnl <= Config.STOP_PNL
        print(f"✓ Stop Loss (<={Config.STOP_PNL}): {'YES' if stop_hit else 'NO'} (Current: Rs{current_pnl:,.2f})")
        if stop_hit:
            return "Stop loss hit"
        
        # 3. Trading time check
        trading_time_ok = is_trading_time()
        print(f"✓ Trading Time Active: {'YES' if trading_time_ok else 'NO'}")
        if not trading_time_ok:
            return "Trading time ended"
        
        # 4. Daily loss limit check
        daily_pnl = self.db.get_daily_pnl() + current_pnl
        daily_loss_ok = daily_pnl > Config.MAX_DAILY_LOSS
        print(f"✓ Daily Loss Limit (>{Config.MAX_DAILY_LOSS}): {'YES' if daily_loss_ok else 'NO'} (Total: Rs{daily_pnl:,.2f})")
        if not daily_loss_ok:
            return "Daily loss limit reached"
        
        # 5. Multi-condition exits
        print(f"\n🔍 MULTI-CONDITION EXIT ANALYSIS:")
        print(f"{'='*50}")
        
        pcr_change_from_entry = 0.0
        oi_change_from_entry = 0.0
        
        if position.entry_pcr != 0:
            pcr_change_from_entry = (market_data.pcr - position.entry_pcr) / abs(position.entry_pcr)
        
        if position.entry_oi_diff != 0:
            oi_change_from_entry = (market_data.oi_diff - position.entry_oi_diff) / abs(position.entry_oi_diff)
        
        print(f"Entry PCR: {position.entry_pcr:.4f} | Current: {market_data.pcr:.4f} | Change: {pcr_change_from_entry:.2%}")
        print(f"Entry OI: {position.entry_oi_diff:,.0f} | Current: {market_data.oi_diff:,.0f} | Change: {oi_change_from_entry:.2%}")
        
        # Position-specific multi-condition exits
        if position.direction == "CALL":
            print(f"\n📈 CALL EXIT CONDITIONS:")
            pcr_breach = pcr_change_from_entry <= -0.20  # 20% decline
            oi_decline = oi_change_from_entry <= -0.20   # 20% decline
            vwap_breach = market_data.price < market_data.vwap
            
            print(f"   PCR Decline >= 20% ({pcr_change_from_entry:.2%}): {'YES' if pcr_breach else 'NO'}")
            print(f"   OI Decline >= 20% ({oi_change_from_entry:.2%}): {'YES' if oi_decline else 'NO'}")
            print(f"   Price < VWAP ({market_data.price:.2f} < {market_data.vwap:.2f}): {'YES' if vwap_breach else 'NO'}")
            
            if pcr_breach and oi_decline and vwap_breach:
                print(f"   🎯 MULTI-CONDITION EXIT: YES")
                return "Multi-condition exit (CALL)"
            else:
                print(f"   🎯 MULTI-CONDITION EXIT: NO")
        
        elif position.direction == "PUT":
            print(f"\n📉 PUT EXIT CONDITIONS:")
            pcr_breach = pcr_change_from_entry >= 0.20   # 20% increase
            oi_increase = oi_change_from_entry >= 0.20   # 20% increase  
            vwap_breach = market_data.price > market_data.vwap
            
            print(f"   PCR Increase >= 20% ({pcr_change_from_entry:.2%}): {'YES' if pcr_breach else 'NO'}")
            print(f"   OI Increase >= 20% ({oi_change_from_entry:.2%}): {'YES' if oi_increase else 'NO'}")
            print(f"   Price > VWAP ({market_data.price:.2f} > {market_data.vwap:.2f}): {'YES' if vwap_breach else 'NO'}")
            
            if pcr_breach and oi_increase and vwap_breach:
                print(f"   🎯 MULTI-CONDITION EXIT: YES")
                return "Multi-condition exit (PUT)"
            else:
                print(f"   🎯 MULTI-CONDITION EXIT: NO")
        
        return None
    
    def execute_entry(self, direction: str, market_data: MarketData) -> bool:
        """Execute trade entry"""
        try:
            # Generate option symbol
            strike = round_to_strike(market_data.price)
            option_type = "CE" if direction == "CALL" else "PE"
            symbol = make_option_symbol(strike, option_type)
            
            logger.info(f"Executing {direction} entry: {symbol}")
            print(f"🚀 Executing {direction} trade: {symbol} @ strike {strike}")
            
            # Place order
            order_response = self.client.place_order(symbol, "BUY", Config.QTY)
            
            if order_response.get('status') != 'success':
                error_msg = order_response.get('emsg', 'Unknown error')
                logger.error(f"Order failed: {error_msg}")
                print(f"❌ Order REJECTED: {error_msg}")
                return False
            
            order_id = order_response.get('orderid')
            print(f"✅ Order placed successfully. Order ID: {order_id}")
            
            # Wait for execution and verify position exists
            print("⏳ Waiting for order execution...")
            time.sleep(3)  # Wait for execution
            
            # Verify position actually exists in broker account
            if not self.client.validate_position_exists(symbol):
                logger.error(f"Order placed but position not found in broker account: {symbol}")
                print(f"❌ Order execution failed - position not found in broker account")
                return False
            
            # Get entry price
            try:
                entry_price = self.client.get_quote(symbol)
                print(f"📊 Entry price obtained: Rs{entry_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to get entry price: {e}")
                print(f"❌ Failed to get entry price: {e}")
                return False
            
            # Create position object
            position = TradePosition(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=now_ist(),
                entry_pcr=market_data.pcr,
                entry_oi_diff=market_data.oi_diff,
                order_id=order_id
            )
            
            # Save to state and database
            self.state["position"] = position
            self._save_state()
            self.db.save_trade_entry(position)
            
            print(f"✅ Trade executed successfully!")
            print(f"   Symbol: {symbol}")
            print(f"   Entry Price: Rs{entry_price:.2f}")
            print(f"   Quantity: {Config.QTY}")
            print(f"   Order ID: {order_id}")
            
            logger.info(f"Trade executed: {symbol} @ {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            print(f"❌ Trade execution failed: {e}")
            return False
    
    def execute_exit(self, reason: str, market_data: MarketData) -> bool:
        """Execute trade exit"""
        if not self.state.get("position"):
            return False
        
        try:
            position = self.state["position"]
            logger.info(f"Executing exit: {position.symbol} - {reason}")
            print(f"Exiting position: {position.symbol} - {reason}")
            
            # Close position
            close_response = self.client.close_position(position.symbol)
            
            if close_response.get('status') == 'success':
                # Get exit price and calculate P&L
                exit_price = self.client.get_quote(position.symbol)
                pnl = (exit_price - position.entry_price) * Config.QTY
                
                # Update database
                self.db.close_trade(position.symbol, exit_price, pnl, reason)
                
                # Clear state
                self.state["position"] = None
                self._save_state()
                
                print(f"Position closed: P&L = Rs{pnl:,.2f}")
                logger.info(f"Position closed: P&L = Rs{pnl:,.2f}")
                return True
            else:
                logger.error(f"Position close failed: {close_response.get('emsg')}")
                return False
                
        except Exception as e:
            logger.error(f"Exit execution failed: {e}")
            return False
    
    def print_status(self, market_data: MarketData):
        """Print comprehensive market status"""
        print(f"\n{'='*70}")
        print(f"📊 MARKET STATUS - {market_data.timestamp.strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        print(f"💹 Price: {market_data.price:,.2f} | VWAP: {market_data.vwap:,.2f} | Diff: {market_data.price - market_data.vwap:+.2f}")
        print(f"📈 PCR: {market_data.pcr:.4f} | OI Diff: {market_data.oi_diff:,.0f}")
        
        if market_data.pcr_change is not None:
            print(f"📊 PCR Change: {market_data.pcr_change:.2%}")
        if market_data.oi_change is not None:
            print(f"📊 OI Change: {market_data.oi_change:.2%}")
        
        # Validate internal position first
        self._validate_internal_position()
        
        # Position status
        if self.state.get("position"):
            pos = self.state["position"]
            print(f"\n🎯 ACTIVE POSITION:")
            print(f"   Direction: {pos.direction}")
            print(f"   Symbol: {pos.symbol}")
            entry_time = pos.entry_time.strftime('%H:%M:%S')
            duration = now_ist() - pos.entry_time.replace(tzinfo=Config.IST)
            print(f"   Entry Time: {entry_time}")
            print(f"   Duration: {str(duration).split('.')[0]}")
            
            # Verify position exists in broker account
            if self.client.validate_position_exists(pos.symbol):
                print(f"   Broker Status: ✅ CONFIRMED")
                try:
                    current_price = self.client.get_quote(pos.symbol)
                    pnl = (current_price - pos.entry_price) * Config.QTY
                    print(f"   Entry Price: Rs{pos.entry_price:.2f}")
                    print(f"   Current Price: Rs{current_price:.2f}")
                    print(f"   Unrealized P&L: Rs{pnl:,.2f}")
                except Exception as e:
                    print(f"   P&L: Unable to calculate ({e})")
            else:
                print(f"   Broker Status: ❌ NOT FOUND (will be cleared)")
        else:
            print(f"\n🎯 POSITION: None")
        
        daily_pnl = self.db.get_daily_pnl()
        print(f"\n💰 Daily Realized P&L: Rs{daily_pnl:,.2f}")
        print(f"{'='*70}")
        if self.state.get("position"):
            exit_reason = self.check_exit_conditions(market_data)
            if exit_reason:
                print(f"EXIT SIGNAL: {exit_reason}")
            else:
                print("Exit conditions: Holding position")
        
        print(f"{'='*70}")

# ======================
# MAIN EXECUTION
# ======================

def main():
    """Main trading loop"""
    print("🔁 OpenAlgo Python Trading Strategy Starting...")
    print(f"Underlying: {Config.UNDERLYING} | Quantity: {Config.QTY}")
    print(f"Trading Hours: {Config.ENTRY_START} - {Config.EXIT_END}")
    print(f"Target: Rs{Config.TARGET_PNL:,.0f} | Stop: Rs{Config.STOP_PNL:,.0f}")
    print(f"Daily Limit: Rs{Config.MAX_DAILY_LOSS:,.0f}")
    
    logger.info("Starting trading strategy")
    
    # Initialize components
    try:
        client = OpenAlgoClient()
        db = DatabaseManager()
        strategy = TradingStrategy(client, db)
        
        # Test connection
        if not client.test_connection():
            print("OpenAlgo connection failed!")
            return
        
        print("All systems initialized successfully!")
        logger.info("All components initialized")
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        logger.error(f"Initialization failed: {e}")
        return
    
    # Main trading loop
    consecutive_errors = 0
    max_errors = 5
    
    while True:
        try:
            # Check if it's weekend
            if now_ist().weekday() >= 5:
                print("Weekend detected. Strategy paused.")
                time.sleep(3600)  # Sleep 1 hour
                continue
            
            # Check trading time
            if not is_trading_time():
                now = now_ist()
                if now.hour < 11:
                    print(f"Pre-market. Strategy starts at {Config.ENTRY_START}")
                elif now.hour >= 16:
                    print("Post-market. Strategy will resume tomorrow.")
                else:
                    print(f"Outside trading window ({now.strftime('%H:%M')})")
                time.sleep(Config.POLL_INTERVAL)
                continue
            
            # Get market data
            market_data = strategy.fetcher.get_market_data()
            strategy.print_status(market_data)
            
            # Check for exits first (only if we have a position)
            if strategy.state.get("position"):
                exit_reason = strategy.check_exit_conditions(market_data)
                if exit_reason:
                    strategy.execute_exit(exit_reason, market_data)
                    continue
            
            # Check for entries (only if we don't have a position)
            if not strategy.state.get("position"):
                entry_direction = strategy.check_entry_conditions(market_data)
                if entry_direction:
                    strategy.execute_entry(entry_direction, market_data)
                    continue
            
            if strategy.state.get("position"):
                print("💼 Holding position - monitoring exit conditions...")
            else:
                print("🔍 No position - monitoring entry conditions...")
            
            # Reset error counter
            consecutive_errors = 0
            
            # Sleep until next check
            time.sleep(Config.POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nStrategy stopped by user")
            logger.info("Strategy stopped by user")
            
            # Emergency exit
            if strategy.state.get("position"):
                try:
                    market_data = strategy.fetcher.get_market_data()
                    strategy.execute_exit("Emergency stop", market_data)
                except:
                    client.close_position()
            break
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error in main loop (attempt {consecutive_errors}): {e}")
            print(f"Error: {e}")
            
            if consecutive_errors >= max_errors:
                print("Too many consecutive errors. Stopping strategy.")
                logger.critical("Too many consecutive errors. Stopping.")
                break
            
            time.sleep(Config.POLL_INTERVAL)
    
    print("Strategy terminated")
    logger.info("Strategy terminated")

# ======================
# DIAGNOSTICS
# ======================

def run_diagnostics():
    """Run system diagnostics"""
    print("Running system diagnostics...")
    
    try:
        # Test 1: API Key
        print("1. Testing API key...")
        api_key = load_api_key()
        print(f"   API Key loaded: {api_key[:10]}...")
        
        # Test 2: Database
        print("2. Testing database connection...")
        db = DatabaseManager()
        db.connect()
        print("   Database connection: SUCCESS")
        
        # Test 3: OpenAlgo API
        print("3. Testing OpenAlgo API...")
        client = OpenAlgoClient()
        if client.test_connection():
            print("   OpenAlgo API: SUCCESS")
        else:
            print("   OpenAlgo API: FAILED")
            return False
        
        # Test 4: Futures API
        print("4. Testing futures data API...")
        try:
            response = requests.get(Config.FUTURES_API_URL, timeout=10)
            response.raise_for_status()
            api_data = response.json()
            print(f"   Raw API Response: {api_data}")
            
            # Test parsing
            nifty_data = api_data.get(Config.UNDERLYING, {})
            if nifty_data:
                price = float(nifty_data.get('lastPrice', 0))
                vwap = float(nifty_data.get('vwap', 0))
                print(f"   Futures API: SUCCESS")
                print(f"   Parsed - Price: {price}, VWAP: {vwap}")
            else:
                print(f"   Futures API: FAILED - No {Config.UNDERLYING} data found")
                # Test with database fallback
                fetcher = DataFetcher(db)
                price, vwap = fetcher._get_futures_from_db()
                print(f"   Database Fallback: SUCCESS (Price: {price}, VWAP: {vwap})")
        except Exception as e:
            print(f"   Futures API: FAILED - {e}")
            try:
                fetcher = DataFetcher(db)
                price, vwap = fetcher._get_futures_from_db()
                print(f"   Database Fallback: SUCCESS (Price: {price}, VWAP: {vwap})")
            except Exception as db_e:
                print(f"   Database Fallback: FAILED - {db_e}")
                return False
        
        # Test 5: Option symbol generation
        print("5. Testing option symbol generation...")
        call_symbol = make_option_symbol(24500, "CE")
        put_symbol = make_option_symbol(24500, "PE")
        print(f"   CALL symbol: {call_symbol}")
        print(f"   PUT symbol: {put_symbol}")
        
        # Test 6: Position checking (improved with detailed output)
        print("6. Testing position checking...")
        try:
            positions_response = client.get_positions()
            print(f"   Position API Status: {positions_response.get('status', 'unknown')}")
            
            if positions_response.get('status') == 'success':
                positions_data = positions_response.get('data', [])
                print(f"   Total positions returned: {len(positions_data)}")
                
                open_positions = 0
                for pos in positions_data:
                    symbol = pos.get('symbol', 'Unknown')
                    netqty = pos.get('netqty', 0)
                    pnl = pos.get('unrealizedpnl', 0)
                    
                    if int(netqty) != 0:
                        open_positions += 1
                        print(f"   Open Position: {symbol} | Qty: {netqty} | P&L: {pnl}")
                
                if open_positions == 0:
                    print("   No open positions found")
                
                has_pos = client.has_open_position()
                print(f"   has_open_position() result: {has_pos}")
            else:
                print(f"   Position API Error: {positions_response.get('emsg', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"   Position checking FAILED: {e}")
            return False
        
        print("All diagnostics passed!")
        return True
        
    except Exception as e:
        print(f"Diagnostics failed: {e}")
        return False

def demo_flow():
    """Run demo trading flow"""
    print("Running demo flow...")
    
    try:
        client = OpenAlgoClient()
        
        # Test order placement
        symbol = make_option_symbol(24500, "PE")
        print(f"Demo symbol: {symbol}")
        
        # Place demo order
        buy_response = client.place_order(symbol, "BUY", Config.QTY)
        if buy_response.get('status') == 'success':
            print("Demo BUY order placed successfully")
            order_id = buy_response.get('orderid')
            print(f"Order ID: {order_id}")
            
            # Wait and then exit
            print("Waiting 10 seconds before exit...")
            time.sleep(10)
            
            exit_response = client.close_position(symbol)
            if exit_response.get('status') == 'success':
                print("Demo position closed successfully")
            else:
                print(f"Demo exit failed: {exit_response.get('emsg')}")
        else:
            print(f"Demo order failed: {buy_response.get('emsg')}")
            
    except Exception as e:
        print(f"Demo flow failed: {e}")

# ======================
# ENTRY POINT
# ======================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--diagnostics":
            success = run_diagnostics()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--demo":
            demo_flow()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python strategy.py              # Run trading strategy")
            print("  python strategy.py --diagnostics # Run system diagnostics")  
            print("  python strategy.py --demo       # Run demo flow")
            print("  python strategy.py --help       # Show this help")
            sys.exit(0)
    
    # Run main strategy
    main()