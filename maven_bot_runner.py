#!/usr/bin/env python3
"""
MAVEN TRADING BOT v6.0 - FINAL PRODUCTION
GitHub Actions: runs every 15 minutes, 24/7, free
Writes all data to Google Sheet, sends Telegram alerts
"""

import os, json, traceback, tempfile, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from google.oauth2.service_account import Credentials
import gspread
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_SHEET_ID    = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_CREDS_JSON  = os.getenv("GOOGLE_CREDENTIALS_JSON")

SYMBOL   = "BTCUSDT"
INTERVAL = "15"        # Bybit uses minutes as number (15 not "15m")
BYBIT    = "https://api.bybit.com"
ALERT_FILE = "last_alert.json"
START_BAL  = 2006.0   # update when your balance changes

# ============================================================
# SESSION
# ============================================================

def make_session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1,
              status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

SESSION = make_session()

def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ============================================================
# TELEGRAM
# ============================================================

def tg(text):
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("[TG] Not configured")
            return
        r = SESSION.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10
        )
        if r.status_code == 200:
            print(f"[TG SENT] {text[:50]}...")
        else:
            print(f"[TG ERROR] {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"[TG EXCEPTION] {e}")

# ============================================================
# ALERT STATE
# ============================================================

def load_last_candle():
    try:
        if os.path.exists(ALERT_FILE):
            with open(ALERT_FILE) as f:
                return json.load(f).get("last_candle")
    except:
        pass
    return None

def save_last_candle(candle):
    try:
        with open(ALERT_FILE, "w") as f:
            json.dump({"last_candle": candle}, f)
    except:
        pass

# ============================================================
# FETCH BYBIT (replaces Binance - not geo-blocked by US servers)
# Bybit BTCUSDT Perpetual = same instrument as Binance BTCUSDT Futures
# No API key needed for public market data
# ============================================================

def fetch_klines(symbol, interval, limit=300):
    """
    Bybit V5 linear kline endpoint.
    Returns: [startTime, open, high, low, close, volume, turnover]
    Bybit returns newest first — we reverse to oldest-first for our calcs.
    """
    r = SESSION.get(
        f"{BYBIT}/v5/market/kline",
        params={
            "category": "linear",
            "symbol":   symbol,
            "interval": interval,
            "limit":    limit,
        },
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        raise Exception(f"Bybit API error: {data.get('retMsg')}")

    rows = data["result"]["list"]   # newest → oldest
    rows = list(reversed(rows))     # flip to oldest → newest

    df = pd.DataFrame(rows, columns=[
        'open_time','open','high','low','close','volume','turnover'
    ])
    for c in ['open','high','low','close','volume','turnover']:
        df[c] = pd.to_numeric(df[c])
    df['open_time'] = pd.to_numeric(df['open_time'])

    # CVD uses taker_buy_base — Bybit kline doesn't provide it.
    # We estimate using price direction (same method as our Pine Script):
    #   bull candle → buyers dominated → taker_buy ≈ full volume
    #   bear candle → sellers dominated → taker_buy ≈ 0
    bull_candle = df['close'] >= df['open']
    df['taker_buy_base'] = np.where(bull_candle, df['volume'], 0)

    print(f"[BYBIT] {len(df)} candles | BTC @ {df['close'].iloc[-1]:.2f}")
    return df.reset_index(drop=True)

# ============================================================
# PINE HELPERS
# ============================================================

def wilder_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(100)

def wilder_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)\
             .ewm(alpha=1/period, adjust=False).mean()

def ta_pivot_high(series, lb_left, lb_right):
    arr    = series.values
    result = np.full(len(arr), np.nan)
    for i in range(lb_left, len(arr) - lb_right):
        w = arr[i - lb_left : i + lb_right + 1]
        if arr[i] > np.max(np.delete(w, lb_left)):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)

def ta_pivot_low(series, lb_left, lb_right):
    arr    = series.values
    result = np.full(len(arr), np.nan)
    for i in range(lb_left, len(arr) - lb_right):
        w = arr[i - lb_left : i + lb_right + 1]
        if arr[i] < np.min(np.delete(w, lb_left)):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)

def ta_barssince(condition):
    cond   = condition.values
    result = np.full(len(cond), np.nan)
    last   = np.nan
    for i in range(len(cond)):
        if cond[i]:
            last = 0
        elif not np.isnan(last):
            last += 1
        result[i] = last
    return pd.Series(result, index=condition.index)

def ta_valuewhen(condition, value_series, occurrence=0):
    cond   = condition.values
    vals   = value_series.values
    result = np.full(len(cond), np.nan)
    for i in range(len(cond)):
        count = 0
        for j in range(i, -1, -1):
            if cond[j]:
                if count == occurrence:
                    result[i] = vals[j]
                    break
                count += 1
    return pd.Series(result, index=condition.index)

def get_line_price(b1, p1, b2, p2, cur):
    if b2 == b1: return p2
    return p1 + ((p2 - p1) / (b2 - b1)) * (cur - b1)

# ============================================================
# SIGNAL 1: CVD (exact Pine translation)
# ============================================================

def calc_cvd(df):
    df       = df.copy()
    bull     = df['close'] >= df['open']
    buy_vol  = np.where(
        bull, df['volume'],
        df['volume'] * (df['close']-df['open']) / (df['open']-df['close']+0.001)
    )
    sell_vol = np.where(
        ~bull, df['volume'],
        df['volume'] * (df['open']-df['close']) / (df['close']-df['open']+0.001)
    )
    df['delta']      = buy_vol - sell_vol
    df['cvd']        = df['delta'].cumsum()
    lb               = 20
    df['price_high'] = df['high'].rolling(lb).max()
    df['price_low']  = df['low'].rolling(lb).min()
    df['cvd_high']   = df['cvd'].rolling(lb).max()
    df['cvd_low']    = df['cvd'].rolling(lb).min()
    df['avg_vol']    = df['volume'].rolling(lb).mean()
    df['trend_ma']   = df['close'].rolling(20).mean()
    df['rsi']        = wilder_rsi(df['close'])

    last, prev       = df.iloc[-1], df.iloc[-2]
    vol_ok           = last['volume'] > last['avg_vol'] * 1.1
    up               = last['close'] > last['trend_ma']
    dn               = last['close'] < last['trend_ma']

    bull_abs = last['cvd'] > prev['cvd_high'] and last['high'] < prev['price_high'] and last['delta'] > 0
    bear_abs = last['cvd'] < prev['cvd_low']  and last['low']  > prev['price_low']  and last['delta'] < 0
    bull_exh = last['high'] > prev['price_high'] and last['cvd'] < prev['cvd_high'] and vol_ok
    bear_exh = last['low']  < prev['price_low']  and last['cvd'] > prev['cvd_low']  and vol_ok

    b = (bull_abs or bear_exh) and up and vol_ok and last['rsi'] < 70
    s = (bear_abs or bull_exh) and dn and vol_ok and last['rsi'] > 30

    print(f"[CVD] bull={b} bear={s} | abs_b={bull_abs} abs_s={bear_abs} exh_b={bull_exh} exh_s={bear_exh}")
    return b or s, b, s

# ============================================================
# SIGNAL 2: RSI DIVERGENCE PRO v6 (exact Pine translation)
# ============================================================

def calc_rsi_div(df):
    df  = df.copy()
    lbL = lbR = 5
    df['rsi']    = wilder_rsi(df['close'])
    df['ema50']  = df['close'].ewm(span=50,  adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['avg_vol']= df['volume'].rolling(20).mean()

    df['rsiPl']  = ta_pivot_low(df['rsi'],  lbL, lbR)
    df['rsiPh']  = ta_pivot_high(df['rsi'], lbL, lbR)
    df['plF']    = ~df['rsiPl'].isna()
    df['phF']    = ~df['rsiPh'].isna()

    df['barsPl'] = ta_barssince(df['plF'].shift(1).fillna(False))
    df['barsPh'] = ta_barssince(df['phF'].shift(1).fillna(False))
    df['inRPl']  = (df['barsPl'] >= 5) & (df['barsPl'] <= 60)
    df['inRPh']  = (df['barsPh'] >= 5) & (df['barsPh'] <= 60)

    rp = df['rsi'].shift(lbR)
    lp = df['low'].shift(lbR)
    hp = df['high'].shift(lbR)

    df['pRsiL']  = ta_valuewhen(df['plF'], rp, 1)
    df['pPxL']   = ta_valuewhen(df['plF'], lp, 1)
    df['pRsiH']  = ta_valuewhen(df['phF'], rp, 1)
    df['pPxH']   = ta_valuewhen(df['phF'], hp, 1)

    df['bullDiv'] = df['plF'] & df['inRPl'] & (rp > df['pRsiL']) & (lp < df['pPxL'])
    df['bearDiv'] = df['phF'] & df['inRPh'] & (rp < df['pRsiH']) & (hp > df['pPxH'])

    last     = df.iloc[-1]
    bull_tr  = last['ema50'] > last['ema200']
    bear_tr  = last['ema50'] < last['ema200']
    vol_sp   = (last['volume'] > last['avg_vol'] * 1.2 and
                last['volume'] > df['volume'].iloc[-2])
    rsi_v    = float(last['rsi'])

    long_d   = bool(last['bullDiv']) and bull_tr and vol_sp and rsi_v > 45
    short_d  = bool(last['bearDiv']) and bear_tr and vol_sp and rsi_v < 55
    rec_bull = df['bullDiv'].iloc[-lbR-3:-1].any()
    rec_bear = df['bearDiv'].iloc[-lbR-3:-1].any()

    sig = long_d or short_d or rec_bull or rec_bear
    print(f"[RSI DIV] sig={sig} long={long_d} short={short_d} rb={rec_bull} rs={rec_bear} RSI={rsi_v:.1f}")
    return sig, long_d, short_d, rsi_v, float(last['ema50']), float(last['ema200'])

# ============================================================
# SIGNAL 3: TRENDLINE BREAKS PRO v6 (exact Pine translation)
# ============================================================

def calc_trendline(df):
    df  = df.copy()
    sl  = 14
    df['ema50']  = df['close'].ewm(span=50,  adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ma14']   = df['close'].rolling(14).mean()
    df['atr']    = wilder_atr(df, 14)
    df['avg_vol']= df['volume'].rolling(14).mean()
    df['ph']     = ta_pivot_high(df['high'], sl, sl)
    df['pl']     = ta_pivot_low(df['low'],   sl, sl)

    phi = df.index[~df['ph'].isna()].tolist()
    pli = df.index[~df['pl'].isna()].tolist()
    cur = len(df) - 1
    last= df.iloc[-1]
    atr = float(last['atr'])

    res = np.nan
    sup = np.nan
    if len(phi) >= 2:
        res = get_line_price(int(phi[-2]), float(df.loc[phi[-2],'ph']),
                             int(phi[-1]), float(df.loc[phi[-1],'ph']), cur)
    if len(pli) >= 2:
        sup = get_line_price(int(pli[-2]), float(df.loc[pli[-2],'pl']),
                             int(pli[-1]), float(df.loc[pli[-1],'pl']), cur)

    bb = not np.isnan(res) and float(last['close']) > res + atr * 0.25
    bk = not np.isnan(sup) and float(last['close']) < sup - atr * 0.25
    br = (not np.isnan(res) and float(last['low'])  <= res + atr * 0.05
          and float(last['close']) > res)
    sr = (not np.isnan(sup) and float(last['high']) >= sup - atr * 0.05
          and float(last['close']) < sup)

    eb = float(last['ema50']) > float(last['ema200'])
    es = float(last['ema50']) < float(last['ema200'])
    am = float(last['close']) > float(last['ma14'])
    bm = float(last['close']) < float(last['ma14'])
    vs = (last['volume'] > last['avg_vol'] * 1.5 and
          last['volume'] > df['volume'].iloc[-2])

    ls = eb and am and ((bb and vs) or (br and vs))
    ss = es and bm and ((bk and vs) or (sr and vs))

    print(f"[TL] sig={ls or ss} bb={bb} br={br} bk={bk} sr={sr} ATR={atr:.2f}")
    return (ls or ss, ls, ss, bb, bk, br, sr,
            float(last['ema50']), float(last['ema200']), atr, res, sup)

# ============================================================
# SESSION + MARKET CONDITION
# ============================================================

def get_session():
    h = now_utc().hour
    if   12 <= h < 16: return "LONDON/NEW YORK"
    elif  7 <= h < 12: return "LONDON"
    elif  0 <= h <  7: return "TOKYO"
    elif h >= 21:       return "SYDNEY"
    else:               return "NEW YORK"

def get_market_condition(df):
    try:
        e50  = df['close'].ewm(span=50,  adjust=False).mean()
        e200 = df['close'].ewm(span=200, adjust=False).mean()
        atr  = wilder_atr(df, 14)
        px   = float(df['close'].iloc[-1])
        gap  = abs(float(e50.iloc[-1]) - float(e200.iloc[-1])) / px * 100
        la   = float(atr.iloc[-1])
        aa   = float(atr.rolling(50).mean().iloc[-1])
        r    = df.tail(10)
        hh   = r['high'].iloc[-1] > r['high'].iloc[-5]
        ll   = r['low'].iloc[-1]  < r['low'].iloc[-5]
        hl   = r['low'].iloc[-1]  > r['low'].iloc[-5]
        lh   = r['high'].iloc[-1] < r['high'].iloc[-5]
        if la > aa * 1.5:                     return "VOLATILE"
        if gap > 0.3 and ((hh and hl) or (ll and lh)): return "TRENDING"
        return "RANGING"
    except:
        return "RANGING"

# ============================================================
# LOT SIZE
# ============================================================

def calc_lot_size(entry, sl, balance, risk_pct=1.0):
    risk_amt = balance * (risk_pct / 100)
    sl_dist  = abs(entry - sl)
    return round(risk_amt / sl_dist, 4) if sl_dist > 0 else 0.0

# ============================================================
# CONFLUENCE SCORING
# ============================================================

def score_signals(cvd_b, cvd_s, rsi_l, rsi_s, tl_l, tl_s, ema50, ema200):
    bull = ema50 > ema200
    bear = ema50 < ema200
    lc   = sum([bool(cvd_b), bool(rsi_l), bool(tl_l)])
    sc   = sum([bool(cvd_s), bool(rsi_s), bool(tl_s)])

    if bull and lc >= 2:
        return ("TIER 1" if lc == 3 else "TIER 2"), "LONG",  lc
    if bear and sc >= 2:
        return ("TIER 1" if sc == 3 else "TIER 2"), "SHORT", sc
    return "SKIP", "NONE", max(lc, sc)

# ============================================================
# RISK ALERTS FROM BOT
# ============================================================

def check_risk_alerts(balance, daily_pnl):
    dd  = max(0, ((START_BAL - balance) / START_BAL) * 100)
    dl  = abs(min(0, (daily_pnl / START_BAL) * 100))

    if   dd >= 7.5: tg(f"STOP TRADING NOW!\n\nDrawdown: {dd:.2f}%\nLimit: 8% | Buffer: {8-dd:.2f}% left\nBalance: ${balance:.2f}\n\nAccount closes at 8%!")
    elif dd >= 7.0: tg(f"DANGER — Drawdown {dd:.2f}%\n\nOnly {8-dd:.2f}% buffer left!\nClose positions now!")
    elif dd >= 5.0: tg(f"Drawdown warning: {dd:.2f}%\n\nApproaching 8% limit. Trade carefully.")

    if   dl >= 3.5: tg(f"STOP TRADING TODAY!\n\nDaily loss: {dl:.2f}%\nLimit: 4% | Buffer: {4-dl:.2f}% left\n\nAccount closes at 4% daily loss!")
    elif dl >= 3.0: tg(f"DANGER — Daily loss {dl:.2f}%\n\nOnly {4-dl:.2f}% left today!")
    elif dl >= 2.0: tg(f"Daily loss warning: {dl:.2f}%\n\n4% limit. {4-dl:.2f}% remaining.")

# ============================================================
# GOOGLE SHEETS
# ============================================================

def init_sheets():
    try:
        if GOOGLE_CREDS_JSON:
            creds_dict = json.loads(GOOGLE_CREDS_JSON)
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(creds_dict, tmp); tmp.close()
            creds = Credentials.from_service_account_file(
                tmp.name, scopes=['https://www.googleapis.com/auth/spreadsheets'])
            os.unlink(tmp.name)
        else:
            creds = Credentials.from_service_account_file(
                'google_credentials.json',
                scopes=['https://www.googleapis.com/auth/spreadsheets'])

        client = gspread.authorize(creds)
        sheet  = client.open_by_key(GOOGLE_SHEET_ID)

        ws_inputs = sheet.worksheet("Inputs")
        try:    ws_risk = sheet.worksheet("Risk Monitor")
        except: ws_risk = None

        print("[SHEETS] Connected")
        return ws_inputs, ws_risk

    except Exception as e:
        print(f"[SHEETS ERROR] {e}")
        traceback.print_exc()
        return None, None

def update_inputs_sheet(ws, price, ma14, vol, avg_vol, rsi,
                        cvd_sig, rsi_sig, tl_sig, ema50, ema200,
                        direction, session, market_cond):
    try:
        values = [
            [round(price,   2)],           # B2  Price
            [round(ma14,    2)],           # B3  MA14
            [round(vol,     2)],           # B4  Volume
            [round(avg_vol, 2)],           # B5  Avg Volume
            [round(rsi,     2)],           # B6  RSI
            ["YES" if rsi_sig else "NO"],  # B7  RSI Signal
            ["YES" if cvd_sig else "NO"],  # B8  CVD Signal
            ["YES" if tl_sig  else "NO"],  # B9  Trendline Signal
            [""],                          # B10 Entry (user fills)
            [""],                          # B11 SL (user fills)
            [""],                          # B12 TP (user fills)
            [direction],                   # B13 Direction
            [round(ema50,  2)],            # B14 EMA50
            [round(ema200, 2)],            # B15 EMA200
            [session],                     # B16 Session
            [market_cond],                 # B17 Market Condition
        ]
        ws.batch_update([{"range": "B2:B17", "values": values}])
        print(f"[SHEET] Inputs updated: ${price:.2f} | {direction} | {session} | {market_cond}")
    except Exception as e:
        print(f"[SHEET ERROR] {e}")
        traceback.print_exc()

def update_risk_sheet(ws_risk, balance, daily_pnl, prof_days):
    if not ws_risk: return
    try:
        dd_pct  = max(0, ((START_BAL - balance) / START_BAL) * 100)
        dl_pct  = abs(min(0, (daily_pnl / START_BAL) * 100))
        tp_pnl  = balance - START_BAL
        p1_pct  = max(0, (tp_pnl / START_BAL) * 100)
        dd_left = balance - (START_BAL * 0.92)
        dl_left = (START_BAL * 0.04) - abs(min(0, daily_pnl))

        dd_s = "STOP NOW" if dd_pct>=7.5 else "DANGER" if dd_pct>=7 else "CAUTION" if dd_pct>=5 else "SAFE"
        dl_s = "STOP NOW" if dl_pct>=3.5 else "DANGER" if dl_pct>=3 else "CAUTION" if dl_pct>=2 else "SAFE"
        p1_s = "COMPLETE" if p1_pct>=8 else "CLOSE" if p1_pct>=6 else "IN PROGRESS"
        dy_s = "COMPLETE" if prof_days>=3 else f"NEED {3-prof_days} MORE"

        ws_risk.batch_update([
            {"range":"B3:E5","values":[
                [f"${balance:.2f}","",f"${START_BAL:.2f}",""],
                [f"${daily_pnl:.2f}","",f"${tp_pnl:.2f}",""],
                [f"{p1_pct:.2f}%","",f"{prof_days} days",""],
            ]},
            {"range":"C16:E19","values":[
                [f"{dd_pct:.2f}%", f"${dd_left:.2f} left", dd_s],
                [f"{dl_pct:.2f}%", f"${dl_left:.2f} left", dl_s],
                [f"{p1_pct:.2f}%", f"${max(0,(START_BAL*1.08)-balance):.2f} to go", p1_s],
                [f"{prof_days}",   f"{max(0,3-prof_days)} more needed", dy_s],
            ]},
        ])
        print(f"[RISK] DD={dd_pct:.2f}% DL={dl_pct:.2f}% -> {dd_s}/{dl_s}")
    except Exception as e:
        print(f"[RISK ERROR] {e}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print(f"MAVEN TRADING BOT v6 — {now_utc().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    # 1. Fetch market data
    df = fetch_klines(SYMBOL, INTERVAL, limit=300)

    # 2. Run all 3 indicators
    cvd_sig, cvd_b, cvd_s                                  = calc_cvd(df)
    rsi_sig, rsi_l, rsi_s, rsi_val, ema50, ema200          = calc_rsi_div(df)
    tl_sig, tl_l, tl_s, bb, bk, br, sr, te50, te200, atr, res, sup = calc_trendline(df)

    ema50  = ema50  or te50
    ema200 = ema200 or te200

    # 3. Score confluence
    tier, direction, conf = score_signals(cvd_b, cvd_s, rsi_l, rsi_s, tl_l, tl_s, ema50, ema200)

    # 4. Market context
    last      = df.iloc[-1]
    price     = float(last['close'])
    vol       = float(last['volume'])
    avg_vol   = float(df['volume'].rolling(20).mean().iloc[-1])
    ma14      = float(df['close'].rolling(14).mean().iloc[-1])
    session   = get_session()
    mkt_cond  = get_market_condition(df)
    cur_candle= int(last['open_time'])

    # 5. Update Google Sheets
    ws_inputs, ws_risk = init_sheets()
    if ws_inputs:
        update_inputs_sheet(ws_inputs, price, ma14, vol, avg_vol, rsi_val,
                            cvd_sig, rsi_sig, tl_sig, ema50, ema200,
                            direction, session, mkt_cond)

    # 6. Risk checks
    check_risk_alerts(balance=START_BAL, daily_pnl=0)  # replace 0 with actual daily PnL if tracked
    update_risk_sheet(ws_risk, balance=START_BAL, daily_pnl=0, prof_days=0)

    # 7. Trade alert if TIER 1 or TIER 2
    last_candle = load_last_candle()
    if tier in ("TIER 1", "TIER 2") and cur_candle != last_candle:
        res_s = f"${res:.2f}" if not np.isnan(res) else "N/A"
        sup_s = f"${sup:.2f}" if not np.isnan(sup) else "N/A"
        msg = (
            f"{tier} SIGNAL — {SYMBOL}\n"
            f"{'='*30}\n\n"
            f"{direction} @ ${price:.2f}\n"
            f"Session: {session} | {mkt_cond}\n\n"
            f"Signals ({conf}/3):\n"
            f"  CVD:       {'YES' if cvd_sig else 'NO'}\n"
            f"  RSI DIV:   {'YES' if rsi_sig else 'NO'}\n"
            f"  TRENDLINE: {'YES' if tl_sig  else 'NO'}\n\n"
            f"Trendline:\n"
            f"  Bull Break:  {'YES' if bb else 'NO'}\n"
            f"  Bull Retest: {'YES' if br else 'NO'}\n"
            f"  Bear Break:  {'YES' if bk else 'NO'}\n"
            f"  Bear Retest: {'YES' if sr else 'NO'}\n"
            f"  Resistance:  {res_s}\n"
            f"  Support:     {sup_s}\n\n"
            f"RSI:   {rsi_val:.1f}\n"
            f"EMA50: ${ema50:.2f} | EMA200: ${ema200:.2f}\n"
            f"ATR:   ${atr:.2f}\n\n"
            f"CHECK CHART NOW"
        )
        tg(msg)
        save_last_candle(cur_candle)
        print(f"[ALERT SENT] {tier}")
    else:
        print(f"[NO ALERT] Tier={tier} | Same candle={cur_candle == last_candle}")

    # 8. Hourly status at :00
    if now_utc().minute < 2:
        tg(
            f"[STATUS] {now_utc().strftime('%H:%M')} UTC\n\n"
            f"BTCUSDT @ ${price:.2f}\n"
            f"Session: {session} | {mkt_cond}\n\n"
            f"Signals ({conf}/3):\n"
            f"  CVD:       {'YES' if cvd_sig else 'NO'}\n"
            f"  RSI DIV:   {'YES' if rsi_sig else 'NO'}\n"
            f"  TRENDLINE: {'YES' if tl_sig  else 'NO'}\n\n"
            f"RSI: {rsi_val:.1f} | Trend: {'BULL' if ema50>ema200 else 'BEAR'}\n"
            f"Tier: {tier}\n\n"
            f"Bot running on GitHub."
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
