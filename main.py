import pandas as pd
import pandas_ta as ta
import requests
import ccxt
import time
import json
import os
import io
from datetime import datetime
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import mplfinance as mpf

# ─────────────────────────────────────────────
# تنظیمات از فایل .env
# ─────────────────────────────────────────────
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ─────────────────────────────────────────────
# فایل ذخیره سیگنال‌ها و آمار
# ─────────────────────────────────────────────
SIGNALS_FILE = "signals_log.json"

def load_signals():
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_signals(data):
    with open(SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=2)

signals_db = load_signals()
last_signals = {}  # جلوگیری از ارسال تکراری

# ─────────────────────────────────────────────
# ارسال پیام متنی تلگرام
# ─────────────────────────────────────────────
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"❌ خطا پیام: {r.text}")
    except Exception as e:
        print(f"Telegram Error: {e}")

# ─────────────────────────────────────────────
# ارسال عکس تلگرام
# ─────────────────────────────────────────────
def send_telegram_photo(image_bytes, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        r = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=30
        )
        if r.status_code != 200:
            print(f"❌ خطا عکس: {r.text}")
    except Exception as e:
        print(f"Photo Error: {e}")

# ─────────────────────────────────────────────
# رسم چارت کامل
# ─────────────────────────────────────────────
def draw_chart(df, symbol, timeframe, direction, p1, p2, p3,
               entry, sl, tp1, tp2, fib_levels):

    # آخرین ۸۰ کندل برای نمایش
    plot_df = df.tail(80).copy()
    plot_df.index = pd.to_datetime(plot_df["timestamp"], unit="ms")

    ohlc = plot_df[["open", "high", "low", "close", "volume"]].copy()
    ohlc.index.name = "Date"

    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])  # چارت اصلی
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # حجم
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # RSI

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines['bottom'].set_color('#333')
        ax.spines['top'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['right'].set_color('#333')

    # رسم کندل‌ها
    idx = range(len(ohlc))
    for i, (_, row) in enumerate(ohlc.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax1.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax1.bar(i, abs(row["close"] - row["open"]),
                bottom=min(row["open"], row["close"]),
                color=color, width=0.6, alpha=0.9)

    # EMA200 روی کل df
    ema_series = df["EMA200"].tail(80).values
    ax1.plot(idx, ema_series, color="#f0a500", linewidth=1.2, label="EMA200", alpha=0.8)

    # پیدا کردن index نقاط پیوت در plot_df
    def find_idx_in_plot(ts_val):
        matches = [i for i, t in enumerate(plot_df["timestamp"].values) if t == ts_val]
        return matches[0] if matches else None

    i1 = find_idx_in_plot(df.loc[p1.name, "timestamp"])
    i2 = find_idx_in_plot(df.loc[p2.name, "timestamp"])
    i3 = find_idx_in_plot(df.loc[p3.name, "timestamp"] if hasattr(p3, 'name') else p3["timestamp"])

    price_key = "high" if direction == "SHORT" else "low"
    color_signal = "#ef5350" if direction == "SHORT" else "#26a69a"

    # رسم نقاط و خطوط پیوت
    for i_pt, p_pt, label in [(i1, p1, "P1"), (i2, p2, "P2"), (i3, p3, "P3")]:
        if i_pt is not None:
            px = p_pt[price_key] if isinstance(p_pt, pd.Series) else p_pt[price_key]
            ax1.scatter(i_pt, px, color=color_signal, s=80, zorder=5)
            ax1.annotate(label, (i_pt, px),
                         textcoords="offset points", xytext=(0, 8 if direction=="SHORT" else -14),
                         color=color_signal, fontsize=9, fontweight="bold")

    if i1 is not None and i2 is not None and i3 is not None:
        pts_x = [i1, i2, i3]
        pts_y = [p1[price_key], p2[price_key], p3[price_key] if isinstance(p3, pd.Series) else p3[price_key]]
        ax1.plot(pts_x, pts_y, "--", color=color_signal, linewidth=1.2, alpha=0.7)

    # خطوط فیبوناچی
    fib_colors = ["#aaaaaa", "#f0a500", "#7b68ee", "#00bcd4"]
    for (level, price), fc in zip(fib_levels.items(), fib_colors):
        ax1.axhline(price, color=fc, linewidth=0.7, linestyle=":", alpha=0.6)
        ax1.text(len(ohlc)-1, price, f" Fib {level}", color=fc, fontsize=7, va="center")

    # خطوط SL / TP
    ax1.axhline(entry, color="#ffffff", linewidth=1.0, linestyle="-", alpha=0.9)
    ax1.axhline(tp1,   color="#26a69a", linewidth=1.0, linestyle="--", alpha=0.9)
    ax1.axhline(tp2,   color="#00e676", linewidth=1.0, linestyle="--", alpha=0.9)
    ax1.axhline(sl,    color="#ef5350", linewidth=1.0, linestyle="--", alpha=0.9)

    ax1.text(len(ohlc)-1, entry, " Entry", color="#ffffff", fontsize=8, va="center")
    ax1.text(len(ohlc)-1, tp1,   " TP1",   color="#26a69a", fontsize=8, va="center")
    ax1.text(len(ohlc)-1, tp2,   " TP2",   color="#00e676", fontsize=8, va="center")
    ax1.text(len(ohlc)-1, sl,    " SL",    color="#ef5350", fontsize=8, va="center")

    arrow_dir = "▼" if direction == "SHORT" else "▲"
    title_color = "#ef5350" if direction == "SHORT" else "#26a69a"
    ax1.set_title(f"{arrow_dir} {direction} Signal | {symbol} | {timeframe}",
                  color=title_color, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(facecolor="#0d1117", edgecolor="#333", labelcolor="#aaaaaa", fontsize=8)
    ax1.set_ylabel("Price", color="#aaaaaa", fontsize=9)

    # حجم
    vol_colors = ["#26a69a" if ohlc["close"].iloc[i] >= ohlc["open"].iloc[i] else "#ef5350"
                  for i in range(len(ohlc))]
    ax2.bar(idx, ohlc["volume"].values, color=vol_colors, alpha=0.7, width=0.6)
    vol_ma = pd.Series(ohlc["volume"].values).rolling(20).mean()
    ax2.plot(idx, vol_ma, color="#f0a500", linewidth=1.0)
    ax2.set_ylabel("Volume", color="#aaaaaa", fontsize=8)

    # RSI
    rsi_vals = df["RSI"].tail(80).values
    ax3.plot(idx, rsi_vals, color="#7b68ee", linewidth=1.2)
    ax3.axhline(70, color="#ef5350", linewidth=0.7, linestyle="--", alpha=0.5)
    ax3.axhline(30, color="#26a69a", linewidth=0.7, linestyle="--", alpha=0.5)
    ax3.fill_between(idx, rsi_vals, 70, where=[r > 70 for r in rsi_vals],
                     alpha=0.15, color="#ef5350")
    ax3.fill_between(idx, rsi_vals, 30, where=[r < 30 for r in rsi_vals],
                     alpha=0.15, color="#26a69a")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color="#aaaaaa", fontsize=8)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf.read()

# ─────────────────────────────────────────────
# محاسبه سطوح فیبوناچی
# ─────────────────────────────────────────────
def calc_fibonacci(p1_price, p3_price, direction):
    diff = abs(p3_price - p1_price)
    levels = {}
    ratios = {"0.236": 0.236, "0.382": 0.382, "0.5": 0.5, "0.618": 0.618}
    for name, ratio in ratios.items():
        if direction == "SHORT":
            levels[name] = p3_price - diff * ratio
        else:
            levels[name] = p3_price + diff * ratio
    return levels

# ─────────────────────────────────────────────
# محاسبه SL / TP
# ─────────────────────────────────────────────
def calc_targets(entry, direction, atr):
    if direction == "SHORT":
        sl  = entry + atr * 1.5
        tp1 = entry - atr * 2.0
        tp2 = entry - atr * 3.5
    else:
        sl  = entry - atr * 1.5
        tp1 = entry + atr * 2.0
        tp2 = entry + atr * 3.5
    return sl, tp1, tp2

# ─────────────────────────────────────────────
# پیوت‌یابی
# ─────────────────────────────────────────────
def find_pivots(df, order=5):
    df = df.copy()
    df["pivot_high"] = None
    df["pivot_low"]  = None
    for i in range(order, len(df) - order):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if all(h > df["high"].iloc[i-j] and h > df["high"].iloc[i+j] for j in range(1, order+1)):
            df.at[df.index[i], "pivot_high"] = h
        if all(l < df["low"].iloc[i-j]  and l < df["low"].iloc[i+j]  for j in range(1, order+1)):
            df.at[df.index[i], "pivot_low"]  = l
    return df

# ─────────────────────────────────────────────
# پیگیری نتیجه سیگنال‌های قبلی
# ─────────────────────────────────────────────
def track_open_signals(current_price_map):
    global signals_db
    now = time.time()
    changed = False

    for key, sig in signals_db.items():
        if sig.get("result") is not None:
            continue

        symbol = sig["symbol"]
        if symbol not in current_price_map:
            continue

        price = current_price_map[symbol]
        tp1   = sig["tp1"]
        tp2   = sig["tp2"]
        sl    = sig["sl"]
        direction = sig["direction"]

        result = None
        if direction == "SHORT":
            if price <= tp1:
                result = "TP1"
            elif price <= tp2:
                result = "TP2"
            elif price >= sl:
                result = "SL"
        else:
            if price >= tp1:
                result = "TP1"
            elif price >= tp2:
                result = "TP2"
            elif price <= sl:
                result = "SL"

        # بعد از ۴۸ ساعت بدون نتیجه → منقضی
        if result is None and (now - sig["timestamp"]) > 48 * 3600:
            result = "EXPIRED"

        if result:
            sig["result"] = result
            sig["result_time"] = datetime.utcnow().isoformat()
            changed = True

            emoji = "✅" if result in ("TP1","TP2") else ("❌" if result=="SL" else "⏰")
            send_telegram_message(
                f"{emoji} *نتیجه سیگنال قبلی*\n\n"
                f"🪙 {symbol} | {sig['direction']}\n"
                f"📌 نتیجه: *{result}*\n"
                f"⏱ تایم‌فریم: {sig['timeframe']}"
            )

    if changed:
        save_signals(signals_db)

# ─────────────────────────────────────────────
# گزارش آمار وین ریت
# ─────────────────────────────────────────────
def win_rate_report():
    wins = losses = expired = 0
    for sig in signals_db.values():
        r = sig.get("result")
        if r in ("TP1","TP2"):
            wins += 1
        elif r == "SL":
            losses += 1
        elif r == "EXPIRED":
            expired += 1

    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    pending = len([s for s in signals_db.values() if s.get("result") is None])

    send_telegram_message(
        f"📊 *گزارش عملکرد ربات*\n\n"
        f"✅ بردها: {wins}\n"
        f"❌ باخت‌ها: {losses}\n"
        f"⏰ منقضی: {expired}\n"
        f"⏳ در انتظار: {pending}\n\n"
        f"🎯 *وین ریت: {wr:.1f}%*\n"
        f"📈 کل سیگنال‌های بررسی‌شده: {total}"
    )

# ─────────────────────────────────────────────
# آنالیز اصلی
# ─────────────────────────────────────────────
def analyze(symbol, timeframe, df):
    global signals_db, last_signals

    if len(df) < 250:
        return

    df = df.copy()
    df["RSI"]    = ta.rsi(df["close"], length=14)
    df["EMA200"] = ta.ema(df["close"], length=200)
    df["ATR"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["VOL_MA"] = df["volume"].rolling(20).mean()

    df = find_pivots(df, order=5)

    def check_direction(direction):
        pk = "pivot_high" if direction == "SHORT" else "pivot_low"
        price_key = "high" if direction == "SHORT" else "low"

        pivots = df[df[pk].notna()].iloc[:-1]  # پیوت‌های قطعی (نه آخری)
        if len(pivots) < 2:
            return

        p1 = pivots.iloc[-2]
        p2 = pivots.iloc[-1]

        # P3: بالاترین/پایین‌ترین ۱۰ کندل اخیر
        recent = df.tail(10)
        if direction == "SHORT":
            p3_idx = recent["high"].idxmax()
        else:
            p3_idx = recent["low"].idxmin()
        p3 = df.loc[p3_idx]

        if not (pivots.index[-2] < pivots.index[-1] < p3_idx):
            return

        # شرط قیمتی
        if direction == "SHORT":
            price_cond = p1[price_key] < p2[price_key] < p3[price_key]
        else:
            price_cond = p1[price_key] > p2[price_key] > p3[price_key]

        if not price_cond:
            return

        # واگرایی RSI
        if direction == "SHORT":
            rsi_cond = (p1["RSI"] > p2["RSI"] > p3["RSI"]) and (p1["RSI"] - p3["RSI"]) >= 5
        else:
            rsi_cond = (p1["RSI"] < p2["RSI"] < p3["RSI"]) and (p3["RSI"] - p1["RSI"]) >= 5

        if not rsi_cond:
            return

        # فیلتر EMA200
        if direction == "SHORT":
            trend_cond = p3["close"] > p3["EMA200"]
        else:
            trend_cond = p3["close"] < p3["EMA200"]

        if not trend_cond:
            return

        # فیلتر حجم
        if p3["volume"] <= p3["VOL_MA"]:
            return

        # تقارن زمانی
        idx1 = df.index.get_loc(p1.name)
        idx2 = df.index.get_loc(p2.name)
        idx3 = df.index.get_loc(p3_idx)
        d1 = idx2 - idx1
        d2 = idx3 - idx2
        if d1 == 0 or d2 == 0:
            return
        if (abs(d1 - d2) / max(d1, d2)) >= 0.40:
            return

        # بررسی فیبوناچی: P3 باید نزدیک 1.272 یا 1.618 از P1→P2 باشد
        move = abs(p2[price_key] - p1[price_key])
        fib_127 = p2[price_key] + (move * 1.272) * (1 if direction=="SHORT" else -1)
        fib_162 = p2[price_key] + (move * 1.618) * (1 if direction=="SHORT" else -1)
        tolerance = move * 0.15
        near_fib = (abs(p3[price_key] - fib_127) < tolerance or
                    abs(p3[price_key] - fib_162) < tolerance)
        if not near_fib:
            return

        # جلوگیری از سیگنال تکراری
        signal_key = f"{symbol}_{timeframe}_{direction}"
        if last_signals.get(signal_key) == p3_idx:
            return
        last_signals[signal_key] = p3_idx

        # محاسبه targets
        entry = p3["close"]
        atr   = p3["ATR"]
        sl, tp1, tp2 = calc_targets(entry, direction, atr)
        rr = round(abs(tp1 - entry) / abs(sl - entry), 2)

        # سطوح فیبوناچی برای چارت
        fib_levels = calc_fibonacci(p1[price_key], p3[price_key], direction)

        # ذخیره سیگنال در دیتابیس
        sig_id = f"{symbol}_{timeframe}_{direction}_{int(time.time())}"
        signals_db[sig_id] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr": rr,
            "timestamp": time.time(),
            "result": None
        }
        save_signals(signals_db)

        # رسم چارت
        img_bytes = draw_chart(df, symbol, timeframe, direction,
                                p1, p2, p3, entry, sl, tp1, tp2, fib_levels)

        # کپشن پیام
        arrow = "🔴" if direction == "SHORT" else "🟢"
        caption = (
            f"{arrow} *{direction} SIGNAL*\n\n"
            f"🪙 *{symbol}* | ⏱ {timeframe}\n"
            f"📊 الگو: Three Drives + RSI Divergence + Fibonacci\n\n"
            f"📌 *ورود:* `{entry:.4f}`\n"
            f"🎯 *TP1:* `{tp1:.4f}`\n"
            f"🎯 *TP2:* `{tp2:.4f}`\n"
            f"🛑 *SL:*  `{sl:.4f}`\n"
            f"⚖️ *R/R:* `1:{rr}`\n\n"
            f"📈 RSI P1:{p1['RSI']:.1f} → P2:{p2['RSI']:.1f} → P3:{p3['RSI']:.1f}\n"
            f"🔥 حجم، تقارن زمانی، فیبوناچی و EMA200 تأیید شد"
        )

        send_telegram_photo(img_bytes, caption)
        print(f"✅ سیگنال ارسال شد: {symbol} {direction}")

    check_direction("SHORT")
    check_direction("LONG")

# ─────────────────────────────────────────────
# اجرای اصلی
# ─────────────────────────────────────────────
def run_bot():
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'}
    })

    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT',
        'BNB/USDT', 'XRP/USDT', 'DOGE/USDT',
        'ADA/USDT', 'AVAX/USDT', 'LINK/USDT'
    ]
    timeframe = '1h'
    report_counter = 0

    send_telegram_message(
        "✅ *ربات Three Drives Pro روشن شد!*\n\n"
        "🔍 تایم‌فریم: 1H\n"
        "📊 فیلترها: Three Drives + RSI + Fibonacci + EMA200 + Volume\n"
        "📸 چارت کامل با SL/TP ارسال می‌شود\n"
        "🤖 نتایج سیگنال‌ها پیگیری می‌شود"
    )

    print("🤖 ربات استارت شد...")

    while True:
        current_prices = {}

        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=350)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                current_prices[symbol] = df["close"].iloc[-1]
                analyze(symbol, timeframe, df)
                time.sleep(1)
            except Exception as e:
                print(f"❌ خطا {symbol}: {e}")
                time.sleep(5)

        # پیگیری سیگنال‌های باز
        track_open_signals(current_prices)

        # هر ۲۴ ساعت گزارش آمار
        report_counter += 1
        if report_counter >= 24:
            win_rate_report()
            report_counter = 0

        print(f"🔄 دور کامل شد. انتظار ۱ ساعت...")
        time.sleep(3600)

# استارت
run_bot()
