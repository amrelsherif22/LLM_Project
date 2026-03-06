import asyncio
import json
import logging
import os
import random
import tempfile
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    Runner,
    RunContextWrapper,
    function_tool,
    input_guardrail,
    output_guardrail,
    trace,
)
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)


load_dotenv(override=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("StockResearch")

NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "")
TODAY: str        = datetime.now().strftime("%B %d, %Y")
CUTOFF_48H: str   = (
    datetime.now(timezone.utc) - timedelta(hours=48)
).strftime("%Y-%m-%dT%H:%M:%SZ")



@dataclass
class StepRecord:
    name:     str
    status:   str
    duration: float = 0.0
    detail:   str   = ""


@dataclass
class PipelineMonitor:
    run_id:       str              = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    records:      List[StepRecord] = field(default_factory=list)
    judge_scores: List[int]        = field(default_factory=list)
    _timers:      Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self._timers[name] = time.monotonic()
        self.records.append(StepRecord(name=name, status="start"))
        log.info("▶  %s", name)

    def ok(self, name: str, detail: str = "") -> None:
        duration = time.monotonic() - self._timers.get(name, time.monotonic())
        self.records.append(StepRecord(name=name, status="ok", duration=duration, detail=detail))
        log.info("✓  %s  (%.1fs)  %s", name, duration, detail)

    def err(self, name: str, detail: str = "") -> None:
        duration = time.monotonic() - self._timers.get(name, time.monotonic())
        self.records.append(StepRecord(name=name, status="err", duration=duration, detail=detail))
        log.error("✗  %s  (%.1fs)  %s", name, duration, detail)

    def judge_scored(self, score: int) -> None:
        self.judge_scores.append(score)
        log.info("⚖  Judge score: %d/10", score)

    def summary(self) -> None:
        ok_steps  = [r for r in self.records if r.status == "ok"]
        err_steps = [r for r in self.records if r.status == "err"]
        total_sec = sum(r.duration for r in self.records)
        avg_judge = (sum(self.judge_scores) / len(self.judge_scores)) if self.judge_scores else 0

        print("\n" + "═" * 50)
        print(f"  RUN SUMMARY — {self.run_id}")
        print("═" * 50)
        print(f"  Steps OK   : {len(ok_steps)}")
        print(f"  Steps ERR  : {len(err_steps)}" + (f"  ← {[r.name for r in err_steps]}" if err_steps else ""))
        print(f"  Total time : {total_sec:.1f}s")
        print(f"  Judge avg  : {avg_judge:.1f}/10  ({len(self.judge_scores)} attempt(s))")
        if err_steps:
            print("\n  Error details:")
            for r in err_steps:
                print(f"    • {r.name}: {r.detail[:120]}")
        print("═" * 50 + "\n")

        log_path = f"run_{self.run_id}.json"
        try:
            with open(log_path, "w") as f:
                json.dump({
                    "run_id":       self.run_id,
                    "judge_scores": self.judge_scores,
                    "steps":        [r.__dict__ for r in self.records],
                }, f, indent=2)
            log.info("Run log saved → %s", log_path)
        except Exception as e:
            log.warning("Could not save run log: %s", e)


monitor = PipelineMonitor()


_LAST_YF_CALL: float = 0.0
_YF_MIN_GAP:   float = 0.35


def _throttle() -> None:
    global _LAST_YF_CALL
    gap = time.monotonic() - _LAST_YF_CALL
    if gap < _YF_MIN_GAP:
        time.sleep(_YF_MIN_GAP - gap)
    _LAST_YF_CALL = time.monotonic()


def _yf_info(ticker: str, retries: int = 3) -> dict:

    delay = 1.5
    for attempt in range(retries):
        _throttle()
        try:
            return yf.Ticker(ticker).info or {}
        except Exception as exc:
            log.warning("yf.info attempt %d/%d for %s: %s", attempt + 1, retries, ticker, exc)
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 0.5))
                delay *= 2
    return {}


def _yf_hist(ticker: str, period: str, retries: int = 3):

    import pandas as pd
    delay = 1.5
    for attempt in range(retries):
        _throttle()
        try:
            return yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as exc:
            log.warning("yf.hist attempt %d/%d for %s (%s): %s", attempt + 1, retries, ticker, period, exc)
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 0.5))
                delay *= 2
    return pd.DataFrame()



SECTOR_LAUNCH_DELAYS = [0, 8, 16]


async def _run_with_429_retry(agent, prompt: str, max_turns: int, label: str,
                               max_retries: int = 4) -> str:

    delay = 15.0
    for attempt in range(max_retries + 1):
        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            return result.final_output
        except Exception as exc:
            err_str = str(exc)
            is_429  = "429" in err_str or "rate limit" in err_str.lower()
            if is_429 and attempt < max_retries:
                wait = delay + random.uniform(0, delay * 0.3)
                log.warning(
                    "429 rate limit on %s (attempt %d/%d) — waiting %.0fs before retry",
                    label, attempt + 1, max_retries + 1, wait
                )
                await asyncio.sleep(wait)
                delay *= 2
            else:
                raise


class RsiResult(BaseModel):
    ticker:    str
    date:      str
    rsi_value: float
    signal:    str


class InputValidation(BaseModel):
    is_valid: bool
    reason:   str


class OutputValidation(BaseModel):
    is_valid: bool
    reason:   str


class JudgeVerdict(BaseModel):
    score:           int    # Overall quality 1-10. We need 7+ to approve.
    has_min_picks:   bool   # True if report has at least 3 real stock picks
    prices_present:  bool   # True if every pick has a real dollar price
    no_placeholders: bool   # True if no '[Insert X]' placeholder text remains
    approved:        bool   # True when score>=7 AND all three bools are True
    critique:        str    # Specific issues to fix — fed back to the Picker on retry


input_guardrail_agent = Agent(
    name="InputGuardrailAgent",
    instructions=(
        "You validate user requests for a stock research tool. "
        "Approve (is_valid=True) anything related to: stocks, investing, markets, "
        "financial research, trading, portfolio analysis, earnings, or macro economy. "
        "Reject (is_valid=False) ONLY if the request has nothing to do with finance "
        "(e.g. recipes, essays) OR explicitly asks for something illegal like insider trading. "
        "When in doubt — approve."
    ),
    output_type=InputValidation,
    model="gpt-4o-mini",
)

output_guardrail_agent = Agent(
    name="OutputGuardrailAgent",
    instructions=(
        "You review an HTML stock research report. "
        "Return is_valid=False ONLY if the report is completely empty "
        "or contains nothing but unfilled placeholder text like '[Insert X here]'. "
        "Return is_valid=True if the report contains ANY real ticker symbol "
        "(AAPL, MSFT, NVDA, XOM, etc.) — even if some data is missing or partial. "
        "Your only job is to catch a completely broken/unrendered template."
    ),
    output_type=OutputValidation,
    model="gpt-4o-mini",
)


@input_guardrail
async def stock_research_input_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str
) -> GuardrailFunctionOutput:
    result     = await Runner.run(input_guardrail_agent, input, context=ctx.context)
    validation = result.final_output
    log.info("Input guardrail: is_valid=%s | %s", validation.is_valid, validation.reason)
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=not validation.is_valid,
    )


@output_guardrail
async def stock_report_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    preview    = output[:2000]
    result     = await Runner.run(output_guardrail_agent, preview, context=ctx.context)
    validation = result.final_output
    log.info("Output guardrail: is_valid=%s | %s", validation.is_valid, validation.reason)
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=not validation.is_valid,
    )
judge_agent = Agent(
    name="ReportJudge",
    instructions=(
        "You are a quality judge for AI-generated stock research reports. "
        "You will receive a finished HTML stock report. "

        "Check these four things and fill in the structured fields: "

        "1. has_min_picks: Does the report contain at least 3 real stock picks "
        "   with actual ticker symbols (AAPL, MSFT, etc.)? Set True if yes. "

        "2. prices_present: Does every pick have a real dollar price like $213.49? "
        "   Set False if any pick is missing a price or shows 'N/A' for price. "

        "3. no_placeholders: Is the report free of '[Insert X]' style text? "
        "   Set True if there are no unfilled placeholders. "

        "4. score: Rate overall quality 1-10. "
        "   10 = perfect professional report. "
        "   7  = acceptable — picks present, prices real, theses make sense. "
        "   4  = poor — missing picks, missing prices, or all generic vague text. "
        "   1  = completely useless or empty. "

        "Set approved=True ONLY when ALL of these are true: "
        "score >= 7 AND has_min_picks=True AND prices_present=True AND no_placeholders=True. "

        "In critique: list the SPECIFIC problems found. Be direct and brief. "
        "Example: 'Pick 3 (XOM) is missing a price. Pick 5 thesis is generic filler.' "
        "If approved, set critique to empty string."
    ),
    output_type=JudgeVerdict,
    model="gpt-4o-mini",
)


async def run_with_judge(combined_research: str, max_retries: int = 2) -> str:

    best_report: str = ""
    best_score:  int = 0
    critique:    str = ""

    for attempt in range(max_retries + 1):
        label = f"attempt {attempt + 1}/{max_retries + 1}"
        log.info("Picker %s", label)
        monitor.start(f"picker_{label}")

        picker_prompt = combined_research
        if critique:
            picker_prompt += (
                f"\n\n⚠️  JUDGE FEEDBACK — FIX THESE IN THIS ATTEMPT:\n{critique}\n"
                f"REMINDER: The report MUST contain AT LEAST 3 complete stock picks, "
                f"each with a real dollar price."
            )

        try:
            result      = await Runner.run(top5_stock_picker, picker_prompt, max_turns=25)
            html_report = result.final_output
            monitor.ok(f"picker_{label}", "HTML generated")
        except OutputGuardrailTripwireTriggered as e:

            monitor.err(f"picker_{label}", f"OutputGuardrail fired: {e}")
            log.warning(
                "Output guardrail fired on %s — Picker produced an empty/broken report. "
                "Will retry with stronger instructions.", label
            )
            critique = (
                (critique + "\n" if critique else "") +
                "CRITICAL: Your previous response was rejected as empty or unrendered. "
                "You MUST output a complete HTML page starting with <!DOCTYPE html> "
                "containing at least 3 real stock picks with real dollar prices. "
                "Do NOT output plain text. Do NOT output an explanation. Output ONLY the HTML."
            )
            continue
        except Exception as e:
            monitor.err(f"picker_{label}", str(e))
            log.error("Picker failed on %s: %s", label, e)
            continue

        log.info("Sending report to Judge (%s)", label)
        monitor.start(f"judge_{label}")
        try:
            judge_result = await Runner.run(judge_agent, html_report, max_turns=5)
            verdict: JudgeVerdict = judge_result.final_output
            monitor.ok(f"judge_{label}", f"score={verdict.score}")
        except Exception as e:
            monitor.err(f"judge_{label}", str(e))
            log.warning("Judge itself failed on %s — using report anyway: %s", label, e)
            return html_report

        monitor.judge_scored(verdict.score)

        if verdict.score > best_score:
            best_score  = verdict.score
            best_report = html_report

        if verdict.approved:
            log.info("✅ Judge approved on %s (score=%d/10)", label, verdict.score)
            return best_report

        log.warning(
            "Judge rejected on %s (score=%d/10). Issues: %s",
            label, verdict.score, verdict.critique[:200]
        )
        critique = verdict.critique

    log.warning(
        "Max retries reached. Using best report (score=%d/10). "
        "Report will still have picks even if imperfect.",
        best_score
    )
    return best_report



SP500_BY_SECTOR: Dict[str, List[str]] = {
    "technology":  ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "ADBE"],
    "energy":      ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "HAL"],
    "healthcare":  ["JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "ISRG"],
    "finance":     ["BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C"],
    "consumer":    ["AMZN", "TSLA", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW"],
    "industrial":  ["HON", "CAT", "DE", "LMT", "RTX", "GE", "UPS", "FDX", "EMR", "ETN"],
}


@function_tool
def resolve_ticker(company_name_or_ticker: str) -> str:

    raw = company_name_or_ticker.strip()
    log.info("resolve_ticker('%s')", raw)
    try:
        info   = _yf_info(raw)
        symbol = info.get("symbol", "")
        name   = info.get("shortName") or info.get("longName", "")
        sector = info.get("sector", "N/A")


        price = round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None

        if price is None:
            fallback = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            price = round(float(fallback), 2) if fallback else None

        if not symbol or not name or price is None:
            return f"INVALID: '{raw}' could not be resolved. Try a different name or ticker."

        return (
            f"VERIFIED TICKER: {symbol} | Company: {name} | "
            f"Sector: {sector} | Price: ${price} | "
            f"Use '{symbol}' for all other tools."
        )
    except Exception as e:
        return f"INVALID: Could not resolve '{raw}': {e}"


@function_tool
def fetch_sp500_sector_tickers(sector: str) -> str:
    key = sector.lower().strip()
    for k in SP500_BY_SECTOR:
        if k in key or key in k:
            tickers = SP500_BY_SECTOR[k]
            log.info("fetch_sp500_sector_tickers('%s') → %d tickers", sector, len(tickers))
            return (
                f"S&P 500 {k} tickers: {', '.join(tickers)}\n"
                f"Run resolve_ticker on each, then the full analysis pipeline. "
                f"This guarantees you have candidates to work with."
            )
    all_tickers = [t for lst in SP500_BY_SECTOR.values() for t in lst]
    return f"All available tickers: {', '.join(all_tickers)}"


@function_tool
def fetch_market_news(topic: str) -> str:
    if not NEWS_API_KEY:
        return (
            f"NEWS_API_KEY not set. Cannot fetch news for '{topic}'. "
            "Use fetch_sp500_sector_tickers as your fallback instead."
        )
    query = urllib.parse.quote(f"{topic} stock earnings revenue guidance")
    url   = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={CUTOFF_48H}&sortBy=publishedAt&pageSize=12&language=en&apiKey={NEWS_API_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        articles = data.get("articles", [])
        if not articles:
            return (
                f"No news found for '{topic}' in the last 48h. "
                "Fallback: call fetch_sp500_sector_tickers with your sector name."
            )
        lines = []
        for a in articles[:12]:
            pub  = a.get("publishedAt", "")[:10]
            src  = a.get("source", {}).get("name", "")
            desc = (a.get("description") or "")[:200]
            lines.append(f"[{pub}] {a['title']} | {src} | {desc}")
        return (
            f"News (last 48h) for '{topic}':\n\n" + "\n\n".join(lines) +
            "\n\nNext: extract company names, call resolve_ticker(name) for each."
        )
    except Exception as e:
        return (
            f"News fetch failed for '{topic}': {e}. "
            "Fallback: call fetch_sp500_sector_tickers with your sector name."
        )


@function_tool
def fetch_stock_quote(verified_ticker: str) -> str:

    log.info("fetch_stock_quote('%s')", verified_ticker)
    try:
        hist = _yf_hist(verified_ticker, "5d")
        if hist.empty:
            return f"No price data for {verified_ticker}."
        latest     = hist.iloc[-1]
        prev       = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
        price      = round(float(latest["Close"]), 2)
        prev_close = round(float(prev["Close"]), 2)
        change     = round(price - prev_close, 2)
        chg_pct    = round((change / prev_close) * 100, 2) if prev_close else 0.0
        return (
            f"Ticker: {verified_ticker} | Date: {hist.index[-1].strftime('%Y-%m-%d')} | "
            f"Price: ${price} | Change: {change:+.2f} ({chg_pct:+.2f}%) | "
            f"Open: ${round(float(latest['Open']),2)} | "
            f"High: ${round(float(latest['High']),2)} | "
            f"Low: ${round(float(latest['Low']),2)} | "
            f"Volume: {int(latest['Volume']):,}"
        )
    except Exception as e:
        return f"Could not fetch quote for {verified_ticker}: {e}"


@function_tool
def fetch_stock_overview(verified_ticker: str) -> str:

    log.info("fetch_stock_overview('%s')", verified_ticker)
    try:
        info = _yf_info(verified_ticker)
        if not info or not info.get("symbol"):
            return f"No fundamental data for {verified_ticker}."
        target = info.get("targetMeanPrice")

        hist    = _yf_hist(verified_ticker, "2d")
        current = round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None
        if current is None:
            current = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )

        upside  = (
            f"{round(((float(target) - float(current)) / float(current)) * 100, 1)}%"
            if target and current and float(current) > 0 else "N/A"
        )
        summary = (info.get("longBusinessSummary") or "N/A")[:300]
        return (
            f"Ticker: {info.get('symbol')} | Name: {info.get('shortName')} | "
            f"Sector: {info.get('sector','N/A')} | Industry: {info.get('industry','N/A')} | "
            f"Market Cap: ${info.get('marketCap',0):,} | "
            f"PE (TTM): {info.get('trailingPE','N/A')} | Forward PE: {info.get('forwardPE','N/A')} | "
            f"EPS (TTM): {info.get('trailingEps','N/A')} | EPS Fwd: {info.get('forwardEps','N/A')} | "
            f"52W High: ${info.get('fiftyTwoWeekHigh','N/A')} | 52W Low: ${info.get('fiftyTwoWeekLow','N/A')} | "
            f"Analyst Target: ${target} | Upside: {upside} | "
            f"Analysts: {info.get('numberOfAnalystOpinions','N/A')} | "
            f"Rec: {str(info.get('recommendationKey','N/A')).upper()} | "
            f"Business: {summary}"
        )
    except Exception as e:
        return f"Could not fetch overview for {verified_ticker}: {e}"


@function_tool
def fetch_rsi_signal(verified_ticker: str) -> RsiResult:

    log.info("fetch_rsi_signal('%s')", verified_ticker)
    try:
        hist = _yf_hist(verified_ticker, "90d")
        if hist.empty or len(hist) < 16:
            return RsiResult(ticker=verified_ticker, date="N/A", rsi_value=0.0,
                             signal="Not enough history (need 16+ days)")
        delta     = hist["Close"].diff()
        gain      = delta.clip(lower=0).rolling(14).mean()
        loss      = (-delta.clip(upper=0)).rolling(14).mean()
        rs        = gain / loss
        rsi_value = round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)
        date_str  = hist.index[-1].strftime("%Y-%m-%d")
        if rsi_value < 30:
            signal = "OVERSOLD — strong potential buy"
        elif rsi_value <= 55:
            signal = "HEALTHY — good entry zone"
        elif rsi_value <= 65:
            signal = "ELEVATED — proceed with caution"
        else:
            signal = "OVERBOUGHT — avoid"
        return RsiResult(ticker=verified_ticker, date=date_str, rsi_value=rsi_value, signal=signal)
    except Exception as e:
        return RsiResult(ticker=verified_ticker, date="N/A", rsi_value=0.0, signal=f"Error: {e}")


@function_tool
def fetch_sma_trend(verified_ticker: str) -> str:

    log.info("fetch_sma_trend('%s')", verified_ticker)
    try:
        hist = _yf_hist(verified_ticker, "150d")
        if hist.empty or len(hist) < 55:
            return f"Not enough data for SMA on {verified_ticker}"
        closes    = hist["Close"]
        sma       = closes.rolling(50).mean()
        sma_now   = round(float(sma.iloc[-1]),   2)
        sma_5ago  = round(float(sma.iloc[-6]),   2)
        price_now = round(float(closes.iloc[-1]), 2)
        trend     = "RISING" if sma_now > sma_5ago  else "FALLING"
        position  = "ABOVE"  if price_now > sma_now else "BELOW"
        verdict   = "BULLISH SETUP" if trend == "RISING" and position == "ABOVE" else "WEAK — avoid"
        return (
            f"Ticker: {verified_ticker} | 50-day SMA: ${sma_now} | "
            f"Trend: {trend} | Price ${price_now} is {position} SMA | Verdict: {verdict}"
        )
    except Exception as e:
        return f"Could not calculate SMA for {verified_ticker}: {e}"


@function_tool
def fetch_macd_signal(verified_ticker: str) -> str:

    log.info("fetch_macd_signal('%s')", verified_ticker)
    try:
        hist = _yf_hist(verified_ticker, "9mo")
        if hist.empty or len(hist) < 40:
            return f"Not enough data for MACD on {verified_ticker}"
        closes      = hist["Close"]
        macd_line   = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line
        macd_val    = round(float(macd_line.iloc[-1]),   4)
        sig_val     = round(float(signal_line.iloc[-1]), 4)
        hist_val    = round(float(histogram.iloc[-1]),   4)
        prev_hist   = float(histogram.iloc[-2])
        if macd_val > sig_val and hist_val > 0 and hist_val > prev_hist:
            status = "BULLISH CROSSOVER — strong momentum"
        elif macd_val > sig_val and hist_val > 0:
            status = "BULLISH — above signal line"
        elif hist_val > prev_hist:
            status = "MOMENTUM BUILDING — histogram rising"
        elif macd_val < sig_val:
            status = "BEARISH — avoid"
        else:
            status = "NEUTRAL"
        return (
            f"Ticker: {verified_ticker} | MACD: {macd_val} | "
            f"Signal: {sig_val} | Histogram: {hist_val} | Status: {status}"
        )
    except Exception as e:
        return f"Could not calculate MACD for {verified_ticker}: {e}"


@function_tool
def fetch_earnings_calendar(verified_ticker: str) -> str:

    log.info("fetch_earnings_calendar('%s')", verified_ticker)
    try:
        stock = yf.Ticker(verified_ticker)
        parts = [f"Ticker: {verified_ticker}"]
        try:
            cal = stock.calendar
            if cal and isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    parts.append(f"Next Earnings: {ed[0] if isinstance(ed, list) else ed}")
        except Exception:
            pass
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                row = hist.iloc[0]
                parts.append(
                    f"Last EPS Actual: {row.get('epsActual','N/A')} | "
                    f"Estimated: {row.get('epsEstimate','N/A')} | "
                    f"Surprise: {row.get('surprisePercent','N/A')}%"
                )
        except Exception:
            pass
        return " | ".join(parts) if len(parts) > 1 else f"No earnings data for {verified_ticker}"
    except Exception as e:
        return f"Could not fetch earnings for {verified_ticker}: {e}"


@function_tool
def fetch_top_gainers_losers() -> str:

    log.info("fetch_top_gainers_losers()")
    ALWAYS_EXCLUDE = ["GME", "AMC", "BBBY", "SPCE", "MULN", "RIVN", "LCID", "NKLA"]
    try:
        watchlist = [t for lst in SP500_BY_SECTOR.values() for t in lst]
        movers    = []
        for ticker in watchlist[:30]:
            try:
                hist = _yf_hist(ticker, "2d")
                if hist.empty or len(hist) < 2:
                    continue
                pct = ((float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2]))
                       / float(hist["Close"].iloc[-2])) * 100
                if pct >= 4.0:
                    movers.append(f"  AVOID: {ticker} — up {pct:.1f}% today")
            except Exception:
                continue
        result = ["EXCLUSION LIST — do NOT pick these (already moved big today):"]
        result.extend(movers if movers else ["  No major movers detected today."])
        result.append(f"\nAlso always exclude: {', '.join(ALWAYS_EXCLUDE)}")
        return "\n".join(result)
    except Exception as e:
        return f"Could not scan movers: {e}. Always exclude: {', '.join(ALWAYS_EXCLUDE)}"


@function_tool
def debug_yfinance(ticker: str) -> str:

    try:
        info  = _yf_info(ticker)
        hist  = _yf_hist(ticker, "5d")
        price = round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else "NO HISTORY"
        return (
            f"DEBUG {ticker}: symbol={info.get('symbol','MISSING')} | "
            f"name={info.get('shortName','MISSING')} | "
            f"currentPrice={info.get('currentPrice','MISSING')} | "
            f"previousClose={info.get('previousClose','MISSING')} | "
            f"historyPrice={price}"
        )
    except Exception as e:
        return f"DEBUG FAILED for {ticker}: {e}"


@function_tool
def open_report_in_browser(html_body: str) -> Dict[str, str]:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stock_report_{ts}.html"
    tmp_path = os.path.join(tempfile.gettempdir(), filename)
    cwd_path = os.path.join(os.getcwd(), filename)
    for path in [tmp_path, cwd_path]:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_body)
            log.info("Report saved → %s", path)
        except Exception as e:
            log.warning("Could not save to %s: %s", path, e)
    try:
        webbrowser.open(f"file:///{tmp_path.replace(os.sep, '/')}")
    except Exception:
        log.warning("Browser open failed — open manually: %s", cwd_path)
    return {"status": "ok", "path": tmp_path, "backup": cwd_path}



SECTOR_RULES = (
    "\n\n── DATA RULES (non-negotiable) ──"
    "\n1. Call resolve_ticker(name) before ANY other tool for every company."
    "\n2. Use ONLY the ticker resolve_ticker returns. Never guess."
    "\n3. If resolve_ticker returns INVALID → skip that company, try another."
    "\n4. FALLBACK: If fewer than 3 tickers verified from news:"
    "\n   Call fetch_sp500_sector_tickers(your_sector) and resolve_ticker on each."
    "\n   Keep going until you have AT LEAST 4 verified tickers."
    "\n   This is mandatory — you MUST always deliver at least 3 candidates."
    "\n5. Prices MUST come from fetch_stock_quote. Never invent numbers."
    "\n6. Names MUST come from resolve_ticker. Never invent names."

    "\n\n── SELECTION CRITERIA ──"
    "\nAVOID: RSI > 65 (overbought)"
    "\nAVOID: FALLING SMA or price below 50-day SMA"
    "\nAVOID: BEARISH MACD"
    "\nAVOID: Any ticker on the exclusion list"
    "\nTARGET: RSI 30-55 | RISING SMA | Price ABOVE SMA | BULLISH MACD | Analyst target ≥5% upside"
    "\nIf no stock meets all criteria, pick the best available — ALWAYS return at least 3."

    "\n\n── OUTPUT FORMAT ──"
    "\nFor each candidate return:"
    "\n  ticker | name | price | RSI | SMA verdict | MACD | analyst target | upside%"
    "\n  | EPS | news catalyst | thesis | risk | conviction 1-10"
)

ALL_TOOLS = [
    resolve_ticker, fetch_market_news, fetch_stock_quote, fetch_stock_overview,
    fetch_rsi_signal, fetch_sma_trend, fetch_macd_signal, fetch_earnings_calendar,
    fetch_top_gainers_losers, fetch_sp500_sector_tickers, debug_yfinance,
]



def _make_analyst(name: str, sector_label: str, news_query: str, fallback_sector: str) -> Agent:
    return Agent(
        name=name,
        instructions=(
            f"You are a specialist equity analyst for {sector_label} stocks."
            f"\n\nPLANNING LOOP — follow these steps in order:"
            f"\nSTEP 1: Call fetch_top_gainers_losers() → save the exclusion list."
            f"\nSTEP 2: Call fetch_market_news('{news_query}')."
            f"\nSTEP 3: Extract 6-8 company names from the headlines."
            f"\nSTEP 4: Call resolve_ticker(name) for each company."
            f"         Keep only VERIFIED results."
            f"         If fewer than 3 are VERIFIED → call fetch_sp500_sector_tickers('{fallback_sector}')"
            f"         and resolve_ticker on each until you have ≥4 verified tickers."
            f"\nSTEP 5: Remove any tickers on the exclusion list."
            f"\nSTEP 6: For each remaining VERIFIED ticker run IN ORDER:"
            f"         fetch_stock_quote → fetch_stock_overview → fetch_rsi_signal"
            f"         → fetch_sma_trend → fetch_macd_signal → fetch_earnings_calendar."
            + SECTOR_RULES
        ),
        tools=ALL_TOOLS,
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0),
    )


tech_sector_analyst = _make_analyst(
    name="TechSectorAnalyst",
    sector_label="Technology (AI, semiconductors, cloud, software)",
    news_query="technology AI semiconductor software cloud earnings",
    fallback_sector="technology",
)

energy_sector_analyst = _make_analyst(
    name="EnergySectorAnalyst",
    sector_label="Energy (oil, gas, renewables, utilities)",
    news_query="energy oil gas renewable utility earnings guidance",
    fallback_sector="energy",
)

healthcare_sector_analyst = _make_analyst(
    name="HealthcareSectorAnalyst",
    sector_label="Healthcare (biotech, pharma, medical devices)",
    news_query="healthcare biotech pharma FDA approval clinical trial earnings",
    fallback_sector="healthcare",
)



macro_news_analyst = Agent(
    name="MacroNewsAnalyst",
    instructions=(
        "You are a macro market analyst setting context for tomorrow's trading session."
        "\nSTEP 1: Call fetch_market_news('Federal Reserve interest rates inflation CPI')."
        "\nSTEP 2: Call fetch_market_news('S&P 500 Nasdaq market outlook earnings season')."
        "\nSTEP 3: Call fetch_market_news('corporate earnings guidance Wall Street analyst')."
        "\nSTEP 4: Call fetch_top_gainers_losers() to gauge today's risk appetite."
        "\nWrite 300-500 words: overall sentiment, key tailwinds, key headwinds, "
        "which sectors look favored, top 2-3 risks to watch. Cite specific headlines."
    ),
    tools=[fetch_market_news, fetch_top_gainers_losers],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0),
)



top5_stock_picker = Agent(
    name="Top5StockPicker",
    instructions=(
        "You are a senior portfolio manager selecting stocks for tomorrow's open. "
        "You receive research from 3 sector analysts and a macro backdrop. "
        "All tickers are verified. All prices are real — copy them exactly, never change them."

        "\n\nSCORING MATRIX — score every candidate on these 7 criteria (1 point each):"
        "\n  [1] RSI between 30-55"
        "\n  [2] 50-day SMA RISING and price ABOVE it"
        "\n  [3] MACD BULLISH or histogram rising"
        "\n  [4] Analyst target ≥5% above current price"
        "\n  [5] Positive EPS (TTM)"
        "\n  [6] Real news catalyst from last 48h"
        "\n  [7] NOT on the exclusion list"
        "\nReject stocks scoring 2 or below. Rank the rest. Pick the top 5."

        "\n\nMINIMUM OUTPUT RULE (this is mandatory, never skip it):"
        "\nYou MUST include AT LEAST 3 complete stock picks in the report. "
        "\nIf you have fewer than 3 high-scoring candidates, include the best ones you have "
        "\nand add a note explaining their limitations — but NEVER output fewer than 3 picks. "
        "\nA solid report with 3 real picks is far better than an empty or incomplete report."

        "\n\nFOR EACH PICK include:"
        "\n  1. Ticker + full company name (exactly as from research)"
        "\n  2. Current price (copy exactly — never round or change)"
        "\n  3. Score X/7 and which criteria were met"
        "\n  4. RSI value + signal, SMA verdict, MACD status"
        "\n  5. Analyst target + upside %"
        "\n  6. EPS (TTM)"
        "\n  7. News catalyst (specific headline + date)"
        "\n  8. Investment thesis (why this stock, why NOW)"
        "\n  9. Key risk"
        "\n 10. Entry strategy: buy at open / limit at $X / wait for dip to $Y"
        "\n 11. Conviction: HIGH (6-7/7) or MEDIUM (4-5/7)"

        f"\n\nFORMAT: Output a complete self-contained HTML page with embedded CSS. "
        f"Today's date is {TODAY} — put it in the header, never write a placeholder. "
        f"Include a macro context section at the top. "
        f"Include a legal disclaimer at the bottom. "
        f"Never write '[Insert X]' anywhere — every field must have real data or 'N/A'."
    ),
    model="o3-mini",
    model_settings=ModelSettings(temperature=1),
    output_guardrails=[stock_report_output_guardrail],
)



report_opener = Agent(
    name="ReportOpener",
    instructions=(
        "You receive a complete HTML investment report. "
        "Call open_report_in_browser exactly once with the full HTML. "
        "Do not summarize, truncate, or modify it in any way."
    ),
    tools=[open_report_in_browser],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0),
)



orchestrator = Agent(
    name="Orchestrator",
    instructions="You are the entry point for a stock research pipeline. Confirm the request is valid.",
    model="gpt-4o-mini",
    input_guardrails=[stock_research_input_guardrail],
)


async def run_stock_research_pipeline(user_prompt: Optional[str] = None) -> None:
    if user_prompt is None:
        user_prompt = (
            "Run a full stock research analysis for tomorrow's trading session. "
            "Find the best opportunities across technology, energy, and healthcare. "
            "Use technical analysis, fundamentals, and latest news catalysts. "
            "You MUST include at least 3 complete stock picks with real prices in the report."
        )

    log.info("═══ PIPELINE START — %s ═══", TODAY)

    monitor.start("startup")
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — will use S&P500 fallback for all sectors")
    if datetime.now().weekday() >= 5:
        log.warning("Weekend run — prices will be from last trading day (Friday)")
    monitor.ok("startup")

    monitor.start("input_guardrail")
    try:
        await Runner.run(orchestrator, user_prompt, max_turns=2)
        monitor.ok("input_guardrail")
    except InputGuardrailTripwireTriggered as e:
        monitor.err("input_guardrail", str(e))
        print(f"\n❌ Request blocked by input guardrail: {e}\n")
        monitor.summary()
        return


    research_prompt = (
        "Follow your planning steps exactly. "
        "Resolve every ticker before using it. "
        "Use exact prices from fetch_stock_quote. "
        "Return AT LEAST 3 candidates no matter what — use the S&P500 fallback if needed."
    )

    async def _staggered_sector(agent, prompt: str, max_turns: int,
                                 label: str, delay_s: int) -> str:
        if delay_s > 0:
            log.info("Staggering %s by %ds to spread TPM load", label, delay_s)
            await asyncio.sleep(delay_s)
        monitor.start(label)
        try:
            output = await _run_with_429_retry(agent, prompt, max_turns, label)
            monitor.ok(label, f"{len(output)} chars")
            return output
        except Exception as exc:
            monitor.err(label, str(exc))
            raise

    monitor.start("parallel_research")
    with trace("Parallel Sector Research"):
        results = await asyncio.gather(
            _staggered_sector(tech_sector_analyst,       research_prompt, 50, "tech_sector",       SECTOR_LAUNCH_DELAYS[0]),
            _staggered_sector(energy_sector_analyst,     research_prompt, 50, "energy_sector",     SECTOR_LAUNCH_DELAYS[1]),
            _staggered_sector(healthcare_sector_analyst, research_prompt, 50, "healthcare_sector", SECTOR_LAUNCH_DELAYS[2]),
            _run_with_429_retry(macro_news_analyst, "Provide the macro backdrop for tomorrow.", 15, "macro"),
            return_exceptions=True,
        )
    monitor.ok("parallel_research")

    def _safe(result, label: str) -> str:
        if isinstance(result, Exception):
            return f"[{label} unavailable — {result}]"
        return result

    tech_out       = _safe(results[0], "tech_sector")
    energy_out     = _safe(results[1], "energy_sector")
    healthcare_out = _safe(results[2], "healthcare_sector")
    macro_out      = _safe(results[3], "macro")


    def _sector_failed(output: str) -> bool:
        return output.strip().startswith("[") and "unavailable" in output

    if all(_sector_failed(o) for o in [tech_out, energy_out, healthcare_out]):
        log.warning(
            "All 3 sector agents failed (sustained rate-limit). "
            "Building direct yfinance fallback package for the Picker."
        )
        monitor.start("sector_fallback")
        fallback_tickers = (
            SP500_BY_SECTOR["technology"][:4]
            + SP500_BY_SECTOR["energy"][:3]
            + SP500_BY_SECTOR["healthcare"][:3]
        )
        fallback_lines = [
            "⚠️  Sector agents unavailable. Fallback data built directly from yfinance:",
            ""
        ]
        for tkr in fallback_tickers:
            try:
                hist = _yf_hist(tkr, "2d")
                if hist.empty:
                    continue
                price  = round(float(hist["Close"].iloc[-1]), 2)
                info   = _yf_info(tkr)
                name   = info.get("shortName", tkr)
                target = info.get("targetMeanPrice")
                upside = (
                    f"{round(((float(target) - price) / price) * 100, 1)}%"
                    if target and price > 0 else "N/A"
                )
                fallback_lines.append(
                    f"  {tkr} | {name} | Price: ${price} | "
                    f"Analyst Target: ${target} | Upside: {upside}"
                )
                log.info("Fallback: %s = $%s", tkr, price)
            except Exception as exc:
                log.warning("Fallback data failed for %s: %s", tkr, exc)

        fallback_text  = "\n".join(fallback_lines)
        tech_out       = fallback_text
        energy_out     = "(included in fallback block above)"
        healthcare_out = "(included in fallback block above)"
        monitor.ok("sector_fallback", f"{len(fallback_tickers)} tickers")

    combined = f"""
STOCK RESEARCH PACKAGE — {TODAY}

MACRO BACKDROP:
{macro_out}

TECHNOLOGY SECTOR:
{tech_out}

ENERGY SECTOR:
{energy_out}

HEALTHCARE SECTOR:
{healthcare_out}

INSTRUCTIONS:
Score every stock using the 7-criteria matrix.
Select the top 5 (minimum 3 if fewer available).
HTML report must have real prices on every pick — no placeholders.
"""


    monitor.start("picker_and_judge")
    html_report: Optional[str] = None

    try:
        html_report = await run_with_judge(combined, max_retries=2)
        monitor.ok("picker_and_judge", "Report generated and judge-approved")
    except OutputGuardrailTripwireTriggered:

        monitor.err("picker_and_judge", "Output guardrail fired on all 3 attempts")
        print("\n❌ All Picker attempts produced an empty report.")
        print("💡 This usually means the API is still heavily rate-limited.")
        print("   Wait 2–3 minutes and try again.\n")
        monitor.summary()
        return
    except Exception as e:
        monitor.err("picker_and_judge", str(e))
        log.error("Picker/Judge pipeline failed: %s", e)
        monitor.summary()
        return

    if not html_report or len(html_report.strip()) < 100:
        monitor.err("picker_and_judge", "Report is empty after all retries")
        print("\n❌ Report was empty. Check your API keys and try again.\n")
        monitor.summary()
        return

    monitor.start("open_report")
    try:
        await Runner.run(
            report_opener,
            f"Open this HTML report:\n\n{html_report}",
            max_turns=5,
        )
        monitor.ok("open_report")
    except Exception as e:
        monitor.err("open_report", str(e))
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.getcwd(), f"stock_report_{ts}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_report)
        print(f"\n📄 Browser failed to open — report saved to: {path}\n")

    print("\n✅ Pipeline complete — report opened in browser.")
    monitor.summary()


if __name__ == "__main__":
    asyncio.run(run_stock_research_pipeline())