from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool, ModelSettings
from typing import Dict
import os
import asyncio
import json
import urllib.request
import urllib.parse
import webbrowser
import tempfile
from datetime import datetime

load_dotenv(override=True)

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")


@function_tool
def fetch_market_news(sector: str) -> str:
    query = urllib.parse.quote(f"{sector} stock market investment")
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            articles = data.get("articles", [])
            if not articles:
                return f"No recent news found for {sector}"
            summaries = []
            for a in articles[:5]:
                summaries.append(f"- {a['title']} ({a['source']['name']}): {a.get('description', '')}")
            return f"Latest news for {sector}:\n" + "\n".join(summaries)
    except Exception as e:
        return f"Could not fetch news for {sector}: {str(e)}"


@function_tool
def fetch_stock_overview(ticker: str) -> str:
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            if not data or "Symbol" not in data:
                return f"No overview data found for {ticker}"
            return (
                f"Ticker: {data.get('Symbol')} | Name: {data.get('Name')} | Sector: {data.get('Sector')} | "
                f"Industry: {data.get('Industry')} | Market Cap: {data.get('MarketCapitalization')} | "
                f"PE Ratio: {data.get('PERatio')} | EPS: {data.get('EPS')} | "
                f"52W High: {data.get('52WeekHigh')} | 52W Low: {data.get('52WeekLow')} | "
                f"Analyst Target: {data.get('AnalystTargetPrice')} | "
                f"Description: {data.get('Description', '')[:300]}"
            )
    except Exception as e:
        return f"Could not fetch overview for {ticker}: {str(e)}"


@function_tool
def fetch_stock_quote(ticker: str) -> str:
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            quote = data.get("Global Quote", {})
            if not quote:
                return f"No quote data for {ticker}"
            return (
                f"Ticker: {quote.get('01. symbol')} | Price: {quote.get('05. price')} | "
                f"Open: {quote.get('02. open')} | High: {quote.get('03. high')} | "
                f"Low: {quote.get('04. low')} | Volume: {quote.get('06. volume')} | "
                f"Previous Close: {quote.get('08. previous close')} | "
                f"Change: {quote.get('09. change')} | Change%: {quote.get('10. change percent')}"
            )
    except Exception as e:
        return f"Could not fetch quote for {ticker}: {str(e)}"


@function_tool
def fetch_earnings_calendar(ticker: str) -> str:
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            annual = data.get("annualEarnings", [])
            quarterly = data.get("quarterlyEarnings", [])
            if not quarterly:
                return f"No earnings data for {ticker}"
            latest = quarterly[0]
            return (
                f"Ticker: {ticker} | Latest Reported EPS: {latest.get('reportedEPS')} | "
                f"Estimated EPS: {latest.get('estimatedEPS')} | Surprise: {latest.get('surprise')} | "
                f"Surprise%: {latest.get('surprisePercentage')} | Date: {latest.get('fiscalDateEnding')}"
            )
    except Exception as e:
        return f"Could not fetch earnings for {ticker}: {str(e)}"


@function_tool
def fetch_top_gainers_losers() -> str:
    url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_VANTAGE_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            gainers = data.get("top_gainers", [])[:5]
            most_active = data.get("most_actively_traded", [])[:5]
            result = "Top Gainers Today:\n"
            for g in gainers:
                result += f"  {g.get('ticker')}: {g.get('price')} ({g.get('change_percentage')})\n"
            result += "\nMost Active Today:\n"
            for a in most_active:
                result += f"  {a.get('ticker')}: {a.get('price')} ({a.get('change_percentage')})\n"
            return result
    except Exception as e:
        return f"Could not fetch top gainers/losers: {str(e)}"


@function_tool
def fetch_rsi_signal(ticker: str) -> str:
    url = (
        f"https://www.alphavantage.co/query?function=RSI&symbol={ticker}"
        f"&interval=daily&time_period=14&series_type=close&apikey={ALPHA_VANTAGE_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            rsi_data = data.get("Technical Analysis: RSI", {})
            if not rsi_data:
                return f"No RSI data for {ticker}"
            latest_date = sorted(rsi_data.keys(), reverse=True)[0]
            rsi_value = rsi_data[latest_date]["RSI"]
            signal = "OVERSOLD (potential buy)" if float(rsi_value) < 30 else ("OVERBOUGHT (caution)" if float(rsi_value) > 70 else "NEUTRAL")
            return f"Ticker: {ticker} | RSI ({latest_date}): {rsi_value} | Signal: {signal}"
    except Exception as e:
        return f"Could not fetch RSI for {ticker}: {str(e)}"


@function_tool
def open_report_in_browser(html_body: str) -> Dict[str, str]:
    filename = f"stock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_body)
    webbrowser.open(f"file:///{filepath.replace(os.sep, '/')}")
    return {"status": "opened", "path": filepath}


tech_sector_analyst = Agent(
    name="TechSectorAnalyst",
    instructions=(
        "You are a specialist equity analyst focused on Technology stocks (semiconductors, software, AI, cloud). "
        "Use fetch_market_news to get the latest news for 'technology AI semiconductor cloud stocks'. "
        "Use fetch_top_gainers_losers to identify momentum. "
        "Then use fetch_stock_quote and fetch_stock_overview on 3-5 promising tech tickers you identify. "
        "Use fetch_rsi_signal to check technicals. Use fetch_earnings_calendar for catalyst checks. "
        "Return a structured analysis: for each stock give ticker, current price, key thesis, risk, and a conviction score 1-10."
    ),
    tools=[fetch_market_news, fetch_stock_quote, fetch_stock_overview, fetch_rsi_signal, fetch_earnings_calendar, fetch_top_gainers_losers],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0)
)

energy_sector_analyst = Agent(
    name="EnergySectorAnalyst",
    instructions=(
        "You are a specialist equity analyst focused on Energy stocks (oil, gas, renewables, utilities). "
        "Use fetch_market_news to get the latest news for 'energy oil gas renewable stocks'. "
        "Use fetch_top_gainers_losers to identify momentum names. "
        "Then use fetch_stock_quote and fetch_stock_overview on 3-5 promising energy tickers you identify. "
        "Use fetch_rsi_signal for technical signals. Use fetch_earnings_calendar for upcoming catalysts. "
        "Return a structured analysis: for each stock give ticker, current price, key thesis, risk, and a conviction score 1-10."
    ),
    tools=[fetch_market_news, fetch_stock_quote, fetch_stock_overview, fetch_rsi_signal, fetch_earnings_calendar, fetch_top_gainers_losers],
    model="gpt-4o-mini",
model_settings=ModelSettings(temperature=0)
)

healthcare_sector_analyst = Agent(
    name="HealthcareSectorAnalyst",
    instructions=(
        "You are a specialist equity analyst focused on Healthcare stocks (biotech, pharma, medical devices, health services). "
        "Use fetch_market_news to get the latest news for 'healthcare biotech pharma FDA approval stocks'. "
        "Use fetch_top_gainers_losers to identify momentum. "
        "Then use fetch_stock_quote and fetch_stock_overview on 3-5 promising healthcare tickers you identify. "
        "Use fetch_rsi_signal for technical entry signals. Use fetch_earnings_calendar for catalyst events. "
        "Return a structured analysis: for each stock give ticker, current price, key thesis, risk, and a conviction score 1-10."
    ),
    tools=[fetch_market_news, fetch_stock_quote, fetch_stock_overview, fetch_rsi_signal, fetch_earnings_calendar, fetch_top_gainers_losers],
    model="gpt-4o-mini",
model_settings=ModelSettings(temperature=0)
)

macro_news_analyst = Agent(
    name="MacroNewsAnalyst",
    instructions=(
        "You are a macro market analyst. "
        "Use fetch_market_news with topics: 'Federal Reserve interest rates inflation economy', 'S&P 500 market outlook tomorrow', and 'earnings season Wall Street'. "
        "Use fetch_top_gainers_losers to understand today's market tone. "
        "Synthesize a macro backdrop summary: what is the market sentiment heading into tomorrow, "
        "what macro tailwinds or headwinds exist, which sectors look most favored, and what risks to watch. "
        "Be specific and actionable."
    ),
    tools=[fetch_market_news, fetch_top_gainers_losers],
    model="gpt-4o-mini",
model_settings=ModelSettings(temperature=0)
)

top5_stock_picker = Agent(
    name="Top5StockPicker",
    instructions=(
        "You are a senior portfolio manager and the final decision-maker. "
        "You receive research from three sector analysts (Tech, Energy, Healthcare) and a macro backdrop summary. "
        "Your job: select exactly the TOP 5 stocks to buy tomorrow based on the combined evidence. "
        "Weigh: news catalysts, fundamentals (PE, EPS, analyst target), momentum (price change, volume), "
        "technical signal (RSI), earnings surprises, and macro fit. "
        "For each of the 5 picks provide: "
        "1. Ticker and company name "
        "2. Current price — use the EXACT price from the research data provided. If no price was found, write 'Price unavailable' "
        "3. Why this stock NOW - specific data-backed reasoning "
        "4. Key risk to watch "
        "5. Suggested entry strategy (e.g. buy at open, wait for dip to X) "
        "6. Conviction level: HIGH / MEDIUM "
        f"CRITICAL RULES — you must follow these exactly: "
        f"- The report date is: {datetime.now().strftime('%B %d, %Y')}. Use this exact date string. NEVER write '[Insert today's date]' or any placeholder. "
        "- NEVER write '[Insert current price]' or any bracketed placeholder anywhere in the report. "
        "- Use only real values from the research data given to you. "
        "Format the output as clean HTML suitable for a browser report with a professional layout. "
        "Include a header with the exact date above, a brief macro context paragraph, then the 5 picks as cards. "
        "End with a disclaimer."
    ),
    model="gpt-4o-mini",
model_settings=ModelSettings(temperature=0)
)

report_opener = Agent(
    name="ReportOpener",
    instructions=(
        "You receive a fully formatted HTML investment report. "
        "Use open_report_in_browser to save and open it immediately. "
        "Call the tool exactly once."
    ),
    tools=[open_report_in_browser],
    model="gpt-4o-mini",
    handoff_description="Save the HTML report and open it in the browser",
model_settings=ModelSettings(temperature=0)
)


async def run_stock_research_pipeline():
    research_prompt = (
        "Analyze your sector thoroughly using all available tools. "
        "Identify the 3-5 strongest stock opportunities for tomorrow. "
        "Back every claim with data from the tools."
    )

    with trace("Parallel Sector Research"):
        sector_results = await asyncio.gather(
            Runner.run(tech_sector_analyst, research_prompt, max_turns=30),
            Runner.run(energy_sector_analyst, research_prompt, max_turns=30),
            Runner.run(healthcare_sector_analyst, research_prompt, max_turns=30),
            Runner.run(macro_news_analyst, "Provide a macro market backdrop for tomorrow's trading session.", max_turns=30),
        )

    tech_analysis = sector_results[0].final_output
    energy_analysis = sector_results[1].final_output
    healthcare_analysis = sector_results[2].final_output
    macro_summary = sector_results[3].final_output

    combined_research = f"""
MACRO BACKDROP:
{macro_summary}

TECHNOLOGY SECTOR ANALYSIS:
{tech_analysis}

ENERGY SECTOR ANALYSIS:
{energy_analysis}

HEALTHCARE SECTOR ANALYSIS:
{healthcare_analysis}

Based on all the above research, select the TOP 5 stocks to invest in tomorrow and format the full HTML report.
"""

    with trace("Top 5 Selection and Browser Open"):
        top5_result = await Runner.run(top5_stock_picker, combined_research, max_turns=20)
        html_report = top5_result.final_output
        await Runner.run(
            report_opener,
            f"Here is the final investment report HTML to open:\n\n{html_report}"
        )

    print("✅ Stock research pipeline complete. Report opened in browser.")
    print("\n--- FINAL REPORT PREVIEW ---\n")
    print(html_report)


asyncio.run(run_stock_research_pipeline())
