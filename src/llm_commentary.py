"""Optional Gemini LLM commentary module.

The Gemini API is used **only to explain** the rule-based signals and metrics.
It never generates or overrides Buy / Hold / Sell / Cash signals — the
rule-based strategy remains the single source of truth.

The API key is read from (in order): Streamlit secrets, a local .env file, or
environment variables. It is never hard-coded. If no key is found, the app
still runs and AI commentary is shown as disabled.
"""

from __future__ import annotations

import os

NO_KEY_MESSAGE = "AI commentary is disabled because no Gemini API key was found."
DISCLAIMER = "This is for educational and backtesting purposes only and is not financial advice."

# Default model; override with the GEMINI_MODEL environment variable if needed.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

_PROMPT_SAFETY_RULE = (
    "The rule-based strategy has already generated the signal. Your task is to "
    "explain the signal, performance metrics, benchmark comparison, and "
    "limitations. Do not provide personalised financial advice. Do not claim "
    "the model can predict future returns. Do not suggest real-money trading "
    "actions."
)


def _get_api_key() -> str | None:
    """Find the Gemini API key from Streamlit secrets, .env, or environment."""
    # 1. Streamlit secrets (works when running under Streamlit).
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
            if key:
                return str(key)
    except Exception:  # noqa: BLE001 - secrets may be unavailable outside Streamlit
        pass

    # 2. Load a local .env file if python-dotenv is installed.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001 - dotenv is optional
        pass

    # 3. Environment variable.
    key = os.environ.get("GEMINI_API_KEY")
    return key or None


def is_gemini_available() -> bool:
    """True if a Gemini API key is available (does not validate the key)."""
    return _get_api_key() is not None


def _format_metrics(metrics: dict | None) -> str:
    if not metrics:
        return "  (none provided)"
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if any(tag in key for tag in ("Return", "Volatility", "Drawdown", "Win Rate")):
                lines.append(f"  - {key}: {value:.2%}")
            else:
                lines.append(f"  - {key}: {value:.2f}")
        else:
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def build_signal_commentary_prompt(
    asset: str,
    signal: str,
    signal_reason: str,
    metrics: dict,
    benchmark_metrics: dict | None = None,
    strategy_type: str = "Moving Average Crossover",
) -> str:
    """Build a controlled single-asset prompt using only model outputs/metrics."""
    benchmark_block = ""
    if benchmark_metrics:
        benchmark_block = f"\nBenchmark (buy-and-hold) metrics:\n{_format_metrics(benchmark_metrics)}"

    return f"""{_PROMPT_SAFETY_RULE}

You are writing a concise, research-style commentary (about 150-250 words) for
a backtesting dashboard. Explain the rule-based output below in plain English.

Asset: {asset}
Strategy: {strategy_type}
Current rule-based signal: {signal}
Why the signal was produced: {signal_reason}

Strategy metrics:
{_format_metrics(metrics)}{benchmark_block}

In your commentary:
- Explain what the current signal means for this rule-based model.
- Comment on strategy performance vs the buy-and-hold benchmark.
- Interpret the Sharpe ratio, volatility and maximum drawdown.
- Note 1-2 key limitations of this backtest.
- End with this exact disclaimer on its own line: "{DISCLAIMER}"
"""


def build_portfolio_commentary_prompt(
    assets: list[str],
    signal_table_text: str,
    metrics: dict,
    benchmark_metrics: dict | None = None,
    strategy_type: str = "Momentum Ranking Portfolio",
) -> str:
    """Build a controlled multi-asset prompt using only model outputs/metrics."""
    benchmark_block = ""
    if benchmark_metrics:
        benchmark_block = f"\nEqual-weight benchmark metrics:\n{_format_metrics(benchmark_metrics)}"

    return f"""{_PROMPT_SAFETY_RULE}

You are writing a concise, research-style portfolio commentary (about 150-250
words) for a backtesting dashboard. Explain the rule-based output below.

Selected assets: {", ".join(assets)}
Strategy: {strategy_type}

Current momentum ranking and model weights:
{signal_table_text}

Portfolio metrics:
{_format_metrics(metrics)}{benchmark_block}

In your commentary:
- Explain how the momentum ranking selected the current model weights.
- Comment on portfolio performance vs the equal-weight benchmark.
- Interpret the Sharpe ratio, volatility and maximum drawdown.
- Note 1-2 key limitations of this backtest.
- End with this exact disclaimer on its own line: "{DISCLAIMER}"
"""


def generate_signal_commentary(prompt: str, model_name: str | None = None) -> str:
    """Call the Gemini API and return a short research-style commentary.

    Returns a friendly message instead of raising if the key is missing or the
    SDK/API call fails, so the dashboard never crashes on commentary.
    """
    api_key = _get_api_key()
    if not api_key:
        return NO_KEY_MESSAGE

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or DEFAULT_MODEL)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return "The Gemini API returned an empty response. Please try again."
        # Ensure the disclaimer is always present.
        if DISCLAIMER not in text:
            text = f"{text}\n\n{DISCLAIMER}"
        return text
    except Exception as exc:  # noqa: BLE001 - report errors gracefully in the UI
        return (
            "AI commentary could not be generated due to an error contacting the "
            f"Gemini API: {exc}"
        )
