import streamlit as st
import tempfile
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import json
import duckdb
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
import sys
import io
import os
import plotly.express as px
from PIL import Image
import base64
from rich.prompt import Prompt
from langchain_community.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.schema import HumanMessage
import requests
import time
import random
import ast
import re
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from reportlab.lib.pagesizes import letter
import ssl, certifi
ssl._create_default_https_context = ssl._create_unverified_context
from reportlab.pdfgen import canvas
import textwrap
from fpdf import FPDF
from PIL import Image
#import pytesseract  # for OCR on images
import pdfplumber
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pdfplumber
import yfinance as yf
import yahooquery as yq
import subprocess
from scipy.stats import beta, triang, lognorm, norm, truncnorm, uniform
from dcf_engine import DCFEngine
from visualization import *
from styles import *

st.set_page_config(page_title="Valoración de Acciones", layout="wide")

st.title("Valoración de Acciones")
st.subheader("Flujo de Caja Descontado")

def growth_stats(vals: list) -> dict:
    """Calculate growth statistics"""
    if len(vals) < 2: return {'mean': 0.05, 'std': 0.03}
    gr = np.diff(vals) / vals[:-1]
    gr = np.clip(gr, -0.5, 1.0)
    return {'mean': float(gr.mean()), 'std': float(gr.std()) if len(gr) > 1 else 0.03}


def _transpose_and_parse(df):
    # yfinance returns DataFrame with columns = periods; transpose -> rows = periods
    if df is None or df.empty:
        return pd.DataFrame()
    df_t = df.T.copy()
    # ensure index is datetime if possible
    try:
        df_t.index = pd.to_datetime(df_t.index)
    except Exception:
        pass
    return df_t

def compute_ttm_from_quarters(q_df, metric_cols=None, annualize_partial=True):
    """
    q_df: DataFrame of quarterly data (rows = quarter end dates, columns = metrics)
    metric_cols: iterable of column names to compute TTM for (None => all columns)
    annualize_partial: if <4 quarters available, scale the sum by 4/n_quarters
    Returns: DataFrame with index = "TTM as of <last quarter date>" and columns = metrics
    """
    if q_df is None or q_df.empty:
        return pd.DataFrame()
    q_df_sorted = q_df.sort_index()
    # choose last up to 4 quarters
    available = q_df_sorted.shape[0]
    n = min(4, available)
    last_quarters = q_df_sorted.iloc[-n:]
    if metric_cols is None:
        metric_cols = last_quarters.columns
    ttm = last_quarters[metric_cols].sum(axis=0)
    annualized_partial = False
    if n < 4 and annualize_partial and n > 0:
        ttm = ttm * (4.0 / n)
        annualized_partial = True
    # produce DataFrame row label "TTM as of <last_quarter_date>"
    last_date = last_quarters.index[-1]
    idx = pd.to_datetime(last_date)
    result = pd.DataFrame([ttm.values], index=[idx], columns=metric_cols)
    result.attrs['annualized_partial'] = annualized_partial
    return result

def get_financials_with_annualized_ttm(ticker_symbol: str,
                                      statements=('income', 'cashflow', 'balance'),
                                      annualize_partial=True):
    """
    Returns a dict of DataFrames keyed by statement name.
    Each DataFrame has:
      - rows = annual periods (and possibly a TTM row for the latest),
      - columns = financial items,
      - columns/flags: we add 'is_trailing' (row boolean index column-like), and
                      'source' (annual / ttm_quarters / ttm_annualized_partial)
    Notes:
      - Balance sheet is point-in-time: no TTM sum; if annual missing, we use most recent quarter and flag it.
    """
    ticker_obj = yf.Ticker(ticker_symbol)
    number_shares = ticker_obj.info['sharesOutstanding']         
    price = ticker_obj.info['regularMarketPreviousClose']
    out = {}
    # fetch all relevant raw dfs from yfinance
    raw_income_annual = _transpose_and_parse(ticker_obj.financials)       # annual income
    raw_income_quarterly = _transpose_and_parse(ticker_obj.quarterly_financials)
    raw_cash_annual = _transpose_and_parse(ticker_obj.cashflow)          # annual cashflow
    raw_cash_quarterly = _transpose_and_parse(ticker_obj.quarterly_cashflow)
    raw_bs_annual = _transpose_and_parse(ticker_obj.balance_sheet)       # annual balance
    raw_bs_quarterly = _transpose_and_parse(ticker_obj.quarterly_balance_sheet)

    # --- INCOME ---
    if 'income' in statements:
        income_annual = raw_income_annual.copy()
        income_q = raw_income_quarterly.copy()
        df_income = income_annual.copy()
        # add a TTM row computed from quarters
        ttm_income = compute_ttm_from_quarters(income_q, metric_cols=income_q.columns, annualize_partial=annualize_partial)
        if not ttm_income.empty:
            # label row index so later merging is easy; keep both annual rows and TTM row
            # avoid duplicate index: if same date as an annual row, it might overwrite; we'll append and then sort/unique
            ttm_income.rename(index={ttm_income.index[0]: pd.to_datetime(ttm_income.index[0])}, inplace=True)
            # mark source via a MultiIndex column or with attributes - we will append a column later
            # Append TTM row
            combined = pd.concat([df_income, ttm_income], axis=0, sort=False)
        else:
            combined = df_income
        # sort index descending (latest first)
        combined = combined.sort_index(ascending=False)
        # add flags: is_trailing = index == most recent index
        if not combined.empty:
            most_recent_idx = combined.index[0]
            combined = combined.copy()
            combined['is_trailing'] = [idx == most_recent_idx for idx in combined.index]
            # set source column
            def detect_source(idx):
                # if idx equals any annual row originally in income_annual -> 'annual'
                if idx in income_annual.index:
                    return 'annual'
                elif idx in ttm_income.index:
                    return 'ttm_quarters_annualized' if getattr(ttm_income, 'attrs', {}).get('annualized_partial', False) else 'ttm_quarters'
                else:
                    return 'derived'
            combined['source'] = [detect_source(idx) for idx in combined.index]
        out['income'] = combined[[c for c in combined.columns if c not in ['is_trailing', 'source']]]

    # --- CASHFLOW ---
    if 'cashflow' in statements:
        cash_annual = raw_cash_annual.copy()
        cash_q = raw_cash_quarterly.copy()
        df_cash = cash_annual.copy()
        ttm_cash = compute_ttm_from_quarters(cash_q, metric_cols=cash_q.columns, annualize_partial=annualize_partial)
        if not ttm_cash.empty:
            combined = pd.concat([df_cash, ttm_cash], axis=0, sort=False)
        else:
            combined = df_cash
        combined = combined.sort_index(ascending=False)
        if not combined.empty:
            most_recent_idx = combined.index[0]
            combined = combined.copy()
            combined['is_trailing'] = [idx == most_recent_idx for idx in combined.index]
            def detect_source(idx):
                if idx in cash_annual.index:
                    return 'annual'
                elif idx in ttm_cash.index:
                    return 'ttm_quarters_annualized' if getattr(ttm_cash, 'attrs', {}).get('annualized_partial', False) else 'ttm_quarters'
                else:
                    return 'derived'
            combined['source'] = [detect_source(idx) for idx in combined.index]
        out['cashflow'] = combined[[c for c in combined.columns if c not in ['is_trailing', 'source']]]

    # --- BALANCE SHEET ---
    if 'balance' in statements or 'balance_sheet' in statements:
        # naming flexibility
        bs_annual = raw_bs_annual.copy()
        bs_q = raw_bs_quarterly.copy()
        if not bs_annual.empty:
            combined = bs_annual.copy()
            source_map = {idx: 'annual' for idx in combined.index}
        else:
            # if no annual, use the most recent quarter (point-in-time)
            if not bs_q.empty:
                latest_q = bs_q.sort_index(ascending=False).iloc[[0]]
                combined = latest_q.copy()
                source_map = {combined.index[0]: 'quarter (used as proxy)'}
            else:
                combined = pd.DataFrame()
                source_map = {}
        if not combined.empty:
            combined = combined.sort_index(ascending=False)
            most_recent_idx = combined.index[0]
            combined = combined.copy()
            combined['is_trailing'] = [idx == most_recent_idx for idx in combined.index]
            combined['source'] = [source_map.get(idx, 'annual') for idx in combined.index]
            #mark that balance sheet rows are not annualized
            combined['annualized_partial'] = False
        out['balance'] = combined[[c for c in combined.columns if c not in ['is_trailing', 'source', 'annualized_partial']]]

    return out, number_shares, price

def fetch_data(ticker_symbol):
  ticker = yf.Ticker(ticker_symbol)
  info = ticker.info
  # --- Valuation Inputs ---
  shares_outstanding = info.get('sharesOutstanding', 1)
  # Use market cap for Equity Value
  equity_value = info.get('marketCap') 
  # Use Total Debt or Long Term Debt as a proxy for Debt Value
  debt_value = info.get('totalDebt', info.get('longTermDebt', 0))  
  cash_non_operating_asset = info.get('totalCash', info.get('cash', 0)) 

  # --- Revenue and Margin Inputs ---
  # TTM Revenue as Revenue Base
  revenue_base = info.get('trailingAnnualRevenue', 0) / 1_000_000_000 # In Billions
  
  # Historical Financials
  financials = ticker.financials.T

  # Operating Margin (EBIT / Revenue)
  ebit = financials.get('EBIT', pd.Series()).iloc[0] if not financials.empty else 0
  current_operating_margin = (ebit / financials.get('Total Revenue', pd.Series()).iloc[0]) if not financials.empty and financials.get('Total Revenue', pd.Series()).iloc[0] else 0

  # Effective Tax Rate
  income_before_tax = financials.get('Pretax Income', pd.Series()).iloc[0] if not financials.empty else 0
  income_tax_expense = financials.get('Tax Provision', pd.Series()).iloc[0] if not financials.empty else 0
  current_effective_tax_rate = (income_tax_expense / income_before_tax) if income_before_tax else marginal_tax_rate_proxy

  # --- WACC Inputs ---
  # Levered Beta
  levered_beta = info.get('beta', 1.0)
  
  return {
        "sharesOutstanding": shares_outstanding,
        "marketCap": equity_value,
        "total_debt": debt_value,
        "ebit": ebit, 
        "beta": levered_beta
    }

# Sidebar parameters
st.sidebar.header("Parámetros")
ticker_symbol = st.sidebar.text_input('Ticker', value='NVDA')
tax_rate = st.sidebar.number_input('Tax Rate (%)', value=21, step=1)/100
wacc = st.sidebar.number_input("WACC (%)", value=19.28, step=0.1) / 100
tgr = st.sidebar.number_input("Crecimiento de la Perpetuidad (%)", value=5.0, step=0.5) / 100

st.sidebar.write("---")
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Market Parameters")
    rf = st.number_input("Risk-Free Rate", 0.0, 0.10, 0.045, 0.005, format="%.3f")
    mp = st.number_input("Market Risk Premium", 0.03, 0.12, st.session_state.get('mp_calc', 0.065), 0.005, format="%.3f")
    
    st.subheader("Simulation Settings")
    n_sims = st.select_slider("Monte Carlo Simulations", [1000, 5000, 10000, 25000, 50000], 10000)
    n_years = st.slider("Projection Years", 3, 10, 5)


STD_CAP = 0.10

def fit_lognormal(series):
    """Fit a lognormal distribution with sigma capped at 10%."""
    log_vals = np.log1p(series)                     # log(1+g)
    mu = log_vals.mean()
    sigma = min(log_vals.std(), STD_CAP)            # cap std to 10%

    def sampler(n):
        return np.expm1(np.random.normal(mu, sigma, n))

    return sampler, {"dist":"lognormal", "mu":mu, "sigma":sigma}


def fit_beta(series):
    """Stable Beta fit with a cap on variance (std <= 10%)."""
    s = np.clip(series, 1e-3, 1 - 1e-3)

    mean = s.mean()
    var = min(s.var(), STD_CAP**2)                  # cap variance to (0.10)^2

    k = mean*(1-mean)/var - 1
    a = max(mean*k, 1e-3)
    b = max((1-mean)*k, 1e-3)

    def sampler(n):
        return np.random.beta(a, b, n)

    return sampler, {"dist":"beta", "a":a, "b":b}

def fit_normal(series, low=None, high=None):
    """Normal distribution with std capped at 10%."""
    mu = series.mean()
    sigma = min(series.std(), STD_CAP)

    def sampler(n):
        vals = np.random.normal(mu, sigma, n)
        if low is not None: 
            vals = np.maximum(vals, low)
        if high is not None:
            vals = np.minimum(vals, high)
        return vals

    return sampler, {"dist":"normal", "mu":mu, "sigma":sigma}

def fit_triangular(series):
    """Triangular distribution based on 10/50/90 percentiles."""
    s = series.dropna()

    low = s.quantile(0.10)
    mode = s.quantile(0.50)
    high = s.quantile(0.90)

    # Prevent degenerate scale
    if high <= low:
        high = low + 1e-6

    c = (mode - low) / (high - low)

    def sampler(n):
        return triang.rvs(
            c=c,
            loc=low,
            scale=(high - low),
            size=n
        )

    return sampler, {"dist":"triangular", "low":low, "mode":mode, "high":high}


# -----------------------------------------------------------
# 2. Automatic picker
# -----------------------------------------------------------

def capped_std(series, cap=0.10):
    """Ensure σ ≤ cap * mean."""
    mu = series.mean()
    sigma = min(series.std(), abs(mu) * cap)
    return mu, sigma

def fit_truncated_normal(series, max_std=0.10):
    """Fit a truncated normal distribution with clipped std."""
    
    mu = series.mean()
    sigma = min(series.std(), max_std * abs(mu))

    # enforce a positive standard deviation
    if sigma <= 0 or np.isnan(sigma):
        sigma = max_std * abs(mu)
        if sigma == 0:
            # fallback: return constant sampler
            def sampler(n):
                return np.full(n, mu)
            return sampler, {"dist": "constant", "value": mu}

    # compute truncated bounds (± 3 std by default)
    lower, upper = series.min(), series.max()
    a = ((lower - mu) / sigma)/2
    b = ((upper - mu) / sigma)/2

    def sampler(n):
        return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)

    return sampler, {
        "dist": "truncnorm",
        "mu": mu,
        "sigma": sigma,
        "lower": lower,
        "upper": upper
    }

def fit_uniform_margin(series, low=None, high=None):
    """
    EBITDA margin: stable and usually range-bound.
    If no low/high provided, use historical min/max.
    """

    low = series.min() if low is None else low
    high = series.max() if high is None else high

    def sampler(n):
        return uniform.rvs(loc=low, scale=high - low, size=n)

    return sampler, {"dist": "uniform", "low": low, "high": high}
    
def infer_distribution(series):
    """Automatically select distribution based on data properties."""
    series = series.dropna()
    if len(series) < 3:
        return fit_triangular(series)

    skew = series.skew()

    # bounded (0,1)
    if series.min() > 0 and series.max() < 1:
        return fit_beta(series)

    # positive + right skew → lognormal
    if (series > 0).all() and skew > 0.5:
        return fit_lognormal(series)

    # fallback
    return fit_normal(series)

def extract_dcf_variables(income, bs):
    out = {}
    # --------------------
    # Revenue growth
    # --------------------
    revenues = income.loc["Total Revenue"].sort_index()
    rev_growth = revenues.pct_change().dropna()
    sampler, info = fit_truncated_normal(rev_growth)
    out["rev_growth"] = sampler
    out["rev_growth_info"] = info

    # EBIT margin
    ebit = income.loc["Operating Income"].sort_index()
    ebit_margin = (ebit / revenues).dropna()
    sampler, info = fit_uniform_margin(ebit_margin)
    out["ebit_margin"] = sampler
    out["ebit_margin_info"] = info

    # Depreciation / revenue
    if "Depreciation" in income.index:
        dep_ratio = (income.loc["Depreciation"] / revenues).dropna()
    else:
        dep_ratio = (income.loc["Reconciled Depreciation"] / revenues).dropna()

    sampler, info = infer_distribution(dep_ratio)
    out["dep_ratio"] = sampler
    out["dep_ratio_info"] = info

    # CAPEX
    capex = bs.loc["Net PPE"].sort_index().sort_index().diff()
    capex_pct = (capex / revenues).dropna()
    sampler, info = fit_truncated_normal(capex_pct)
    out["capex"] = sampler
    out["capex_info"] = info
    
    # NWC
    nwc = bs.loc["Working Capital"].sort_index().sort_index().diff()
    nwc_pct = (nwc / revenues).dropna()
    sampler, info = fit_truncated_normal(nwc_pct)
    out["nwc"] = sampler
    out["nwc_info"] = info
    
    # Sales-to-capital
    assets = bs.loc["Total Assets"].sort_index()
    sales_to_cap = (revenues / assets).dropna()
    sampler, info = fit_truncated_normal(sales_to_cap)
    out["sales_to_cap"] = sampler
    out["sales_to_cap_info"] = info

    # Tax rate
    tax_exp = income.loc["Tax Provision"].sort_index()
    pretax = income.loc["Pretax Income"].sort_index()
    tax_rate = (tax_exp / pretax).clip(0,1).dropna()
    sampler, info = fit_truncated_normal(tax_rate)
    out["tax_rate"] = sampler
    out["tax_rate_info"] = info

    return out

res, number_shares, price = get_financials_with_annualized_ttm(ticker_symbol, statements=('income','cashflow','balance'), annualize_partial=True)
balance, income, cashflow = res['balance'].T, res['income'].T , res['cashflow'].T 
debt_long = balance.loc['Long Term Debt And Capital Lease Obligation'].iloc[0]
equity_total = balance.loc['Total Equity Gross Minority Interest'].iloc[0]
sharesOutstanding = number_shares
cash = balance.loc['Cash And Cash Equivalents'].iloc[0]

#rev_growth_mean, rev_growth_std = np.log(1+ticker.income_stmt.loc['Total Revenue'].sort_index().pct_change(fill_method=None)).mean(), np.log(1+ticker.income_stmt.loc['Total Revenue'].sort_index().pct_change(fill_method=None)).std()

st.write(debt_long/equity_total)

def monte_carlo_dcf(distros, 
                    current_revenue,
                    cash,
                    debt,
                    shares_outstanding,
                    wacc=0.09,
                    tgr=0.02,
                    horizon=10,
                    n_sims=20000):

    # unpack distributions
    rev_growth_sampler   = distros["rev_growth"]
    ebit_margin_sampler  = distros["ebit_margin"]
    dep_ratio_sampler    = distros["dep_ratio"]
    capex_ratio_sampler = distros["capex"]
    nwc_ratio_sampler = distros["nwc"]
    sales_to_cap_sampler = distros["sales_to_cap"]
    tax_rate_sampler     = distros["tax_rate"]

    # generate simulation vectors
    rev_growth   = rev_growth_sampler(n_sims)          # growth rate
    ebit_margin  = ebit_margin_sampler(n_sims)         # EBIT margin
    dep_ratio    = dep_ratio_sampler(n_sims)           # dep / revenue
    capex_ratio  = capex_ratio_sampler(n_sims)
    nwc_ratio    = nwc_ratio_sampler(n_sims)
    sales_to_cap = sales_to_cap_sampler(n_sims)        # revenue / assets
    tax_rate     = tax_rate_sampler(n_sims)            # tax

    # initialize arrays
    revenues = np.zeros((horizon + 1, n_sims))
    ebit     = np.zeros((horizon + 1, n_sims))
    fcff     = np.zeros((horizon, n_sims))

    # year 0 revenue
    revenues[0] = current_revenue
    st.write(rev_growth)
    # MAIN PROJECTIONS
    for t in range(1, horizon + 1):
        # revenue projection
        revenues[t] = revenues[t-1] * (1 + rev_growth)

        # EBIT
        ebit[t] = revenues[t] * ebit_margin

        # NOPAT
        nopat = ebit[t] * (1 - tax_rate)

        # depreciation
        depreciation = revenues[t] * dep_ratio

        # reinvestment using sales-to-cap
        reinvestment = (revenues[t] - revenues[t-1]) / sales_to_cap

        # reinvestment using sales-to-cap
        capex_fc = revenues[t] * capex_ratio

        # reinvestment using sales-to-cap
        nwc_fc = revenues[t] * nwc_ratio

        # FCFF
        fcff[t-1] = nopat + depreciation - capex_fc - nwc_fc

    # Terminal value
    terminal_fcff = fcff[-1] * (1 + tgr)
    terminal_value = terminal_fcff / (wacc - tgr)
    st.write(fcff)
    # Discounted FCFF
    discount_factors = ((1 + wacc) ** np.arange(1, horizon + 1)).reshape(-1, 1)
    pv_fcff = (fcff / discount_factors).sum(axis=0)
    pv_terminal = terminal_value / ((1 + wacc) ** horizon)

    # Enterprise value
    ev = pv_fcff + pv_terminal
    print(ev)
    # Equity value
    equity_value = ev + cash - debt
    equity_per_share = equity_value / shares_outstanding

    return {
        "ev": ev,
        "equity_value": equity_value,
        "equity_per_share": equity_per_share,
        "fcff": fcff,
        "revenues": revenues
    }


current_revenue = income.loc['Total Revenue'].iloc[0]     
debt = debt_long
shares_outstanding  = sharesOutstanding

distros = extract_dcf_variables(income, balance)

results = monte_carlo_dcf(
    distros=distros,                # <-- from previous code
    current_revenue=current_revenue,
    cash=cash,
    debt=debt,
    shares_outstanding=shares_outstanding,
    wacc=wacc,
    tgr=tgr,
    horizon=10,
    n_sims=20000
)

vals = results["equity_per_share"]  # from your MC simulation
mean_val = vals.mean()

def main():
    # Header
    st.markdown('<h1 class="main-header">Stocks Valuation Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Monte Carlo for intrinsic value estimation</p>', unsafe_allow_html=True)
    
    # Sidebar
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Company Info
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Company Information</div>', unsafe_allow_html=True)
        
        shares = sharesOutstanding
        
        if st.button("Fetch Beta", use_container_width=True):
            with st.spinner("Fetching..."):
                beta, err, mkt_info = fetch_stock_beta(ticker)
                if beta:
                    st.session_state.fetched_beta = 1.68
                    st.session_state.mp_calc = 0.065
                    st.success(f"Beta: {beta:.3f}")
                else:
                    st.error(err or "Unable to fetch beta")
        
        #beta = st.metric("Beta Coefficient", beta)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Financial Structure
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Financial Structure</div>', unsafe_allow_html=True)
        
        net_debt = debt_long - cash
        d2e = debt / (total_equity) 
        
        cod = st.number_input("Cost of Debt", 0.0, 0.15, 0.04, 0.005, format="%.3f")
        tax = st.number_input("Effective Tax Rate", 0.0, 0.50, 0.25, 0.01, format="%.3f")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Historical Financials
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Historical Financials</div>', unsafe_allow_html=True)
        
        years = st.slider("Years of historical data", 3, 5, 5)
        curr_yr = datetime.now().year
        
        # Revenue & FCF
        revs, fcfs = [], []
        
        st.markdown("**Revenue (millions)**")
        rev_cols = st.columns(years)
        for i, col in enumerate(rev_cols):
            yr = curr_yr - i
            revs.append(col.number_input(f"{yr}", 0.0, value=1000.0*(1.05**(years-i-1)), format="%.1f", key=f"r{yr}"))
        
        st.markdown("**Free Cash Flow (millions)**")
        fcf_cols = st.columns(years)
        for i, col in enumerate(fcf_cols):
            yr = curr_yr - i
            fcfs.append(col.number_input(f"{yr}", -1000.0, value=100.0*(1.05**(years-i-1)), format="%.1f", key=f"f{yr}"))
        
        revs.reverse()
        fcfs.reverse()
        
        # Metrics
        rev_g = growth_stats(revs)
        fcf_g = growth_stats(fcfs)
        fcf_margin = np.mean([f/r for f,r in zip(fcfs, revs) if r > 0])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display metrics
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Calculated Metrics</div>', unsafe_allow_html=True)
        
        # Quick WACC calc
        ce = rf + beta * mp
        we = 1 / (1 + d2e)
        wacc = we * ce + (1 - we) * cod * (1 - tax)
        
        cols = st.columns(4)
        cols[0].markdown(metric_card("Avg Revenue Growth", fmt_pct(rev_g['mean']), ('neutral', f"σ: {fmt_pct(rev_g['std'])}")), unsafe_allow_html=True)
        cols[1].markdown(metric_card("Avg FCF Growth", fmt_pct(fcf_g['mean']), ('neutral', f"σ: {fmt_pct(fcf_g['std'])}")), unsafe_allow_html=True)
        cols[2].markdown(metric_card("Avg FCF Margin", fmt_pct(fcf_margin)), unsafe_allow_html=True)
        cols[3].markdown(metric_card("WACC", fmt_pct(wacc)), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Growth assumptions
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Growth Assumptions</div>', unsafe_allow_html=True)
        
        use_range = st.checkbox("Use growth rate ranges", True)
        
        proj_growth = []
        base_g = rev_g['mean']
        
        if use_range:
            st.markdown("**Projected Growth Rate Ranges**")
            for i in range(n_years):
                c1, c2, c3 = st.columns([2, 2, 2])
                center = max(base_g * (0.9**i), 0.02)
                with c1: st.markdown(f"**Year {i+1}**")
                with c2: min_g = st.number_input("Min", -0.5, 1.0, max(center-0.05, -0.1), 0.01, format="%.3f", key=f"ming{i}")
                with c3: max_g = st.number_input("Max", -0.5, 1.0, min(center+0.05, 0.3), 0.01, format="%.3f", key=f"maxg{i}")
                proj_growth.append((min(min_g, max_g), max(min_g, max_g)))
        else:
            st.markdown("**Projected Growth Rates**")
            g_cols = st.columns(n_years)
            for i, col in enumerate(g_cols):
                g = col.number_input(f"Year {i+1}", -0.5, 1.0, max(base_g*(0.9**i), 0.02), 0.01, format="%.3f", key=f"g{i}")
                proj_growth.append(g)
        
        # Terminal value
        st.markdown("**Terminal Value**")
        if use_range:
            c1, c2 = st.columns(2)
            with c1: tg_min = st.number_input("Min Terminal", 0.0, 0.04, 0.02, 0.005, format="%.3f")
            with c2: tg_max = st.number_input("Max Terminal", 0.0, 0.05, 0.03, 0.005, format="%.3f")
            term_growth = (tg_min, tg_max)
        else:
            term_growth = st.number_input("Terminal Growth", 0.0, 0.05, 0.025, 0.005, format="%.3f")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Run valuation
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Run DCF Valuation", type="primary", use_container_width=True):
        if price <= 0 or shares <= 0:
            st.error("Price and shares must be positive")
            return
        
        params = {
            'base_fcf': fcfs[-1] * 1e6,
            'growth_rates': proj_growth,
            'terminal_growth': term_growth,
            'beta': beta,
            'debt_to_equity': d2e,
            'cost_of_debt': cod,
            'tax_rate': tax,
            'net_debt': net_debt * 1e6,
            'shares_outstanding': shares * 1e6
        }
        
        unc = {'fcf_growth_std': max(fcf_g['std'], 0.03), 'terminal_growth_std': 0.005, 'beta_std': 0.1}
        
        with st.spinner(f"Running {n_sims:,} simulations..."):
            dcf = DCFEngine(rf, mp)
            results = dcf.monte_carlo(params, n_sims, unc)
            
            st.session_state.valuation_results = {
                'results': results, 'current_price': price, 'currency': currency,
                'company_name': company, 'parameters': params
            }
    
    # Display results
    if st.session_state.valuation_results:
        data = st.session_state.valuation_results
        res = data['results']
        price = data['current_price']
        curr = data['currency']
        company = data['company_name']
        
        st.markdown(f"## Valuation Results for {company}")
        
        # Summary
        display_summary(res, price, curr)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Distribution", "Scenarios", "Sensitivity"])
        
        with tab1:
            st.plotly_chart(plot_distribution(res['per_share_values'], price, curr), use_container_width=True)
            st.plotly_chart(plot_percentiles(res, price, curr), use_container_width=True)
        
        with tab2:
            scen = {'bear': res['percentiles']['p10'], 'base': res['percentiles']['p50'],
                   'bull': res['percentiles']['p90'], 'expected': res['mean']}
            
            st.plotly_chart(plot_waterfall(price, scen, curr), use_container_width=True)
            
            # Scenario table
            st.markdown("#### Scenario Summary")
            scen_df = pd.DataFrame([
                {'Scenario': name, 'Fair Value': fmt_curr(scen[key], curr),
                 'Upside/Downside': fmt_pct(scen[key]/price - 1)}
                for name, key in [('Bear (P10)', 'bear'), ('Base (P50)', 'base'),
                                 ('Bull (P90)', 'bull'), ('Expected (Mean)', 'expected')]
            ])
            st.dataframe(scen_df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.plotly_chart(plot_sensitivity(res, data['parameters']), use_container_width=True)
            
            # Insight
            up_prob = (res['per_share_values'] > price).mean() * 100
            insight = ("Strong buy" if up_prob > 70 else "Moderate buy" if up_prob > 50 else 
                      "Fairly valued" if up_prob > 30 else "Potentially overvalued")
            
            st.markdown(summary_box("Key Insight",
                f"<strong>{up_prob:.0f}%</strong> probability of upside. {insight}."), 
                unsafe_allow_html=True)
        
        # Export
        st.markdown("### Export Results")
        
        export = {
            'summary': {'company': company, 'current_price': price, 
                       'mean_fair_value': res['mean'], 'upside_probability': f"{up_prob:.1f}%"},
            'percentiles': res['percentiles'],
            'statistics': {'std': res['std'], 'skew': res['skew'], 'kurtosis': res['kurtosis']}
        }
        
        st.download_button("Download Results (JSON)", json.dumps(export, indent=2),
                          f"{company}_DCF_{datetime.now().strftime('%Y%m%d')}.json", "application/json")

if __name__ == "__main__":
    main()
st.title(f"Monte Carlo DCF – {ticker_symbol}")

st.subheader(f"Valor Esperado por Acción: **${mean_val:,.2f}**")

# Histogram bins selector
bins = 80 #st.slider("Number of histogram bins", min_value=20, max_value=200, value=80)


fig = px.histogram(
    vals,
    nbins=bins,
    title="Distribucion del Valor por Acción",
    opacity=0.75
)

# Add expected value vertical line
fig.add_vline(
    x=mean_val,
    line_width=3,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Mean: {mean_val:,.2f}",
    annotation_position="top right"
)

fig.update_layout(
    xaxis_title="Equity Value per Share",
    yaxis_title="Frequency",
    bargap=0.05,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)



