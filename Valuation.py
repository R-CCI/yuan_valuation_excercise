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

st.set_page_config(page_title="Valoración de Acciones", layout="wide")

st.title("Valoración de Acciones")
st.subheader("Flujo de Caja Descontado")


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

    return out

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
tax_rate = st.sidebar.number_input('Tax Rate (%)', value=40, step=1)/100
wacc = st.sidebar.number_input("WACC (%)", value=19.48, step=0.1) / 100

st.sidebar.write("---")
st.sidebar.header("Escenarios de Estrés")

#dict_data = fetch_data('NVDA')
res = get_financials_with_annualized_ttm(ticker_symbol, statements=('income','cashflow','balance'), annualize_partial=True)
balance, income, cashflow = res['balance'].T, res['income'].T, res['cashflow'].T
debt_long = balance.loc['Long Term Debt And Capital Lease Obligation'].iloc[0]
equity_total = balance.loc['Total Equity Gross Minority Interest'].iloc[0]
sharesOutstanding = balance.loc['Share Issued'].iloc[0]
cash = balance.loc['Cash And Cash Equivalents'].iloc[0]

st.write(debt_long/equity_total)
st.dataframe(balance)
st.dataframe(income)
st.dataframe(cashflow)

rates_df = pd.read_csv('rates.csv', parse_dates=["Date"]).set_index("Date")
rfr = rates_df.iloc[-1]['10 Yr']

st.write(rates_df)


