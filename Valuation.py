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


st.set_page_config(page_title="Valoración de Acciones", layout="wide")

st.title("Valoración de Acciones")
st.subheader("Flujo de Caja Descontado")

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
        "sharesOutstanding": sharesOutstanding,
        "marketCap": equity_value,
        "total_debt": debt_value,
        "ebit": ebit, 
        "beta": beta
    }, ticker.balance_sheet, ticker.income_stmt 

# Sidebar parameters
st.sidebar.header("Parámetros")

wacc = st.sidebar.number_input("WACC (%)", value=19.48, step=0.1) / 100
fx_rate = st.sidebar.number_input("Tasa de Cambio DOP/USD", value=63.0, step=0.5)

st.sidebar.write("---")
st.sidebar.header("Escenarios de Estrés")

dict_data, b_s, i_s = fetch_data('NVDA')

st.dataframe(b_s)
st.dataframe(i_s)

