FROM python:3.10

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install --upgrade yfinance==0.2.54
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Valuation.py", "--server.port=8080", "--server.address=0.0.0.0"]
