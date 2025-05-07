FROM python:3.11-slim

WORKDIR /app

# Copies only the essential files
COPY app.py .
COPY threshold.json .
COPY requirements.txt .

COPY images/logo_clean.PNG images/logo_clean.PNG
COPY images/logo_name.PNG images/logo_name.PNG
COPY models/sbert_models/gte-small.zip models/sbert_models/gte-small.zip

COPY mlruns/294574624479352871/76bceb6b85914aa59da16f77472c86aa/artifacts/model/model.pkl \
     mlruns/294574624479352871/76bceb6b85914aa59da16f77472c86aa/artifacts/model/model.pkl

# Upgrades pip and installs dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposes Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]