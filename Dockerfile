FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model during image build
RUN python train_model.py

# Expose Streamlit's default port
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
