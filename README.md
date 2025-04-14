# Netflix Capstone Project

This repository contains:
- A classification model to predict if a title is a Movie or TV Show
- A hybrid genre-aware content recommender
- A Streamlit UI for deployment

## Run the app locally
```bash
streamlit run streamlit_app_final.py
```

## Docker Deployment
```bash
docker build -t netflix-app .
docker run -p 8502:8501 netflix-app
```
