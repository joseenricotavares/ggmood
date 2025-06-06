> **Update – 25th May 2025**  
> This project was awarded **1st place** 🥇 in the final ranking of the *Machine Learning Engineering Tutorship at Itaú Unibanco*.  
> Thank you to everyone involved in this incredible learning journey!

![GGMood Logo](https://github.com/joseenricotavares/ggmood/blob/master/images/logo_name.PNG?raw=true)

**Sentiment Analysis on Steam Game Reviews**

This project was developed as the final deliverable of the *Machine Learning Engineering tutorship at Itaú Unibanco*.

GGMood analyzes sentiment in Steam game reviews, classifying them as **positive** or **negative**. A total of **140 sentiment models** were trained, and the best-performing one is deployed on **Streamlit Cloud**:

👉 **[Launch the app](https://ggmood.streamlit.app/)**

---

## 🎥 Presentation  
Watch the full project presentation here:  
[![Watch the video](https://img.youtube.com/vi/jUDVw-lGnl4/0.jpg)](https://www.youtube.com/watch?v=jUDVw-lGnl4)

---

## 🧱 Project Main Structure

- `notebooks/` – Main model training notebooks.
- `app.py` – Streamlit application using the best sentiment model.
- `requirements.txt` – Dependencies for both Docker and Streamlit Cloud deployment.
- `project_env.yml` – Environment for full project execution (training, app, etc.).
- `Dockerfile` – Containerization focused on the application.
- `mlruns/` – MLflow artifacts from tracked experiments.
- `reports/` – `.xlsx` files with detailed model performance analysis.
