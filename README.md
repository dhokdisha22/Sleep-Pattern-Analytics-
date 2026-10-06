# Sleep Pattern Analysis

TY (Third Year) Individual College Project — a data-driven system to analyze
sleep habits, visualize trends, classify sleep quality with a Random Forest
model, and present everything through an interactive Streamlit dashboard.

> This project is for academic analysis only. It is **not** a medical
> diagnostic tool.

## Project Structure

```
Sleep_Pattern_Analysis/
├── data/
│   ├── raw/sleep_dataset.csv           # sample dataset (synthetic)
│   └── processed/cleaned_sleep_data.csv
├── notebooks/sleep_analysis.ipynb      # exploratory analysis
├── src/
│   ├── data_cleaning.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── ml_model.py
│   └── recommendations.py
├── models/sleep_quality_model.pkl      # generated after training
├── app/app.py                          # Streamlit dashboard
├── outputs/{charts,reports}/
├── documentation/                      # PRD/FRD/TRD/Synopsis
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the Dashboard

```bash
cd app
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).
Use the sidebar to upload your own CSV or use the bundled sample dataset,
then filter by date range and age.

## Running the Notebook

```bash
jupyter notebook notebooks/sleep_analysis.ipynb
```

## Running Modules Standalone

Each module in `src/` has a `if __name__ == "__main__":` block for quick
standalone testing:

```bash
cd src
python data_cleaning.py
python analysis.py
python ml_model.py
python recommendations.py
```

## Dataset

`data/raw/sleep_dataset.csv` is a synthetic sample dataset used for academic analysis.

The dataset contains the following fields:

- Date
- Gender
- Age
- Bedtime
- Wake-up Time
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- Heart Rate
- Daily Steps

The dataset contains some missing values and duplicate records to demonstrate
data validation and cleaning.

The application normalizes and maps the dataset columns internally before
performing analysis and machine learning.

`data/raw/sleep_dataset.csv` is a synthetic sample dataset with the fields:
`date, bedtime, wake_time, sleep_duration, age, stress_level,
physical_activity_min, sleep_quality`. It intentionally includes some
missing values, duplicate rows, and an invalid `sleep_duration` value so
`data_cleaning.py` has real cases to handle. Replace it with a real dataset
(same column names) for actual analysis.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Plotly, Scikit-learn (Random Forest),
Streamlit, Jupyter Notebook, joblib. All free and open-source, per the TRD.

## Documentation

See `documentation/` for the full PRD, FRD, TRD, and Synopsis.
