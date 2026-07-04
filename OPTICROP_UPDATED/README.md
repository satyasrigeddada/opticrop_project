# 🌾 OptiCrop: Smart Agricultural Production Optimization Engine

OptiCrop is a machine learning web application designed to help farmers and agronomists choose the right crop for their land. By analyzing soil nutrients (N, P, K) and climate conditions (temperature, humidity, pH, rainfall), the system instantly recommends the most suitable crop using a tuned Random Forest classifier.

---

## 🌟 Highlights

- **Robust ML Pipeline:** Data cleaning, duplicate removal, and Exploratory Data Analysis (EDA).
- **Model Comparison:** Benchmarked KNN, Logistic Regression, Decision Tree, and Random Forest side by side.
- **Validated Performance:** Reliability verified using 5-Fold Cross-Validation (achieved **99.5% test accuracy** with Random Forest).
- **Modular Architecture:** Clean separation of dataset, training notebook, production-ready Flask backend, and templates.
- **Seamless Deployment:** End-to-end Flask web application with a clean HTML/CSS front-end.

---

## 📝 Notes

- The model expects numeric (`float`) inputs in a fixed order: `N, P, K, temperature, humidity, ph, rainfall`. The Flask `app.py` script pulls these directly from the HTML form and converts them before prediction.
- `scaler.pkl` (a fitted `StandardScaler`) ensures the exact same feature scaling used during training is applied at inference time, preventing silent prediction drift.
- `FEATURE_ORDER` is defined identically in both `train_model.py`/`model_training.ipynb` and `app.py` to guarantee the form inputs line up with what the model was trained on.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Machine Learning:** scikit-learn
- **Data Analysis:** Pandas, NumPy, Seaborn, Matplotlib
- **Web Framework:** Flask
- **Serialization:** Pickle

---

## 📁 Project Structure

```text
OPTICROP
│
├── 📂 Dataset
│   └── Crop_recommendation.csv
│
├── 📂 Training
│   └── model_training.ipynb
│
├── 📂 Flask
│   ├── app.py
│   ├── crop_recommendation_model.pkl
│   ├── scaler.pkl
│   └── Templates/
│
├── 📂 Project_Documentation
│   ├── 1. Brainstorming_&_Ideation
│   ├── 2. Requirement_Analysis
│   ├── 3. Project_Design_Phase
│   ├── 4. Project_Planning_Phase
│   ├── 5. Project_Development_Phase
│   ├── 6. Project_Testing
│   ├── 7. Project_Documentation
│   └── 8. Project_Demonstration
│
├── README.md
└── requirements.txt
```

---

## 🚀 Setup & Installation

Follow these steps to get the web application running on your local machine.

### Prerequisites

Make sure you have [Python](https://www.python.org/) installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/opticrop.git
cd opticrop
```

### 2. Create and Activate a Virtual Environment

Set up a virtual environment to avoid dependency conflicts.

```bash
# Create venv
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

All required Python packages are listed in `requirements.txt` at the project root. Install them using pip:

```bash
pip install -r requirements.txt
```

### 4. Train the Model *(optional — `.pkl` files are already included in `Flask/`)*

```bash
cd Training
jupyter notebook model_training.ipynb
```
Run all cells — this reads the dataset from `../Dataset/` and saves `crop_recommendation_model.pkl` and `scaler.pkl` directly into `../Flask/`.

### 5. Run the Application

Navigate into the `Flask/` directory and start the development server:

```bash
cd Flask
python app.py
```

The application will be available at:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📊 Machine Learning Workflow

*(Performed inside `model_training.ipynb`)*

1. **Data Loading:** Ingested `Crop_recommendation.csv` from the `Dataset/` folder (2200+ rows, 7 features + label).
2. **Data Cleaning:** Checked for missing values and removed duplicate rows.
3. **EDA:** Used Seaborn and Matplotlib to visualize class balance, feature correlation, and feature distributions.
4. **Feature & Target Split:** Split into `N, P, K, temperature, humidity, ph, rainfall` as features and `label` (crop name) as target, with an 80/20 train-test split.
5. **Feature Scaling:** Standardized inputs with `StandardScaler` for scale-sensitive models.
6. **Model Selection:** Compared KNN, Logistic Regression, Decision Tree, and Random Forest using test accuracy and 5-fold cross-validation.
7. **Validation & Metrics:**
   - Best Model: **Random Forest**
   - Test Accuracy: **99.5%**
   - Evaluated with classification report, confusion matrix, and feature importance plot.
8. **Exploratory Clustering:** Ran K-Means as an additional unsupervised view of how crops group by soil & climate conditions.
9. **Export:** Saved the trained model and scaler using `pickle.dump()` for direct use by the Flask app.

---

## 📜 License

This project is open source and available for personal or educational use.
