# -Product-Defect-Classification
Machine learning project that classifies manufactured products as Defective or Non-Defective using Random Forest, Scikit-learn, and manufacturing quality parameters, with an interactive Streamlit dashboard.
# ⚙️ Product Defect Classification using Machine Learning

A machine learning project that predicts whether a manufactured product is **Defective** or **Non-Defective** based on manufacturing process and quality parameters.

The project uses a **Random Forest Classifier** built with Scikit-learn and includes an interactive **Streamlit dashboard** for real-time inspection and batch production screening.

##  Project Overview

In manufacturing industries, identifying defective products early can reduce production losses and improve quality control.

This project uses machine-learning techniques to analyze manufacturing data and classify products into:

*  **Non-Defective**
*  **Defective**

The model considers equipment conditions, production speed, raw-material quality, and operator experience to make its prediction.

##  Features

### 1. Individual Product Prediction

Enter manufacturing parameters through the Streamlit interface and receive:

* Product quality verdict
* Prediction confidence
* Defect probability
* Quality-control diagnostics
* Recommended QA action

### 2. Batch Production Screening

Upload a CSV file containing multiple production records and classify the entire production lot.

The application provides:

* Predicted defect status
* Defect probability
* Batch-level quality analysis

### 3. Model Performance

The application includes a model-performance section with a **confusion matrix** for evaluating classification results.

##  Machine Learning Model

The project uses:

**Random Forest Classifier**

Random Forest is an ensemble machine-learning algorithm that combines multiple decision trees to improve classification performance and robustness.

### Input Features

The model uses six manufacturing features:

| Feature                     | Description                          |
| --------------------------- | ------------------------------------ |
| `temperature_c`             | Manufacturing temperature in °C      |
| `pressure_bar`              | Hydraulic/clamping pressure          |
| `vibration_level`           | Equipment/spindle vibration level    |
| `production_speed`          | Production speed in units per minute |
| `material_quality_score`    | Raw-material quality score           |
| `operator_experience_years` | Operator experience in years         |

### Target Variable

`defect_status`

Possible values:

* `Defective`
* `Non-Defective`

##  Project Structure

```text
Product_Defect_Classification_Sklearn/
│
├── app.py
├── train_model.py
├── predict.py
├── product_defect_classifier.pkl
├── confusion_matrix.png
├── requirements.txt
│
├── data/
│   └── product_defects.csv
│
└── README.md
```

##  Machine Learning Workflow

```text
Manufacturing Dataset
        ↓
Data Loading
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Random Forest Classifier
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Trained Model
        ↓
Prediction
        ↓
Defective / Non-Defective
```

##  Model Training

The dataset is divided into training and testing sets using an **80:20 split**.

The Random Forest model is configured with:

* **200 decision trees**
* `random_state = 42`
* Balanced class weights

The trained model is saved using **Joblib** as:

```text
product_defect_classifier.pkl
```

##  Streamlit Application

The project includes an interactive Streamlit application with three main sections:

###  Inline QA Inspection

Allows users to enter manufacturing parameters using sliders and receive an instant quality prediction.

###  Batch Production Lot Screening

Allows users to upload a CSV file and classify multiple products at once.

###  Model Performance

Displays model evaluation information and the confusion matrix.

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Product_Defect_Classification_Sklearn.git
cd Product_Defect_Classification_Sklearn
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

##  Run the Project

### Train the Model

```bash
python train_model.py
```

### Make a Command-Line Prediction

```bash
python predict.py
```

### Launch the Streamlit Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

##  Technologies Used

* **Python**
* **Pandas** – Data processing
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning
* **Random Forest** – Classification algorithm
* **Joblib** – Model serialization
* **Matplotlib** – Confusion matrix visualization
* **Streamlit** – Interactive web application

##  Model Evaluation

The training script evaluates the model using:

* Accuracy
* Classification Report
* Confusion Matrix

The confusion matrix is generated and saved as:

```text
confusion_matrix.png
```

##  Dataset

The project uses a **synthetic manufacturing dataset** containing product and production-related measurements.

Each record contains manufacturing conditions and the corresponding defect status.

> **Note:** The dataset is synthetic and intended for educational and demonstration purposes. It should not be used as a real-world manufacturing quality-control system without validation using production data.

##  Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Cross-validation
* Feature importance visualization
* Additional machine-learning models
* Real-time sensor/IoT integration
* Model explainability using SHAP
* Cloud deployment
* Database integration
* Automated production alerts
* Real manufacturing datasets

##  Author

**Your Name**

This project demonstrates practical skills in:

`Python` • `Machine Learning` • `Scikit-learn` • `Pandas` • `Data Analysis` • `Streamlit`

##  Project Highlights

* Machine-learning based defect prediction
* Random Forest classification
* Individual product prediction
* Batch CSV screening
* Prediction probability
* Confusion matrix evaluation
* Interactive Streamlit dashboard
* End-to-end ML workflow

---

⭐ If you find this project useful, consider giving the repository a star!
