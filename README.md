# 🎓 StudentPass Predictor

A Machine Learning project that predicts whether a student will **pass or fail** based on their study habits, attendance, past performance, and sleep hours.  
This project uses **Logistic Regression** for binary classification.

---

## 📌 Project Description
The dataset includes the following columns:
- **StudentHours** – Study hours
- **Attendance** – Class attendance rate
- **PastScore** – Previous academic scores
- **SleepHours** – Average sleep duration
- **Internet** – Internet access (encoded)
- **passed** – Target column (0 = Fail, 1 = Pass)

The workflow includes:
1. Data cleaning & preprocessing (Label Encoding, Standard Scaling)  
2. Training and testing using Logistic Regression  
3. Evaluation using Classification Report & Confusion Matrix  
4. Visualization with Heatmaps  

---

## 🛠️ Technologies Used
- **Python 3**
- **Pandas** – data handling
- **Matplotlib & Seaborn** – visualization
- **Scikit-learn** – ML model & evaluation

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
