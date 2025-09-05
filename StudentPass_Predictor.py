import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
import seaborn as sns

data = pd.read_csv('Machine Learning\student_sucess_dataset.csv')

le = LabelEncoder()
data['Internet'] = le.fit_transform(data['Internet'])
data['Passed'] = le.fit_transform(data['Passed'])

features = ['StudentHours', 'Attendance', 'PastScore', 'SleepHours']
Scaler = StandardScaler()
df_scaled = data.copy()
df_scaled[features] = Scaler.fit_transform(data[features])

X = df_scaled[features]
Y = df_scaled['Passed']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2 , random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print('Classification report')
print(classification_report(Y_test,y_pred))

conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.tight_layout()
plt.show()


print('----Predict your result----')
try:
    Study_hours = float(input('Enter Study hours '))
    attendance = float(input('Enter Attendance '))
    past_score = float(input('Enter past score '))
    Sleep_hours = float(input('Enter Sleep hours '))

    user_input_df = pd.DataFrame([{
        'StudentHours': Study_hours,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': Sleep_hours
    }])

    user_input_scaled = Scaler.fit_transform(user_input_df)
    prediction = model.predict(user_input_scaled)[0]

    result = 'Pass' if prediction == 1 else 'Fail'
    print(f'Prediction based on input {result}')
except Exception as e:
    print("An Error Occour")    