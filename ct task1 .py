import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


data = {'study_hours': [12, 21, 31, 44, 15, 25, 37, 42, 27, 17, 14, 23, 33, 46, 19, 35, 39, 40, 24 ],
        'exam_scores': [50, 70, 84, 97, 52, 73, 87, 95, 75, 58, 52, 72, 87, 99, 62, 85, 90, 92, 73, 59 ]}

df = pd.DataFrame(data)


X = df[['study_hours']]
y = df['exam_scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)


mse = mean_squared_error(y_test, y_pred)


plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Fitted Line')
plt.xlabel('Study Hours')
plt.ylabel('Exams Score')
plt.title('Linear Regression')
plt.legend()
plt.show()

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared Error : {r2:.2f}")

study_hours = int(input("Enter Studied Hours"))


study_hours = np.array(study_hours).reshape(-1, 1)


predicted_marks = model.predict(study_hours)

print(f"Predicted Score for {study_hours[0][0]} hours are : {predicted_marks[0]:.2f}")