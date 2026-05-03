import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("E:\~Projects\Students marks prediction\student_habits_performance.csv")

X = data[["study_hours_per_day"]]
y = data["exam_score"]

model = LinearRegression()
model.fit(X,y)
predict_score = model.predict(X)

mae = mean_absolute_error(y, predict_score)
mse = mean_squared_error(y, predict_score)
rmse = np.sqrt(mse)
r2 = r2_score(y, predict_score)
 
print("Mean Absolute error ", round(mae,2))
print("Mean square error ", round(mse))
print("Root mean squared error ", round(rmse))
print("r^2 score (Model accuracy) ", round(r2))

# Histogram 
plt.figure(figsize=(10,6))
plt.hist(data["exam_score"], bins=30, color=  'skyblue', edgecolor ="black" )
plt.title("Distribution of final exam score")
plt.xlabel("Final Exam score")
plt.ylabel("Number of students")
plt.grid(True)
plt.show()


# SCATTER Plot and new prediction

plt.scatter(X,y, color=  'blue', label = 'Actual score' )
plt.plot(X, predict_score, color = 'red', label = "predict_score(Regression line)")
plt.title("Distribution of final exam score")
plt.xlabel("Final Exam score")
plt.ylabel("Number of students")
plt.grid(True)
plt.show()


new_hours =9
predict_new_score = model.predict([[new_hours]])
print(f"Predicted final score for {new_hours} Hours is {predict_new_score} Score")
