from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name="mlflow_lm500_2")
mlflow.autolog()

x =[] 
y =[]

for i in range(0, 1000):
    x.append(i)
    if i<500 == 0:
        y.append('less than 500')
    else:
        y.append('more than 500')
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
x_train = [[i] for i in x_train]
x_test = [[i] for i in x_test]

model = DecisionTreeClassifier(random_state=42, max_depth=1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test) *100

print(f"Model score = {score}%")
