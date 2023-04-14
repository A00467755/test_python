import streamlit as st
from joblib import load

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from dtreeviz import dtreeviz
import graphviz as graphviz


st.title("Deploying the model")
LABELS = ["setosa","versicolor","virgnica"]

clf = load("DT.joblib")

sp_l = st.slider("sepal length (cm)", min_value = 0, max_value = 10)
sp_w = st.slider("sepal width (cm)", min_value = 0, max_value = 10)

pe_l = st.slider("petal length (cm)", min_value = 0, max_value = 10)
pe_w = st.slider("petal width (cm)", min_value = 0, max_value = 10)

prediction = clf.predict([[sp_l,sp_w,pe_l,pe_w]])

st.write(LABELS[prediction[0]])







data = load_iris()

X = data["data"]
y = data["target"]

train_X, test_X, train_y, test_y = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=44)
clf = DecisionTreeClassifier()
clf = clf.fit(train_X, train_y)

# Predict test data set
pred_y = clf.predict(test_X)
accuracy = accuracy_score(test_y, pred_y)
print(accuracy)

# Predict train data set
pred_y_train = clf.predict(train_X)
accuracy_train = accuracy_score(train_y, pred_y_train)
print(accuracy_train)

print(data["target_names"][pred_y])


viz = dtreeviz(clf, 
               train_X,
               train_y,
               target_name="Classes",feature_names=["f0", "f1"], class_names=["c0", "c1"]),800)

viz.view()

