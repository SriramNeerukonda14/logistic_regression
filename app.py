import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Logistic Regression App", layout="centered")

st.title("📚 Student Pass/Fail Prediction")
st.write("Predict whether a student will pass based on study hours.")

# Dataset
data = {
    'Study Hours': [1,2,3,4,5,6,7,8,9,10],
    'Pass/Fail': [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Split
X = df[['Study Hours']]
y = df['Pass/Fail']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

st.subheader("Dataset")
st.dataframe(df)

st.write("Model Accuracy:", round(accuracy*100,2), "%")

st.subheader("Enter Study Hours")

hours = st.slider(
    "Study Hours",
    min_value=1,
    max_value=12,
    value=5
)

# Prediction
input_data = pd.DataFrame([[hours]], columns=['Study Hours'])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

st.subheader("Prediction Result")

if prediction==1:
    st.success("✅ Student will PASS")
else:
    st.error("❌ Student will FAIL")

st.write("Pass Probability:",round(probability*100,2),"%")

# Sigmoid graph
st.subheader("Sigmoid Curve")

x=np.linspace(-10,10,100)
y=1/(1+np.exp(-x))

fig,ax=plt.subplots()
ax.plot(x,y)
ax.set_xlabel("z values")
ax.set_ylabel("Sigmoid Output")
ax.set_title("Sigmoid Curve")
ax.grid()

st.pyplot(fig)