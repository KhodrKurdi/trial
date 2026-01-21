import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="My First Streamlit App", layout="wide")

st.title("🚀 My First Streamlit Trial App")

# Sidebar inputs
st.sidebar.header("Controls")
name = st.sidebar.text_input("Your name", "Khodr")
num = st.sidebar.slider("Choose a number", 0, 100, 50)

st.write(f"### Hello, **{name}** 👋")
st.write("Your selected number is:", num)

# Example table
st.write("## 📊 Random Data")
df = pd.DataFrame({
    "A": np.random.randn(10),
    "B": np.random.randn(10),
    "C": np.random.randn(10),
})
st.dataframe(df, use_container_width=True)

# Example chart
st.write("## 📈 Chart")
st.line_chart(df)

# Button action
if st.button("Click me"):
    st.success("✅ Button clicked! Streamlit works 🎉")
