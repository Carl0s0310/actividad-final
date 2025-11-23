# Proyect.py
# Streamlit app para clasificar especies Iris
# Integrantes: Hernando Luiz Calvo Ochoa, Carlos Antonio Ardila Ruiz
# Universidad de la Costa — Minería de Datos

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.columns = [c.replace(' (cm)','').replace(' ','_') for c in df.columns]
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

@st.cache_data
def train_model(X, y, random_state=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(Xs, y)
    return model, scaler

# --- App ---
st.set_page_config(page_title='Iris Species Classification', layout='wide')
st.title('Iris Species Classification — Minería de Datos')
st.markdown('Integrantes: **Hernando Luiz Calvo Ochoa** — **Carlos Antonio Ardila Ruiz**')

# Load
df, iris_meta = load_data()
X = df.drop(columns=['species'])
y = df['species']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train
model, scaler = train_model(X_train, y_train)

# Evaluate on test
X_test_s = scaler.transform(X_test)
y_pred = model.predict(X_test_s)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Layout: metrics and dataset/controls
col1, col2 = st.columns([1,2])
with col1:
    st.header('Métricas del modelo')
    st.metric('Accuracy', f'{accuracy:.3f}')
    st.metric('Precision (macro)', f'{precision:.3f}')
    st.metric('Recall (macro)', f'{recall:.3f}')
    st.metric('F1-score (macro)', f'{f1:.3f}')
    st.write('### Reporte de clasificación')
    st.text(classification_report(y_test, y_pred, digits=3))

with col2:
    st.header('Visualizaciones')
    st.write('Scatter 2D — Petal length vs Petal width')
    fig2 = px.scatter(df, x='petal_length', y='petal_width', color='species', symbol='species')
    st.plotly_chart(fig2, use_container_width=True)

# 3D scatter
st.write('### Scatter 3D (Sepal length, Sepal width, Petal length)')
fig3 = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length', color='species', symbol='species')
st.plotly_chart(fig3, use_container_width=True)

# User input for prediction
st.sidebar.header('Ingresar nueva muestra')
sepal_length = st.sidebar.slider('Sepal length', float(df.sepal_length.min()), float(df.sepal_length.max()), float(df.sepal_length.mean()))
sepal_width  = st.sidebar.slider('Sepal width', float(df.sepal_width.min()), float(df.sepal_width.max()), float(df.sepal_width.mean()))
petal_length = st.sidebar.slider('Petal length', float(df.petal_length.min()), float(df.petal_length.max()), float(df.petal_length.mean()))
petal_width  = st.sidebar.slider('Petal width', float(df.petal_width.min()), float(df.petal_width.max()), float(df.petal_width.mean()))

if st.sidebar.button('Predecir especie'):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_s = scaler.transform(sample)
    pred = model.predict(sample_s)[0]
    pred_proba = model.predict_proba(sample_s)[0]
    st.sidebar.success(f'Predicción: {pred}')
    probs_df = pd.DataFrame({'species': model.classes_, 'probability': pred_proba})
    st.sidebar.table(probs_df)

    # Mostrar la muestra en el scatter 3D junto al dataset
    fig_new = go.Figure()
    # puntos originales
    for s in df['species'].unique():
        dsub = df[df['species'] == s]
        fig_new.add_trace(go.Scatter3d(
            x=dsub['sepal_length'], y=dsub['sepal_width'], z=dsub['petal_length'],
            mode='markers', name=s, marker=dict(size=4)
        ))
    # punto nuevo
    fig_new.add_trace(go.Scatter3d(
        x=[sepal_length], y=[sepal_width], z=[petal_length],
        mode='markers', name='Nueva muestra', marker=dict(size=8, symbol='x')
    ))
    fig_new.update_layout(scene=dict(xaxis_title='sepal_length', yaxis_title='sepal_width', zaxis_title='petal_length'))
    st.plotly_chart(fig_new, use_container_width=True)

# Additional plots: histograms
st.write('## Distribuciones')
cols = ['sepal_length','sepal_width','petal_length','petal_width']
for c in cols:
    st.write(f'### {c}')
    st.plotly_chart(px.histogram(df, x=c, color='species', barmode='overlay'), use_container_width=True)

st.write('---')
st.write('Repositorio y video: incluir enlace en README.md')
