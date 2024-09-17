# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings(action='ignore')

# Função para carregar e preparar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv('./diabetes_prediction_dataset.csv')
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    df['gender'] = df['gender'].map(gender_map)
    mode_gender = df['gender'].mode()[0]
    df['gender'] = df['gender'].replace(2, mode_gender)
    
    smoking_map = {
        'No Info': 0,
        'never': 1,
        'former': 2,
        'not current': 3,
        'ever': 4,
        'current': 5
    }
    df['smoking_history'] = df['smoking_history'].map(smoking_map)
    
    return df

# Função para balancear e dividir os dados
@st.cache_data
def processar_dados(df):
    X = df.drop(columns=['diabetes'], axis=1)
    y = df['diabetes']
    
    smote = SMOTE(random_state=25)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_resampled, y_resampled, test_size=0.1, stratify=y_resampled, random_state=25)
    
    return Xtrain, Xtest, ytrain, ytest

# Função para treinar o modelo
def treinar_modelo(Xtrain, ytrain):
    lr = LogisticRegression(random_state=1000)
    lr.fit(Xtrain, ytrain)
    pickle.dump(lr, open('modelo_treinado.pkl', 'wb'))
    return lr

# Função para avaliação do modelo
def avaliar_modelo(model, Xtest, ytest):
    ypred_proba = model.predict_proba(Xtest)[:, 1]
    fpr, tpr, thresholds = roc_curve(ytest, ypred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ypred_adjusted = (ypred_proba >= optimal_threshold).astype(int)
    
    class_report = classification_report(ytest, ypred_adjusted, target_names=["Non-Diabetic", "Diabetic"])
    auc_roc = roc_auc_score(ytest, ypred_proba)
    
    conf_matrix = confusion_matrix(ytest, ypred_adjusted)
    
    return class_report, auc_roc, conf_matrix, fpr, tpr

# Função para plotar a curva ROC
def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='best')
    st.pyplot(plt)

# Função para carregar o modelo treinado
@st.cache_resource
def carregar_modelo():
    return pickle.load(open('modelo_treinado.pkl', 'rb'))

# Função para prever novos dados
def previsao_novo_paciente(model, new_data):
    result = model.predict(new_data)[0]
    result_prob = model.predict_proba(new_data)[0, 1]
    
    if result == 1:
        return f"Diabético com {result_prob*100:.2f}% de chance."
    else:
        return f"Não diabético com {(1 - result_prob)*100:.2f}% de chance."

# Interface com Streamlit
def main():
    st.title("Previsão de Diabetes")
    st.write("Modelo preditivo para identificar a probabilidade de um paciente ser diabético.")

    # Carregar e processar os dados
    df = carregar_dados()
    Xtrain, Xtest, ytrain, ytest = processar_dados(df)

    # Treinar ou carregar o modelo
    if st.button('Treinar Modelo'):
        model = treinar_modelo(Xtrain, ytrain)
    else:
        model = carregar_modelo()

    if model:
        # Avaliar o modelo
        class_report, auc_roc, conf_matrix, fpr, tpr = avaliar_modelo(model, Xtest, ytest)

        # Exibir métricas e gráficos
        st.subheader("Relatório de Classificação")
        st.text(class_report)

        st.subheader("AUC-ROC Score")
        st.text(f"AUC-ROC Score: {auc_roc:.4f}")

        st.subheader("Matriz de Confusão")
        st.write(conf_matrix)

        st.subheader("Curva ROC")
        plot_roc_curve(fpr, tpr, auc_roc)

        # Prever novos dados
        st.subheader("Previsão para novo paciente")
        gender = st.selectbox("Gênero", ['Masculino', 'Feminino'])
        age = st.slider("Idade", 0, 100)
        hypertension = st.selectbox("Hipertensão", [0, 1])
        heart_disease = st.selectbox("Doença Cardíaca", [0, 1])
        smoking_history = st.selectbox("Histórico de Fumo", ['No Info', 'never', 'former', 'not current', 'ever', 'current'])
        bmi = st.number_input("Índice de Massa Corporal (IMC)", 10.0, 50.0)
        hba1c = st.number_input("HbA1c (%)", 3.0, 10.0)
        blood_glucose = st.number_input("Nível de Glicose no Sangue", 50, 300)

        # Mapear inputs para os dados
        smoking_map = {'No Info': 0, 'never': 1, 'former': 2, 'not current': 3, 'ever': 4, 'current': 5}
        gender_map = {'Masculino': 0, 'Feminino': 1}

        novo_paciente = np.array([gender_map[gender], age, hypertension, heart_disease, smoking_map[smoking_history], bmi, hba1c, blood_glucose]).reshape(1, -1)

        if st.button("Prever"):
            resultado = previsao_novo_paciente(model, novo_paciente)
            st.write(resultado)

if __name__ == '__main__':
    main()
    