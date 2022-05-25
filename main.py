import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Clasificadores ML", page_icon=":bar_chart:", layout="wide")


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css(r"style/style.css")




st.title('Clasificadores de Machine Learning')

st.write("""Machine learning o aprendizaje de maquina es un componente dentro de la inteligencia artificial que permite realizar diversas
         acciones, en este caso abarcaremos modelos de clasificacion que permitira clasificar informacion
         basada en ejemplos con los que fue entrenado este modelo.""")

st.title("Conociendo los modelos")

st.write("K-Nearest Neighbors (KNN) : Es un algoritmo que agrupa la informacion y clasifica conforme a la distancia de lo centroides de cada grupo de datos. ")
st.write("Support Vector Machine (SVM) : Algoritmo que agrega una tercera dimension a nuestros datos para su clasificación. ")
st.write("Random Forest: Genera multiples arboles de decisión para la clasificación de los datos.")

st.write("""
# Explora los diferentes clasificadores y datasets
""")



st.write("Los datasets contienen distinta informacion, para comprender la informacion que se visualiza en la grafica debemos conocer al menos 3 conceptos:")
st.write("-Clasificador: Tipo de algoritmo utilizado")
st.write("-Clases: Representa el numero de posibles resultados dentro de los que se pueden agrupar los datos.")
st.write("-Precision: Representa la efectividad con la que el modelo clasifica los elementos.")


st.write("""
        A continuacion podras explorar a traves de los distintos algortimos de clasficación y datasets para analizar cual de ellos es mejor.
""")



dataset_name = st.sidebar.selectbox(
    'Selecciona Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Selecciona clasificador',
    ('KNN', 'SVM', 'Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Forma del dataset:', X.shape)
st.write('Numero de clases:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classificador = {classifier_name}')
st.write(f'Precision =', acc)



#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure(figsize=(10,6))
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='winter')

plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)



#Explanation

st.write("En la grafica podemos visualizar los distintos grupos que existen dentro del dataset y la manera en como se encuentran distribuidos, de esto dependera la eficacia de cada modelo.")
