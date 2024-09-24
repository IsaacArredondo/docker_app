from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt  # Falta esta importación
from sklearn import datasets
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d  # Importación necesaria para proyecciones en 3D

app = FastAPI()

@app.get("/iris")
def get_iris():
    # Carga el conjunto de datos de Iris
    iris = datasets.load_iris()

    # Configura el gráfico en 3D
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    # Realiza la reducción de dimensionalidad con PCA
    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    # Configura los títulos y etiquetas
    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])

    # Guarda la imagen en un archivo PNG
    plt.savefig("iris.png")
    
    # Abre el archivo y devuelve la respuesta
    file = open("iris.png", "rb")
    return StreamingResponse(file, media_type="image/png")
