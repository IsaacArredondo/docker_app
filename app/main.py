from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/iris")
def get_iris():
    from sklearn import datasets

    iris = datasets.load_iris()
    # unused but required import for doing 3d projections with matplotlib < 3.2
    import mpl_toolkits.mplot3d  # noqa: F401

    from sklearn.decomposition import PCA

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])

    plt.savefig("iris.png")
    file = open("iris.png", "rb")
    return StreamingResponse(file, media_type="image/png")