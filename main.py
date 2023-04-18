from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = FastAPI()


lista = ["kadaram kondan", "the mystic river", "the grand seduction"]


def listaPalabrasDicFrec(listaPalabras):
    frecuenciaPalab = [listaPalabras.count(p) for p in listaPalabras]
    return dict(list(zip(listaPalabras, frecuenciaPalab)))


def ordenaDicFrec(dicFrec):
    aux = [(dicFrec[key], key) for key in dicFrec]
    aux.sort()
    aux.reverse()
    return aux



@app.get('/get_max_duration/{anio}/{plataforma}/{dtype}')
def get_max_duration(anio: int, plataforma: str, dtype: str):
    data = pd.read_csv('data_final.csv')
    data.duration_int = pd.to_numeric(data.duration_int, errors='coerce')
    data.release_year = pd.to_numeric(data.release_year, errors='coerce')

    query = data[(data['release_year'] == anio)
                 & (data['plataforma'] == plataforma) &
                 (data['duration_type'] == dtype)]
    query = query.sort_values('duration_int', ascending=False)

    res = query.head(1)
    res = res.title.to_list()

    return {'El film que mas duro fue': str(''.join(res))}



@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(plataforma: str, scored: float, anio: int):
    df = pd.read_csv('data_final.csv')
    cant = df.loc[(df.score > scored) & (df.plataforma == plataforma) &
                  (df.release_year == anio)]
    conteo = cant.shape
    return {
        'plataforma': plataforma,
        'cantidad': conteo[0],
        'anio': anio,
        'score': scored
    }


@app.get('/get_count_platform/{plataforma}')
def get_count_platform(plataforma: str):
    data = pd.read_csv('data_final.csv')
    query = data['plataforma'] == plataforma
    count_query = data[query]['type'].value_counts()

    return {'plataforma': plataforma, 'peliculas': str(count_query[0])}


@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(plataforma: str, anio: int):
    data_final = pd.read_csv('data_final.csv')

    act = data_final[(data_final['plataforma'] == plataforma) &
                     (data_final['release_year'] == anio)].cast.str.split(',')
    act = act.dropna()

    actores_año = []
    for actores in act:
        for actor in actores:
            actor = actor.rstrip()
            actor = actor.lstrip()
            actores_año.append(actor)

    actor = listaPalabrasDicFrec(actores_año)
    actor = ordenaDicFrec(actor)

    return {
        'plataforma': plataforma,
        'anio': anio,
        'actor': actor[0][1],
        'apariciones': actor[0][0]
    }


@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    df = pd.read_csv('data_final.csv')
    cant3 = df.loc[(df.rating == rating)]
    conteo = cant3.shape[0]
    return {'rating': rating, 'contenido': conteo}


@app.get('/prod_per_country/{tipo}/{pais}/{anio}')
def prod_per_country(tipo: str, pais: str, anio: int):
    df = pd.read_csv('data_final.csv')
    film = df.loc[(df.type == tipo) & (df.country.str.contains(pais)) &
                  (df.release_year == anio)]
    cantidad = film.shape[0]
    return {'pais': pais, 'anio': anio, 'contenido': cantidad}



@app.get('/get_recomendation/{title}')
def get_recomendation(title,):
    df = pd.read_csv("data_final.csv").iloc[0:1000]
    tfidf = TfidfVectorizer(stop_words="english")
    df["description"] = df["description"].fillna("")

    tfidf_matriz = tfidf.fit_transform(df["description"])
    coseno_sim = linear_kernel(tfidf_matriz, tfidf_matriz)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices[title]

    # Obtenga la puntuacion de similitud de esa pelicula con todas las peliculas
    simil = list(enumerate(coseno_sim[idx]))

    # Ordenar las peliculas segun puntuacion
    simil = sorted(simil, key=lambda x: x[1], reverse=True)

    # Obtener las puntuaciones de las 10 primeras
    simil = simil[1:11]

    # Obtener los indices
    movie_index = [i[0] for i in simil]

    # Devuelve el top 10
    lista = df["title"].iloc[movie_index].to_list()[:5]
    return {'recomendacion':lista}
