# Data analysis libraries
import pandas as pd
from scipy.sparse import csr_matrix
# Scikit Learn Models:
from sklearn.neighbors import NearestNeighbors
# String metrics
import Levenshtein as Lev

# import books dataframe
df_books = pd.read_csv('files/BX-Books.csv',
                       encoding="ISO-8859-1",
                       sep=";",
                       header=0,
                       names=['isbn', 'title', 'author'],
                       usecols=['isbn', 'title', 'author'],
                       dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})
# import ratings dataframe
df_ratings = pd.read_csv('files/BX-Book-Ratings.csv',
                         encoding="ISO-8859-1",
                         sep=";",
                         header=0,
                         names=['user', 'isbn', 'rating'],
                         usecols=['user', 'isbn', 'rating'],
                         dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

ratings = df_ratings
# remove users that have given less than 200 ratings
users_count = ratings['user'].value_counts()
ratings = ratings[~ratings['user'].isin(users_count[users_count < 200].index)]

# remove books that have been rated less than 100 times.
books_count = ratings['rating'].value_counts()
ratings = ratings[~ratings['isbn'].isin(books_count[books_count < 100].index)]

# merge dataframe and get rid of duplicates:
df_ratings_books = pd.merge(ratings, df_books, on='isbn').dropna(axis=0, subset=['title']).drop_duplicates(
    ['user', 'title'])

# pivot ratings into matrix style
df_ratings_books_mtrx = df_ratings_books.pivot(index='title', columns='user', values='rating').fillna(0)

# convert dataframe to scipy sparse matrix for more efficient calculations
ratings_mtrx = csr_matrix(df_ratings_books_mtrx.values)

# instantiate Knn model
model = NearestNeighbors(algorithm='brute', leaf_size=30, metric='cosine',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2, ).fit(ratings_mtrx)


def get_recommends(book="", verbose=False):
    """ Take the title of a book and checks if it is in the database, then prints 5 recommendations using KNN and returns a list of
    each recommendation with its distance, if verbose is set, it also prints the distances"""
    try:
        index = df_ratings_books_mtrx.index.get_loc(book)
    except:
        print("Couldn't find any :'(")
        return [book, ["", 0.] * 5]

    knn_dist, knn_ind = model.kneighbors(df_ratings_books_mtrx.iloc[index, :].values.reshape(1, -1), n_neighbors=6)
    recommendations = [book, []]

    for i in range(0, len(knn_dist.flatten())):
        if i == 0:
            book_to_recommand = df_ratings_books_mtrx.index[index]
            print('Recommendations for {}:\n'.format(book_to_recommand))
        else:
            book_to_recommand = df_ratings_books_mtrx.index[knn_ind.flatten()[i]]
            recommendations[1].append([book_to_recommand, knn_dist.flatten()[i]])
            if verbose:
                print('{}: {}, with a distance of {:.4f}'.format(i, book_to_recommand, knn_dist.flatten()[i]))
            else:
                print('{}: {}'.format(i, book_to_recommand))
    return recommendations


def title_suggest(title, lst=list(dict.fromkeys(list(df_books['title']))), k=20):
    """Gives available suggestions of books in the database based on the Jaro distance for string matching"""

    comp = list()
    for name in lst:
        comp.append((name, Lev.jaro(title, name)))
    comp.sort(key=lambda x: x[1], reverse=True)
    print("Possible suggestions:")
    for i in range(k):
        print(comp[i][0])
    return comp[:5]
