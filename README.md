# Book recommendation engine

A book recommendation algorithm using Scikit learn's [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html).

We use the [Book-Crossings dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) which contains 1.1 million ratings (scale of 1-10) of 270,000 books by 90,000 users.

We use Collaborative Filtering to make our recommendations out based other user's similar preferences. We will use a machine learning algorithm : the NearestNeighbors from sklearn.neighbors to produce book recommendations that are similar (based on user's ratings) to a given book.

For more details, see the Notebook version: 
