import nbformat

nb1 = nbformat.read("combined34.ipynb", as_version=4)
nb2 = nbformat.read("lab5.ipynb", as_version=4)

nb1.cells.extend(nb2.cells)

nbformat.write(nb1, "combined345.ipynb")