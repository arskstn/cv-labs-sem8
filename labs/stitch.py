import nbformat

nb1 = nbformat.read("lab1.ipynb", as_version=4)
nb2 = nbformat.read("lab2.ipynb", as_version=4)

nb1.cells.extend(nb2.cells)

nbformat.write(nb1, "combined.ipynb")