import nbformat

MAX_LINES = 4

nb = nbformat.read("combined.ipynb", as_version=4)

for cell in nb.cells:
    if "outputs" not in cell:
        continue

    new_outputs = []
    for out in cell.outputs:
        if out.output_type == "stream":
            lines = out.text.splitlines()
            if len(lines) <= MAX_LINES:
                new_outputs.append(out)
        else:
            new_outputs.append(out)

    cell.outputs = new_outputs

nbformat.write(nb, "clean.ipynb")