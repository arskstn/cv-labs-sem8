input_file = "lab1.py"
output_file = "lab1_functions.py"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
inside_function = False
indent_level = None

for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("def "):
        inside_function = True
        indent_level = len(line) - len(stripped)
        new_lines.append(line)
    elif inside_function:
        # проверяем, не вышли ли из функции
        current_indent = len(line) - len(stripped)
        if stripped and current_indent <= indent_level:
            inside_function = False
            indent_level = None
        else:
            new_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Функции записаны в {output_file}")