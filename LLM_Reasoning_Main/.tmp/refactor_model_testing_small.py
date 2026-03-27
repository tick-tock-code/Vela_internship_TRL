from pathlib import Path

path = Path('python/pipelines/model_testing_pipeline.py')
lines = path.read_text(encoding='utf-8').splitlines()
start_def = next(i for i,l in enumerate(lines) if l.startswith('    def _run_for_transform'))
if_line = next(i for i in range(start_def, len(lines)) if lines[i].lstrip().startswith('if sweep_mode:'))
indent = len(lines[if_line]) - len(lines[if_line].lstrip())
else_line = next(i for i in range(if_line+1, len(lines)) if lines[i].startswith(' ' * indent + 'else:'))
end_func = next(i for i,l in enumerate(lines) if l.startswith('    for transform in transforms'))

new_lines = []
new_lines.extend(lines[:if_line])
for line in lines[if_line+1:else_line]:
    if line.startswith(' ' * (indent + 4)):
        new_lines.append(line[4:])
    else:
        new_lines.append(line)
new_lines.extend(lines[end_func:])

path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
print('if/else removed')
