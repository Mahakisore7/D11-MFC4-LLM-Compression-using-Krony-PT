import os
import json
import ast

jupyter_dir = r"c:\Users\mahakisore\Academics\SEM-4\MFC-4\Project\AI\LLM-Compression-using-Krony-PT\Jupyter"

# read notebooks
for f in os.listdir(jupyter_dir):
    if f.endswith('.ipynb'):
        path = os.path.join(jupyter_dir, f)
        print(f"--- Notebook: {f} ---")
        try:
            with open(path, 'r', encoding='utf-8') as file:
                nb = json.load(file)
            
            md_cells = []
            code_length = 0
            outputs_summary = []
            for cell in nb.get('cells', []):
                if cell['cell_type'] == 'markdown':
                    md_cells.append("".join(cell.get('source', [])).strip()[:100].replace('\n', ' '))
                elif cell['cell_type'] == 'code':
                    src = "".join(cell.get('source', []))
                    code_length += len(src)
                    # Check for print outputs or error outputs
                    for output in cell.get('outputs', []):
                        if output.get('output_type') == 'error':
                            outputs_summary.append("Error: " + output.get('ename', ''))
                        elif output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                            out_text = "".join(output.get('text', []))[:50].replace('\n', ' ')
                            outputs_summary.append("Output: " + out_text)
                        elif output.get('output_type') == 'display_data':
                            outputs_summary.append("Display: Figure/Image")
            print(f"Markdown cells excerpt (first 3): {md_cells[:3]}")
            print(f"Total Code Length: {code_length} chars")
            print(f"Outputs summary (first 5 unique): {list(set(outputs_summary))[:5]}")
        except Exception as e:
            print(f"Error reading notebook: {e}")
        print("\n")
