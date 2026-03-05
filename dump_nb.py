import os
import json

base_dir = r'c:\Users\mahakisore\Academics\SEM-4\MFC-4\Project\AI\LLM-Compression-using-Krony-PT\Jupyter'
notebooks = [
    'Comparisons.ipynb', 
    'Pre_Training.ipynb', 
    'Adaptive Normalization.ipynb', 
    'Final-Plots.ipynb', 
    'Rank-2_Compression.ipynb',
    'Logit_Score_Calculation_Layer_by-layer.ipynb',
    'Baseline_GPT2small.ipynb'
]

out = open('nb_details.txt', 'w', encoding='utf-8')

for nb in notebooks:
    out.write(f'=== {nb} ===\n')
    try:
        with open(os.path.join(base_dir, nb), 'r', encoding='utf-8') as f:
            d = json.load(f)
            out.write(f"Total cells: {len(d['cells'])}\n")
            for i, c in enumerate(d['cells']):
                source = ''.join(c.get('source', []))
                if not source.strip(): continue
                
                if c['cell_type'] == 'markdown':
                    out.write(f"--- MD [{i}] ---\n{source[:200]}\n")
                else:
                    out.write(f"--- CODE [{i}] ---\n{source[:500]}\n")
                    
                if c.get('outputs'):
                    errs = [o for o in c['outputs'] if o.get('output_type') == 'error']
                    if errs: out.write(f"  [HAS ERRORS: {errs[0].get('ename')} - {errs[0].get('evalue')}]\n")
                    
                    streams = [o for o in c['outputs'] if o.get('output_type') == 'stream']
                    if streams: out.write(f"  [STREAM OUT: {''.join(streams[-1].get('text', []))[:100].replace(chr(10), ' ')}]\n")
                    
                    data = [o for o in c['outputs'] if o.get('output_type') == 'execute_result']
                    if data: out.write(f"  [EXEC RESULT]\n")
                    
                    display = [o for o in c['outputs'] if 'display_data' == o.get('output_type')]
                    if len(display) > 0: out.write(f"  [HAS PLOT/DISPLAY]\n")
    except Exception as e:
        out.write(f"Error reading {nb}: {e}\n")
    out.write('\n')

out.close()
