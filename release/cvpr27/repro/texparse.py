# Read the printed numbers of the three main tables straight from the paper's .tex (read-only).
import re
PAPER = 'D:/CVPR27_UAV'
K6 = ['AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS']


def _strip(c):
    c = c.strip(); c = re.sub(r'\\best\{([^}]*)\}', r'\1', c); c = re.sub(r'\\textbf\{([^}]*)\}', r'\1', c)
    c = re.sub(r'~?\\cite\{[^}]*\}', '', c)
    return c.replace('$-$', '-').strip()


def _rows(tex):
    out = []
    for line in open(PAPER + '/tables/' + tex, encoding='utf-8').read().split('\n'):
        if '&' not in line or line.strip().startswith('%') or 'multicolumn' in line or line.strip().startswith('Method') or line.strip().startswith('Configuration') or line.strip().startswith('Variant'):
            continue
        cells = [_strip(c) for c in line.replace('\\\\', '').split('&')]
        out.append(cells)
    return out


def main_rows():
    """{(row label, 'mav'|'anti'): {metric: printed string}}"""
    res = {}
    for c in _rows('tab_main.tex'):
        if len(c) != 13: continue
        res[(c[0], 'mav')] = dict(zip(K6, c[1:7])); res[(c[0], 'anti')] = dict(zip(K6, c[7:13]))
    return res


def ablation_rows():
    return {c[0]: dict(zip(K6, c[1:7])) for c in _rows('tab_ablation.tex') if len(c) == 7}


def sensor_rows():
    res = {}
    for c in _rows('tab_sensor.tex'):
        if len(c) != 3: continue
        for tgt, cell in zip(('mav', 'anti'), c[1:]):
            m, s = [x.strip() for x in cell.split('$\\pm$')]
            res[(c[0], tgt)] = {'mean': m, 'sd': s}
    return res


if __name__ == '__main__':
    for k, v in main_rows().items(): print(k, v)
    for k, v in ablation_rows().items(): print(k, v)
    for k, v in sensor_rows().items(): print(k, v)
