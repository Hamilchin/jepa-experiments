"""Render README.md to a PDF (single source of truth; no LaTeX needed).

Handles the Markdown subset used in README.md: #/##/### headers, **bold**, *italic*,
`code`, ``` code blocks ```, - bullets, > quotes, --- rules, ![alt](img) images, and
| pipe | tables |.  Run:  ../../.venv/bin/python build_pdf.py
"""
import os
import re
import matplotlib
from PIL import Image
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
SRC = f'{HERE}/README.md'
OUT = f'{HERE}/LeWM_OGBench_report.pdf'

pdf = FPDF(format='A4')
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_font('DV', '', f'{FONT}/DejaVuSans.ttf')
pdf.add_font('DV', 'B', f'{FONT}/DejaVuSans-Bold.ttf')
pdf.add_font('DV', 'I', f'{FONT}/DejaVuSans-Oblique.ttf')
pdf.add_font('DVm', '', f'{FONT}/DejaVuSansMono.ttf')
pdf.add_page()
W = pdf.epw

INLINE = re.compile(r'(\*\*.+?\*\*|`.+?`|\*.+?\*)')


def clean(s):
    return s.replace('&nbsp;', ' ')


def inline(line, size=9.5, base='', h=4.6):
    """Write a line with inline **bold** / *italic* / `code`, wrapping at the margin."""
    for tok in INLINE.split(clean(line)):
        if not tok:
            continue
        if tok.startswith('**') and tok.endswith('**'):
            pdf.set_font('DV', 'B', size); pdf.write(h, tok[2:-2])
        elif tok.startswith('`') and tok.endswith('`'):
            pdf.set_font('DVm', '', size - 1); pdf.write(h, tok[1:-1])
        elif tok.startswith('*') and tok.endswith('*'):
            pdf.set_font('DV', 'I', size); pdf.write(h, tok[1:-1])
        else:
            pdf.set_font('DV', base, size); pdf.write(h, tok)
    pdf.ln(h + 1.5)


def image(path):
    full = f'{HERE}/{path}'
    w_px, h_px = Image.open(full).size
    disp_w = 45 if 'fig0' in path else W           # tiny parity frames small
    disp_h = disp_w * h_px / w_px
    if pdf.get_y() + disp_h + 8 > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.image(full, w=disp_w)
    pdf.ln(1)


def table(rows):
    pdf.set_font('DV', '', 8.5)
    with pdf.table(text_align='LEFT', first_row_as_headings=True) as t:
        for r in rows:
            t.row([c.strip() for c in r])
    pdf.ln(2)


lines = open(SRC).read().split('\n')
i = 0
in_code = False
code_buf = []
while i < len(lines):
    ln = lines[i]
    if ln.strip().startswith('```'):
        if in_code:
            pdf.set_font('DVm', '', 8)
            pdf.set_fill_color(244, 244, 244)
            pdf.multi_cell(W, 4, '\n'.join(code_buf), fill=True)
            pdf.ln(2); code_buf = []
        in_code = not in_code
        i += 1; continue
    if in_code:
        code_buf.append(clean(ln)); i += 1; continue

    s = ln.strip()
    if not s:
        pdf.ln(2)
    elif s.startswith('### '):
        pdf.set_font('DV', 'B', 10.5); pdf.ln(1); pdf.multi_cell(W, 5.2, clean(s[4:]).replace('**', '')); pdf.ln(0.5)
    elif s.startswith('## '):
        pdf.set_font('DV', 'B', 13); pdf.ln(2); pdf.multi_cell(W, 6.5, clean(s[3:]).replace('**', '')); pdf.ln(1)
    elif s.startswith('# '):
        pdf.set_font('DV', 'B', 16); pdf.multi_cell(W, 8, clean(s[2:]).replace('**', '')); pdf.ln(1)
    elif s == '---':
        pdf.ln(1); pdf.set_draw_color(200, 200, 200); pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y()); pdf.ln(2)
    elif s.startswith('!['):
        m = re.search(r'\]\((.+?)\)', s)
        if m:
            image(m.group(1))
    elif s.startswith('|'):
        block = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            row = lines[i].strip().strip('|').split('|')
            if not all(set(c.strip()) <= set('-: ') for c in row):   # skip |---| separator
                block.append(row)
            i += 1
        table(block); continue
    elif s.startswith('> '):
        pdf.set_x(pdf.l_margin + 4); inline(s[2:], size=9, base='I')
    elif s.startswith('- '):
        pdf.set_font('DV', '', 9.5); pdf.set_x(pdf.l_margin + 3); pdf.write(4.6, '•  ')
        inline(s[2:], size=9.5)
    elif s.startswith('*') and s.endswith('*') and not s.startswith('**'):
        inline(s, size=8.5, base='I')      # caption lines
    else:
        inline(s, size=9.5)
    i += 1

pdf.output(OUT)
print('wrote', OUT, f'({os.path.getsize(OUT)//1024} KB)')
