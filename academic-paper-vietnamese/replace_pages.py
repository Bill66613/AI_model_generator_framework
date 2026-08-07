"""
Replace pages 2 and 3 in main.pdf with scanned/external PDF files.
Usage: python replace_pages.py
"""
import pypdf
import shutil

MAIN_PDF = r'd:\Workspaces\Master\ComputerScience\Thesis\AI_model_generator_framework\academic-paper-vietnamese\main.pdf'
TRANG2   = r'D:\Workspaces\Master\ComputerScience\Thesis\AI_model_generator_framework\academic-paper-vietnamese\Trang2.pdf'
TRANG3   = r'D:\Workspaces\Master\ComputerScience\Thesis\AI_model_generator_framework\academic-paper-vietnamese\Trang3.pdf'

# Backup original before overwriting
shutil.copy2(MAIN_PDF, MAIN_PDF + '.bak')
print(f'Backup saved to {MAIN_PDF}.bak')

reader   = pypdf.PdfReader(MAIN_PDF)
r_trang2 = pypdf.PdfReader(TRANG2)
r_trang3 = pypdf.PdfReader(TRANG3)
writer   = pypdf.PdfWriter()

for i, page in enumerate(reader.pages):
    if i == 1:
        writer.add_page(r_trang2.pages[0])
    elif i == 2:
        writer.add_page(r_trang3.pages[0])
    else:
        writer.add_page(page)

with open(MAIN_PDF, 'wb') as f:
    writer.write(f)

print(f'Done. Output has {len(writer.pages)} pages.')
