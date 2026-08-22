import sys
from pypdf import PdfReader

path = r"C:\Users\Admin\Desktop\SEU_liangji\software\TM-MAN-0004-ARS_A18 Original instruction for use for the gaitway-3D instrumentation.pdf"
reader = PdfReader(path)
print("PAGES:", len(reader.pages))
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    print(f"\n===== PAGE {i+1} =====")
    print(text)
