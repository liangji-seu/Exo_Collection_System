import sys
from pypdf import PdfReader

paths = [
    r"C:\Users\Admin\Desktop\SEU_liangji\software\XING_Python_SDK-4.1.0.5645\XING_Python_SDK-4.1.0.5645\说明.pdf",
    r"C:\Users\Admin\Desktop\SEU_liangji\software\XING_Python_SDK-4.1.0.5645\XING_Python_SDK-4.1.0.5645\readme.pdf",
]
for path in paths:
    print(f"\n\n########## {path} ##########")
    try:
        reader = PdfReader(path)
        print("PAGES:", len(reader.pages))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            print(f"\n===== PAGE {i+1} =====")
            print(text)
    except Exception as exc:
        print("ERROR:", exc)
