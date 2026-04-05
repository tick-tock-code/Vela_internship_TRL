## LaTeX Build Notes

- `pdflatex.exe` install path:
  `C:\Users\joelb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`
- Template root:
  `C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\Vela__Sample_Template`
- Compile command:
  ```powershell
  & 'C:\Users\joelb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe' -interaction=nonstopmode -halt-on-error main.tex
  ```

If `pdflatex` is not on `PATH`, use the full executable path above.
