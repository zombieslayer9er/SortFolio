# NDL Library Manager v1.3

Patch notes:
- Fixes mixed timestamp parsing for files with fractional seconds.
- Keeps v1.2 indexed duplicate detection.

Run from the outer folder:

```powershell
.\.venv\Scripts\python.exe -m ndl_library_manager scan `
  --sfw "C:\Path\To\SFW" `
  --nsfw "C:\Path\To\NSFW" `
  --out "C:\Users\zombi\Desktop\NDL_Output"
```
