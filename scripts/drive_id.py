"""Look up Drive cloud IDs by file name in the DriveFS metadata DB (read-only)."""
import glob
import os
import sqlite3
import sys

db = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Google\DriveFS\*\metadata_sqlite_db"))[0]
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cols = [r[1] for r in con.execute("PRAGMA table_info(items)")]
if "--schema" in sys.argv:
    print(cols)
    sys.exit()
for name in sys.argv[1:]:
    for row in con.execute("SELECT id, local_title, is_folder, mime_type FROM items WHERE local_title = ?", (name,)):
        print(row)
