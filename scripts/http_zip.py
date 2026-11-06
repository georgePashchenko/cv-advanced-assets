"""List or extract members of a remote zip through HTTP Range requests (no full download).

Usage: http_zip.py <url> list [substring]
       http_zip.py <url> get <member> <out_path>
"""
import io
import sys
import urllib.request
import zipfile


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


url, cmd = sys.argv[1], sys.argv[2]
zf = zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=1 << 20))
if cmd == "list":
    sub = sys.argv[3] if len(sys.argv) > 3 else ""
    for i in zf.infolist():
        if sub in i.filename:
            print(f"{i.file_size / 1e6:10.1f} MB  {i.filename}")
else:
    member, out = sys.argv[3], sys.argv[4]
    with zf.open(member) as src, open(out, "wb") as dst:
        while chunk := src.read(1 << 22):
            dst.write(chunk)
    print("saved", out)
