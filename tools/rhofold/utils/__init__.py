import contextlib
import tempfile
import time
from typing import Optional
import shutil

@contextlib.contextmanager
def tmpdir(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@contextlib.contextmanager
def timing(msg: str):
    print('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    print('Finished %s in %.3f seconds', msg, toc - tic)

