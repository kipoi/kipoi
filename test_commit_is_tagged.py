import sys
from git import Repo

repo = Repo('.')

tags = [tag for tag in repo.tags if tag.commit == repo.commit()]

if len(tags) == 0:
    sys.exit(1)
else:
    sys.exit(0)
