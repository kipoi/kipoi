.PHONY: help clean build

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build      to generate a static webpage"
	@echo "  serve      start mkdocs serve on port 8000"
	@echo "  clean      clean the generated files"
	@echo "  push       push the repository to ouga"

build:
	python autogen.py
	./render_ipynb.bash
	mkdocs build
	@echo "Build finished"

serve:
	mkdocs serve

clean:
	rm -r site sources

# push:
#	# rsync -r site/ /s/public_webshare/public/docs/concise
