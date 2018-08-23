.PHONY: help clean build

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build      to generate a static webpage"
	@echo "  serve      start mkdocs serve on port 8000"
	@echo "  clean      clean the generated files"
	@echo "  push       push the repository to ouga"

build:
	./render_ipynb.bash
	pydocmd build
	@echo "Build finished"

serve:
	pydocmd serve

clean:
	rm -r site sources
