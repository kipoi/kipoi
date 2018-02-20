Adopted from Keras - [](https://github.com/fchollet/keras/tree/master/docs).

# Documentation

The source for Keras documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- install MkDocs: `pip install mkdocs`
- `cd` to the `docs/` folder and run:
    - `make build`      # Builds a static site in "site" directory
	  - injects the docstrings into placeholders. See `docs/autogen.py` and `docs/templates/api/model.md`
	  - converts the ipynbs (`docs/ipynb_pages.txt`) to .md
	  - Converts the .md files to a static page. See `docs/mkdocs.yml`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](localhost:8000)
