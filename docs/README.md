

# Documentation

The source for Kipoi documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org) and wrapped by Keras.

## Building the documentation

- Install MkDocs: `pip install mkdocs`
- If you are on OS-x, install `gnu-sed` and replace `sed` in https://github.com/kipoi/kipoi/blob/master/docs/render_ipynb.bash#L11 with the path to your `gnu-sed`
- `cd` to the `docs/` folder and run:
    - `make build`      # Builds a static site in "site" directory
	  - injects the docstrings into placeholders. See `docs/autogen.py` and `docs/templates/api/model.md`
	  - converts the ipynbs (`docs/ipynb_pages.txt`) to .md
	  - Converts the .md files to a static page. See `docs/mkdocs.yml`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](localhost:8000)
