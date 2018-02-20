#!/bin/bash
#
# Compile .md from .ipynb files for mkdocs

# Names of ipynb files to convert (from /nbs) are specified in ipynb_pages.txt

DIR=tutorials
for file in $(cat ipynb_pages.txt); do

    echo "converting: ${file}.ipynb"
    # .ipynb -> .md
    jupyter nbconvert --to markdown ../nbs/${file}.ipynb --output-dir sources/${DIR}/

    file_out=sources/${DIR}/${file}.md
    dir_out=sources/${DIR}/${file}_files

    if [ -d "${dir_out}" ]; then

	# move the additional files <file>_files to theme_dir/img/ipynb/
	rm -r theme_dir/img/ipynb/${file}_files
	mv -f ${dir_out} theme_dir/img/ipynb/

	# fix the path in the .md file
	replace '![png]('${file}'_files' '![png](/img/ipynb/'${file}'_files' -- $file_out
    fi
done    

