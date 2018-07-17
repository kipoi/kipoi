#!/bin/bash
#
# Compile .md from .ipynb files for mkdocs

# Names of ipynb files to convert (from /notebooks) are specified in ipynb_pages.txt
set -e

function sed_replace {
  # https://stackoverflow.com/a/10467453/7529152
  # sed_replace <from> <to> <file>"
  sed -i "s/$(echo $1 | sed -e 's/\([[\/.*]\|\]\)/\\&/g')/$(echo $2 | sed -e 's/[\/&]/\\&/g')/g" $3
}


DIR=tutorials
for file in $(cat ipynb_pages.txt); do

    echo "converting: ${file}.ipynb"
    # .ipynb -> .md
    jupyter nbconvert --to markdown ../notebooks/${file}.ipynb --output-dir sources/${DIR}/

    file_out=sources/${DIR}/${file}.md
    dir_out=sources/${DIR}/${file}_files

    # fix the paths for the original images
    sed_replace '![img](../docs/theme_dir/img/' '![img](/img/' $file_out
    # TODO - prepend the original ipython notebook link
    echo -e "Generated from [notebooks/${file}.ipynb](https://github.com/kipoi/kipoi/blob/master/notebooks/${file}.ipynb)\n$(cat ${file_out})" > ${file_out}

    if [ -d "${dir_out}" ]; then

	# move the additional files <file>_files to theme_dir/img/ipynb/ (inline plots in ipynb)
	if [ -d "theme_dir/img/ipynb/${file}_files" ]; then
	    rm -r theme_dir/img/ipynb/${file}_files
	fi
	mv -f ${dir_out} theme_dir/img/ipynb/

	# fix the path in the .md file
	sed_replace '![png]('${file}'_files' '![png](/img/ipynb/'${file}'_files' $file_out
	sed_replace '![svg]('${file}'_files' '![svg](/img/ipynb/'${file}'_files' $file_out
    fi
done
