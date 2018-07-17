ls *.gv | xargs -n1 -I {} dot -Tsvg {} -o {}.svg
