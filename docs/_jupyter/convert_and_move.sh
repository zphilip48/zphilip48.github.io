nb=$1

echo $nb
jupyter nbconvert $nb --to markdown
base_name=$(basename ${nb})

markdown_file=${base_name%.ipynb}.md
file_folder=${base_name%.ipynb}_files

echo "Moving ${nb%.ipynb}.md to ../_posts/${markdown_file}"
mv ${nb%.ipynb}.md ../_posts/${markdown_file}
echo "Moving ${nb%.ipynb}_files to ../images/${file_folder}"
mv ${nb%.ipynb}_files ../assets/${file_folder}

# Make the file names work - replace any instance of ![png]( with ![png](../images/
sed -i .bak 's:\!\[png\](:\!\[png\](\/assets\/:' ../_posts/${markdown_file}
# Remove the backup sed made
rm ../_posts/${markdown_file}.bak
