nb=$1
root_path="/workspace/blog-sites/my-hyde-2"
target_path="NLP"
echo $nb
jupyter nbconvert $nb --to markdown
base_name=$(basename ${nb})

markdown_file=${base_name%.ipynb}.md
file_folder=${base_name%.ipynb}_files

echo "Moving ${nb%.ipynb}.md to ${root_path}/_posts/${target_path}/${markdown_file}"
mv ${nb%.ipynb}.md ${root_path}/_posts/${target_path}/${markdown_file}
echo "Moving ${nb%.ipynb}_files to ${root_path}/images/${target_path}/${file_folder}"
mv ${nb%.ipynb}_files ${root_path}/assets/${file_folder}

# Make the file names work - replace any instance of ![png]( with ![png](../images/
sed -i .bak 's:\!\[png\](:\!\[png\](\/assets\/:' ${root_path}/_posts/${target_path}/${markdown_file}
# Remove the backup sed made
rm ${root_path}/_posts/${target_path}/${markdown_file}.bak
