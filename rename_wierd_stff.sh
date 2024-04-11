for file in "__saved_models\\"*;do
   echo $file
   echo ${file/__saved_models\\/}
   mv -i "$file" "${file/__saved_models\\/}"
done
