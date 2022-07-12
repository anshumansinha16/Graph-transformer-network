# Graph-transformer-network
An adaption of GTN's Pytorch model.

By Anshuman Sinha (anshs@gatech.edu)/ (sinha.anshuman16@gmail.com)

Re-creation of original GTN paper code in tensorflow

DBLP dataset. Download datasets (DBLP, ACM, IMDB) from this link <https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view>  and extract data.zip into data folder.

`! mkdir data`

`! cd data`

Copy all the data files in .pkl format in the folder data and cd to parent directory

Run the following code in parent directory

`! python main.py --dataset DBLP --num_layers 3`
