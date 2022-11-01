mkdir data
python download.py
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip
unzip wiki.simple.zip
mv wiki.simple.bin data/
rm wiki.simple.vec wiki.simple.zip
cd ..
cd saved_model
unzip grammar.zip
rm grammar.zip
cd ..


