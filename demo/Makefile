
flickr/data:
	echo "Downloading data" ;\
	mkdir -p flickr ; \
	cd flickr ;\
	wget -O data.tgz https://surfdrive.surf.nl/files/index.php/s/kTOqAxYrvyA1Py0/download ;\
	tar -xzvf data.tgz ;\
	rm data.tgz

flickr/models/conditional:
	echo "Downloading conditional model" ;\
	mkdir -p flickr/models ;\
	cd flickr/models ;\
	wget -O conditional.tgz https://surfdrive.surf.nl/files/index.php/s/HJT3fnHBM6oB5G6/download ;\
	tar -xzvf conditional.tgz ;\
	rm conditional.tgz ;\
	mkdir -p conditional/model/bleu/ ;\
	mv conditional/model.de-en.pt conditional/model/bleu/de-en.pt

flickr/models/aevnmt/supervised:
	echo "Downloading supervised AEVNMT model" ;\
	mkdir -p flickr/models/aevnmt ;\
	cd flickr/models/aevnmt ;\
	wget -O supervised.tgz https://surfdrive.surf.nl/files/index.php/s/5DAs3pTFJp7Lqyx/download ;\
	tar -xzvf supervised.tgz ;\
	rm supervised.tgz ;\
	mkdir -p supervised/model/bleu/ ;\
	mv supervised/model.de-en.pt supervised/model/bleu/de-en.pt ;
