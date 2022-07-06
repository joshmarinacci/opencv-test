opencv_createsamples -info training/pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

opencv_traincascade -data cascade/ -vec training/pos.vec -bg training/neg.txt -numPos 4 -numNeg 10 -numStages 10 -w 24 -h 24

opencv_traincascade -data cascade/ -vec pos.vec -bg training/neg.txt -numPos 5 -numNeg 10 -numStages 2 -w 24 -h 24 -featureType LBP

