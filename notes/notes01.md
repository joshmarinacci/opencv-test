For my next project I want to learn some simple machine learning in the form of
object recognition. I began with a set of tutorials 
called [Learn Code by Gaming](https://www.learncodebygaming.com/blog/tutorial/opencv-object-detection-in-games).

The code [is here](https://github.com/joshmarinacci/opencv-test).
I'm using OpenCV to recognize diamond crystals in a Roblox game. It 
technically works but is very flaky. And very slow. But it works, sorta. Check out
`test3.py` . This is using standard image rec techniques that look for parts of
an image that match a test image. There is not actual machine learning in it.

I tried to do some real ML with a cascade classifier and failed. OpenCV itself
seems pretty crufty now. Many of the command line utils required for this
tutorial are deprecated. I did the training succuessfuly, I thought, but the
classifier ran forever, never returning results but also never printing an error.
It just pegged 1 CPU forever.

Tomorrow I’ll try using a neural network with some variation of tensor flow.

```shell
opencv_createsamples -info training/pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

opencv_traincascade -data cascade/ -vec training/pos.vec -bg training/neg.txt -numPos 4 -numNeg 10 -numStages 10 -w 24 -h 24

opencv_traincascade -data cascade/ -vec pos.vec -bg training/neg.txt -numPos 5 -numNeg 10 -numStages 2 -w 24 -h 24 -featureType LBP
```
