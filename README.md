# Visualization Tool for Tensorflow models

Display activation maps in real time from video input.

<p align="left">
<img src="https://github.com/ai-fi/vis-tool/blob/master/screen.png", width="720">
</p>

 ## Installation
 * Install [libavg](https://www.libavg.de/site/projects/libavg/wiki/ReleaseInstall)
 * Install [OpenCV](http://opencv.org)
 
 ## Usage
 1. Convert your model to .pb 
 1. Run as follows:
```{r, engine='bash'}
    $ python vis_tool.py --pb=<PB_FILE_PATH> --gpu=<True or False >
```
Example
```{r, engine='bash'}
    $ python vis_tool.py --pb=alexnet.pb --gpu=True
```

