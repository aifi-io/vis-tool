# Visualization Tool for Tensorflow models

Display activation maps in real time from video input.

<p align="left">
<img src="https://github.com/ai-fi/vis-tool/blob/master/screen.png", width="720">
</p>

 ## Installation
 * Install [libavg](https://www.libavg.de/site/projects/libavg/wiki/ReleaseInstall)
 * Install [OpenCV](http://opencv.org)
 
 ## Usage
 * Convert your model to .pb 
 * run as
```{r, engine='bash'}
    $ python visualization_real_time.py --pb=<PB_FILE_PATH> --gpu=<True or False >
```
Example
```{r, engine='bash'}
    $ python visualization_real_time.py --pb=alexnet.pb --gpu=True
```

