# sequential-manifold-planning
Code of the 'Planning on Sequenced Manifolds' (PSM) algorithm proposed in https://arxiv.org/abs/2006.02027

![alt text](http://www.peter-englert.net/graphics/hourglass.png?raw=true)


## Usage
Install required python packages 
```
pip install -r requirements.txt
```
Run planner
```
python run_planner.py
```

Compare planner
```
python run_comparison.py
```
## Algorithms
- PSM*  ```psm/algorithms/psm.py```
- PSM* (Single Tree) ```psm/algorithms/psm_single_tree.py```
- PSM* (Greedy) ```psm/algorithms/psm_greedy.py```
- CBIRRT ```psm/algorithms/cbirrt.py```
- RRT*+IK ```psm/algorithms/ik_rrt_star.py```
- Random MMP ```psm/algorithms/random_mmp.py```

## Results
3D Point w/o obstacles
```
               Success  Path length    Comp. time
PSM            10 / 10  14.47\pm 0.04  10.64\pm 0.16
PSMSingleTree  10 / 10  14.47\pm 0.05  13.72\pm 0.18
PSMGreedy      10 / 10  16.20\pm 0.05  10.36\pm 0.10
IKRRTStar      10 / 10  17.84\pm 2.23  28.35\pm 13.50
CBIRRT         10 / 10  14.70\pm 0.71  5.04\pm 0.30
RandomMMP      10 / 10  17.33\pm 1.28  34.75\pm 17.21
-------------  -------  -------------  --------------
```
3D Point w/ obstacles
```
                      Success  Path length    Comp. time
PSM                   10 / 10  15.95\pm 0.13  13.74\pm 0.51
PSMSingleTree         10 / 10  15.87\pm 0.18  20.42\pm 0.86
PSMGreedy             10 / 10  19.69\pm 0.27  12.89\pm 0.51
IKRRTStar             10 / 10  21.56\pm 3.05  30.21\pm 7.55
CBIRRT                10 / 10  16.66\pm 1.34  3.34\pm 0.25
RandomMMP             10 / 10  22.15\pm 2.20  42.09\pm 17.22
--------------------  -------  -------------  --------------
```