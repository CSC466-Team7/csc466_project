# Running Code

First and foremost, the Jupyter Notebook is an interactive environment for writing and running code. The notebook is capable of running code in a wide range of languages. However, each notebook is associated with a single kernel.  This notebook is associated with the IPython kernel, therefore runs Python code.

## Code cells allow you to enter and run code

Run a code cell using `Shift-Enter` or pressing the <button class='btn btn-default btn-xs'><i class="icon-step-forward fa fa-play"></i></button> button in the toolbar above:


```python
a = 10
```

There are two other keyboard shortcuts for running code:

* `Alt-Enter` runs the current cell and inserts a new one below.
* `Ctrl-Enter` run the current cell and enters command mode.

## Managing the Kernel

Code is run in a separate process called the Kernel.  The Kernel can be interrupted or restarted.  Try running the following cell and then hit the <button class='btn btn-default btn-xs'><i class='icon-stop fa fa-stop'></i></button> button in the toolbar above.


```python
import time
time.sleep(10)
```

If the Kernel dies you will be prompted to restart it. Here we call the low-level system libc.time routine with the wrong argument via
ctypes to segfault the Python interpreter:


```python
import sys
from ctypes import CDLL
# This will crash a Linux or Mac system
# equivalent calls can be made on Windows

# Uncomment these lines if you would like to see the segfault

# dll = 'dylib' if sys.platform == 'darwin' else 'so.6'
# libc = CDLL("libc.%s" % dll) 
# libc.time(-1)  # BOOM!!
```

## Cell menu

The "Cell" menu has a number of menu items for running code in different ways. These includes:

* Run and Select Below
* Run and Insert Below
* Run All
* Run All Above
* Run All Below

## Restarting the kernels

The kernel maintains the state of a notebook's computations. You can reset this state by restarting the kernel. This is done by clicking on the <button class='btn btn-default btn-xs'><i class='fa fa-repeat icon-repeat'></i></button> in the toolbar above.

## sys.stdout and sys.stderr

The stdout and stderr streams are displayed as text in the output area.


```python
print("hi, stdout")
```

    hi, stdout



```python
from __future__ import print_function
print('hi, stderr', file=sys.stderr)
```

    hi, stderr


## Output is asynchronous

All output is displayed asynchronously as it is generated in the Kernel. If you execute the next cell, you will see the output one piece at a time, not all at the end.


```python
import time, sys
for i in range(8):
    print(i)
    time.sleep(0.5)
```

    0
    1
    2
    3
    4
    5
    6
    7


## Large outputs

To better handle large outputs, the output area can be collapsed. Run the following cell and then single- or double- click on the active area to the left of the output:


```python
for i in range(50):
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49


Beyond a certain point, output will scroll automatically:


```python
for i in range(100):
    print(2**i - 1)
```

    0
    1
    3
    7
    15
    31
    63
    127
    255
    511
    1023
    2047
    4095
    8191
    16383
    32767
    65535
    131071
    262143
    524287
    1048575
    2097151
    4194303
    8388607
    16777215
    33554431
    67108863
    134217727
    268435455
    536870911
    1073741823
    2147483647
    4294967295
    8589934591
    17179869183
    34359738367
    68719476735
    137438953471
    274877906943
    549755813887
    1099511627775
    2199023255551
    4398046511103
    8796093022207
    17592186044415
    35184372088831
    70368744177663
    140737488355327
    281474976710655
    562949953421311
    1125899906842623
    2251799813685247
    4503599627370495
    9007199254740991
    18014398509481983
    36028797018963967
    72057594037927935
    144115188075855871
    288230376151711743
    576460752303423487
    1152921504606846975
    2305843009213693951
    4611686018427387903
    9223372036854775807
    18446744073709551615
    36893488147419103231
    73786976294838206463
    147573952589676412927
    295147905179352825855
    590295810358705651711
    1180591620717411303423
    2361183241434822606847
    4722366482869645213695
    9444732965739290427391
    18889465931478580854783
    37778931862957161709567
    75557863725914323419135
    151115727451828646838271
    302231454903657293676543
    604462909807314587353087
    1208925819614629174706175
    2417851639229258349412351
    4835703278458516698824703
    9671406556917033397649407
    19342813113834066795298815
    38685626227668133590597631
    77371252455336267181195263
    154742504910672534362390527
    309485009821345068724781055
    618970019642690137449562111
    1237940039285380274899124223
    2475880078570760549798248447
    4951760157141521099596496895
    9903520314283042199192993791
    19807040628566084398385987583
    39614081257132168796771975167
    79228162514264337593543950335
    158456325028528675187087900671
    316912650057057350374175801343
    633825300114114700748351602687



```python

```
