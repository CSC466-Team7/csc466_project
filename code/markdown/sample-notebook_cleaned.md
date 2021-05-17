# Running Code

First and foremost, the Jupyter Notebook is an interactive environment for writing and running code. The notebook is capable of running code in a wide range of languages. However, each notebook is associated with a single kernel.  This notebook is associated with the IPython kernel, therefore runs Python code.

## Code cells allow you to enter and run code

Run a code cell using `Shift-Enter` or pressing the <button class='btn btn-default btn-xs'><i class="icon-step-forward fa fa-play"></i></button> button in the toolbar above:


```python
a = 10
```


```python
print(a)
```

    10


There are two other keyboard shortcuts for running code:

* `Alt-Enter` runs the current cell and inserts a new one below.
* `Ctrl-Enter` run the current cell and enters command mode.

## Managing the Kernel

Code is run in a separate process called the Kernel.  The Kernel can be interrupted or restarted.  Try running the following cell and then hit the <button class='btn btn-default btn-xs'><i class='icon-stop fa fa-stop'></i></button> button in the toolbar above.

If the Kernel dies you will be prompted to restart it. Here we call the low-level system libc.time routine with the wrong argument via
ctypes to segfault the Python interpreter:

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

```
