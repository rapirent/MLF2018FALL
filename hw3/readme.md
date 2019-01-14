# MLF2018FALL-HW2
implement logistic regression with gradient descent and stochastic gradient descent, and plot the E_in and E_out in line chart.

## How to use

### Pre-install

- In your python virtual enviroments (e.g. `pyenv`) and under this directory, run the command below:

```sh
$ pip3 install -r requirements.txt
```

### Prepare the data

download the  [training data](https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat) and [testing data](https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat
), and place it in a new directory named `data` under this directory

that is, the whole files structure (with `tree` command here) will be
```sh
$ tree
.
├── data
│   ├── hw3_test.dat
│   └── hw3_train.dat
├── hw3-q4.png
├── hw3-q5.png
├── hw3.py
└── readme.md
```

### Run the code

- and run

```sh
$ python3 hw3.py
```
## Output Structure

- output image file  for the question 4 is named `hw3-q4.png`, and `hw3-q5.png` for the question 5, you can check it out under the directory.

## About the code/code reivew

I wrote the question 4 & 5 with one python script, because I think the processes in the two question2 are same.

- I wrote the logistic regression with gradient descent in method `logistic` and logistic regression with stochastic gradient descent in method `sgd`.

- The eta of logistic is set as 0.01 and sgd is 0.001.

- I use 0/1 error in this homework.


## Author

學號: R07922009
姓名: 丁國騰
系級: 資工所碩一

## LICENSE

MIT @ Kuoteng, 2019
