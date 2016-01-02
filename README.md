# Installation

Run the following command to pull the code from github

```bash
git clone --recursive https://github.com/hbhzwj/librl.git
cd librl/
```

Note that the recursive flag is required here because the librl repo contains pybrain as a submodule.


# Usage

You can run the example using the following command:

```base
./blaze run ./examples/maze/lstdexample.py
```

The output is printed to the stdout. You should see something like below:

```
reward:0,iteration:11683,th1:11.1373988316,th0:3.24122132022
reward:0,iteration:11684,th1:11.1374009958,th0:3.24122132022
reward:0,iteration:11685,th1:11.1374040996,th0:3.24122132022
reward:0,iteration:11686,th1:11.1374055112,th0:3.24120719029
reward:0,iteration:11687,th1:11.1374059995,th0:3.24120719029
```

In many cases, you many need to run a command multiple times to get confidence interval. The following command will run examples/maze/lstdexample.py 5 times and save the output in ./sample_results/

```bash
mkdir sample_results/
./blaze run tools/multirun.py examples/maze/lstdexample.py ./sample_results/lstd_example@5
```
If the # of runs is huge, it may take a long time. The following handy bash function can be used to check the progress of runs

```bash
function check_process() {
  	echo `ls $1 | grep ".done" | wc -l` out of \
       	`ls $1 | grep -v ".done" | wc -l` runs has finished
}
```

Add this to ~/.bashrc and run

```bash
source ~/.bashrc
check_process ./sample_results/
```
The output is

```
0 out of 4 runs has finished
```
It means that 4 runs have started and 0 of them have finished. After all job finishes, type the following command

```bash
./blaze run tools/analyzetrace.py  ./sample_results/lstd_example@5
```

to inspect the results.
