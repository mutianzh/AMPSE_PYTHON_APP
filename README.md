# AMPSE UI 
Analog Mixed Signal Parameter Search Engine (AMPSE) is a module-linking-graph (MLG) assisted hybrid parameter search engine with NNs to meet the demand for the wide range of AMS circuit specifications in the modern system on a chip and faster time to market requirement. It accelerates the design process and cover a wide design parameter range by performing a two-phase hybrid search. In the first phase, the hybrid search exploits the adoption of NN regression models on the MLG in the global search, where it performs a fast and parallel gradient-based optimization on the design parameters. In the second phase, to attenuate the modeling inaccuracy, it performs a local search on the MLG using SPICE simulation. This step is further accelerated with the proposed gradient-based variable reduction technique that limits the number of selected design parameters for optimization.
This is a guidence for installing and launching AMPSE on CentOS 7.
## PREREQUISITES
- A CentOS 7 Linux system
- Have XLaunch or VNC Viewer to show the graphical user interface
- Access to a command line/terminal window
- Access to the root user
### Step 1: Install Required Packages
If you don't have python3 and pip installed, install them:\
`sudo yum -y install python3 python3-pip`\
Install prerequisite packages using the following command:\
`sudo yum -y install SDL`
### Step 2: Create a Virtual Environment
Use the following command to install and upgrade Virtualenv:\
`sudo pip3 install --upgrade virtualenv`\
Create a new environment:\
`virtualenv --system-site-packages ~/venvs/ampse_gui`\
Finally, activate the environment:\
`source ~/venvs/ampse_gui/bin/activate`\
Go to directory of AMPSE_GUI and type:\
`pip install -r requirements.txt`\
Install wxPython library:\
`pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/centos-7 wxPython`

### Step 3: Launch the App
In the future, any time if you'd like to launch the app, make sure you activate the virtual environment first:\
`source ~/venvs/ampse_gui/bin/activate`\
Then go to the directory of AMPSE_GUI and type:\
`python interface_v6.py`


##
If you have any questions, please contact us at:  [uscposh@ee.usc.edu](mailto:uscposh@ee.usc.edu)
