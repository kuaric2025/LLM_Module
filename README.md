# Master-LLM Module-based Framwork (M-LLM) 
This is the repository for module-based objects pick and place task, e.g. stacking blocks task. 

M-LLM framework that leverages a pre-trained LLM (GPT-3.5-turbo) as the task planner. We evaluate the framework in tabletop block-stacking tasks in both Isaac Sim and real experiments.

# Demo 

# LLM Module

🎥 [Demo in Simulation](https://youtu.be/3k-0fHhgRWo)
🎥 [Demo in Real Experiment](https://youtu.be/-ZwCq1v3ZX8)


# Prerequisite
## Install dependencies
<pre>
git clone git@github.com:kuaric2025/LLM_Module.git
cd LLM_Module
pip install -r requirements.txt
</pre>
## Cretae conda environment
<pre>
conda env create -f environment.yml
</pre>
# Usage 
## Compile in ROS workspace
1. select gcc and g++ in bashrc
<pre>
CC=/usr/bin/gcc-11
CXX=/usr/bin/g++-11
</pre>
2. delete log, install and build in workspace 
3. comment out the service related information in ik_solver_node (let srv generate hpp)
4. compile 
<pre>
colcon build --packages-select ik_solver_pkg --cmake-args -DPYTHON_EXECUTABLE=$(which python3)
</pre>
5. recover moved code in step 4, and recompile
   


## Run M-LLM
1. ROS workspace setup under your conda environment
<pre>
source install/setup.sh 
</pre>
2. RUN ik_solver (simulation only)
<pre>
ros2 run ik_solver_pkg urdf_test_node 
</pre>
3. Run master code
<pre>
python master.py 
</pre>

## other useful commands for testing 
1. gripper control  
<pre>
ros2 topic pub /isaac_gripper_state sensor_msgs/msg/JointState "{name: ['finger_joint'], position: [0.0]}"
</pre>
open: position-0.0, close: position-0.8

2. set target pose
<pre>
ros2 service call /set_target_pose ik_solver_pkg/srv/SetTargetPose "{target_pose: {position: {x: 0.6, y: 0.4, z: 0.5}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, execute: True}"
</pre>

execute: True -- Solve IK and execute movement. 
execute: False -- Solve IK only. 

3. set joint states
<pre>
ros2 topic pub /isaac_joint_commands sensor_msgs/msg/JointState "{name: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'], position: [0.0, -1.4, 1.4, -1.6, -1.6, 0.0]}"
</pre>



