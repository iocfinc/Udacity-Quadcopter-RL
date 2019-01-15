import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
#         if np.tanh(1-abs(self.target_pos[:]-self.sim.pose[:3])).sum() >0:
#             reward = 10+ np.sqrt(((self.sim.v[:])**2).sum()) - 10*np.tanh(1-.003*abs(np.sqrt(((self.sim.pose[:3])**2).sum()) - np.sqrt(((self.target_pos[:])**2).sum())))
#             # We reward the velocity and we do not want any deviations as this would be a hover.
#         else:
#             reward = 10+ np.sqrt(((self.sim.v[:])**2).sum()) - 10*(abs(self.target_pos[:]-self.sim.pose[:3])).sum() + np.sqrt(((self.sim.linear_accel[:])**2).sum())
        
#         if np.tanh(1-abs(self.target_pos[:]-self.sim.pose[:3])).sum() >0:
#             reward = 10*np.tanh(1-.003*abs(np.sqrt(((self.sim.pose[:3])**2).sum()) - np.sqrt(((self.target_pos[:])**2).sum())))
#             # We would want to reduce the resultant deviation while keeping the velocity constant to hover. Rare
#         else:
#             reward = np.sqrt(((self.sim.v[:])**2).sum()) - np.tanh(1-0.001*abs(self.target_pos[:]-self.sim.pose[:3]).sum())
        gap_ = .01*np.sqrt((abs(self.sim.pose[:3] - self.target_pos[:])**2).sum())# Resultant gap between the target and pos.
        dif_z = abs(self.sim.pose[:3] - self.target_pos[:]).sum()
#         dif_z = abs(self.sim.pose[2]-self.target_pos[2])
        
#         reward = -10*np.tanh(.01*gap_) + 10*np.tanh(self.sim.v[2])+ 9*np.tanh(self.sim.v[1])+ 9*np.tanh(self.sim.v[0]) + 100*np.tanh(1/(0.00001+dif_z)) #First trial
        res_vel = np.sqrt((abs(self.sim.v[:])**2).sum())
#         reward = 10*np.tanh(self.sim.v[2])+ 10*np.tanh(self.sim.v[1])+ 9.99999*np.tanh(self.sim.v[0]) + 100*np.tanh(1/(0.00001+dif_z)) # Second trial
#         reward = res_vel + 100*np.tanh(1/(0.00001+dif_z))
#         reward = -10*np.tanh(.01*gap_) +  10*np.tanh(1/(0.00001+dif_z))
        reward = -10*np.tanh(.01*gap_) + 0.001*res_vel +  (1000*np.tanh(1/(0.00001+abs(self.sim.pose[:3]-self.target_pos[:])))).sum()        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state