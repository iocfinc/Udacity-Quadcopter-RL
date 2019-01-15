# Link the Actor and Critic modules
from agents.actor import Actor
from agents.critic import Critic
import numpy as np
import random
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=32):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size) # Initialize the mu array?
        self.theta = theta
        self.sigma = sigma
        self.reset()        
    def reset(self):
        self.state = self.mu # Change all internal states to mu.
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu-x) + self.sigma*np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class DDPG():
    def __init__(self,task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        # Create the actor instances for local and target
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        # Create the critic instances for local and target
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        # LOAD the weights
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        # Noise hyperparameters
        self.exploration_mu = 0
        self.exploration_theta = 0.35
        self.exploration_sigma = 0.1
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        # Set the replay memory
        self.buffer_size = 100000
        self.batch_size = 32
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        # Update function hyperparameters
        self.gamma = 0.99
        self.tau = 0.001
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state,action,reward,next_state,done) # Save current experience
        # When there memory is enough for a batch size then we train
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        # Move up one state
        self.last_state = next_state
    def act(self, state):
        #NOTE: This is how the agent will act, which is based on the current policy
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())
    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states,actions_next])
        # Compute Q_targets
        Q_targets = rewards + self.gamma *Q_targets_next* (1-dones)
        self.critic_local.model.train_on_batch(x=[states,actions],y=Q_targets)
        # Train the local actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])
        # Soft-update of target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

