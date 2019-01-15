from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    '''
    Creates the Critic Model for value function.
    So this will be rewarding the action taken by the actor based on the improvement or
    deviation from the goal which should reflect in the state values.
    '''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """
        Create the critic model with mapping of S-A pairs to Q-values.
        """
        states = layers.Input(shape=(self.state_size,), name= 'states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # network for states
        net_states = layers.Dense(units=32, activation='relu', kernel_initializer='he_uniform')(states)
#         net_states = layers.Dropout(0.3)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dense(units=64, activation='relu', kernel_initializer='he_uniform')(net_states)
        # network for actions
        net_actions = layers.Dense(units=64, activation='relu', kernel_initializer='he_uniform')(actions)
#         net_actions = layers.Dropout(0.3)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
#         net_actions = layers.Dense(units=64, activation= 'relu', kernel_initializer='he_uniform')(net_actions)

        # NOTE: We now combine both actions and states
        net = layers.Add()([net_states,net_actions])
        net = layers.Activation('relu')(net)
        Q_values = layers.Dense(units=1,name= 'q_values',kernel_initializer='he_uniform')(net)
        self.model = models.Model(inputs=[states,actions], outputs = Q_values)
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer,loss= 'mse')
        
        action_gradients = K.gradients(Q_values, actions)
        self.get_action_gradients = K.function(inputs=[*self.model.input,K.learning_phase()],
        outputs=action_gradients)