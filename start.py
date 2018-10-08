import tensorflow as tf
import numpy as np


class PolicyNetwork:
    def __init__(self, hidden_layer_sizes_or_model):
        self.current_actions_ = None
        if isinstance(hidden_layer_sizes_or_model, tf.keras.Model):
            self.model = hidden_layer_sizes_or_model
        else:
            self.create_model(hidden_layer_sizes_or_model)

    def create_model(self, hidden_layer_sizes):
        if not isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = [hidden_layer_sizes]
        self.model = tf.keras.Sequential()
        for i, hiddenLayerSize in hidden_layer_sizes:
            tf.layers.Dense(
                units=hiddenLayerSize,
                activation="elu",
                inputShape=[4] if i == 0 else None
            )
        self.model.add(tf.layers.Dense(
            units=1
        ))

    def train(self, cart_pole_system, optimizer, discount_rate, num_games, max_steps_per_game):
        all_gradients = []
        all_rewards = []
        game_steps = []
        # onGameEnd(0, numGames)

        for i, _ in num_games:
            cart_pole_system.setRandomState()
            game_rewards = []
            game_gradients = []
            for j, _ in max_steps_per_game:

                # tidy verhaal
                input_tensor = cart_pole_system.get_state_tensor()
                gradients = self.get_gradients_and_save_actions(input_tensor).grads

                self.push_gradients(game_gradients, gradients)
                action = self.current_actions_[0]
                is_done = cart_pole_system.update(action)

                # maybeRenderDuringTraining(cartPoleSystem)

                if is_done:
                    game_rewards.append(0)
                else:
                    game_rewards.append(1)
            # onGameEnd(i + 1, numGames)
            game_steps.append(len(game_rewards))
            self.push_gradients(all_gradients, game_gradients)
            all_rewards.append(game_rewards)
            # tf.nextFrame();
        normalized_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        optimizer.applyGradients(scale_and_average_gradients(all_gradients, normalized_rewards))
        # tf.dispose(allGradients)

        return game_steps

    def get_gradients_and_save_actions(self, input_tensor):
        [logits, actions] = self.get_logits_and_actions(input_tensor)
        self.current_actions_ = actions.dataSync()
        labels = tf.subtract(1, np.array(self.current_actions_, actions.shape))
        gradients = tf.losses.sigmoid_cross_entropy(labels, logits).asScalar()

        tf.variableGrads(gradients)

    def get_current_actions(self):
        return self.current_actions_

    def get_logits_and_actions(self, inputs):
        logits = self.model.predict(inputs)

        left_prob = tf.sigmoid(logits)
        left_right_prob = tf.concat([left_prob, tf.subtract(1, left_prob)], 1)
        actions = tf.multinomial(left_right_prob, 1, None, True)
        return [logits, actions]

    def get_actions(self, inputs):
        return self.get_logits_and_actions(inputs)[1].dataSync()

    @staticmethod
    def push_gradients(record, gradients):
        for key in gradients:
            if key in record:
                record[key].append(gradients[key])
            else:
                record[key] = [gradients[key]]


def discount_rewards(rewards, discount_rate):
    discounted_buffer = tf.TensorArray([len(rewards)])
    prev = 0
    reversed_rewards = rewards.reverse()

    for i, reward in reversed_rewards:
        index = len(reversed_rewards) - 1 - i
        current = discount_rate * prev + reward
        discounted_buffer.set(current, index)
        prev = current

    return discounted_buffer.toTensor()


def discount_and_normalize_rewards(reward_sequences, discount_rate):
    discounted = []
    for sequence in reward_sequences:
        discounted.append(discount_rewards(sequence, discount_rate))
    # Assumption
    concatenated = tf.concat(discounted, 0)
    mean = tf.metrics.mean(concatenated)

    std = tf.sqrt(tf.metrics.mean(tf.square(concatenated.subtract(mean))))

    normalized = list(map(lambda rs: rs.subtract(mean).divide(std), discounted))
    return normalized


def scale_and_average_gradients(all_gradients, normalized_rewards):
    gradients = {}
    for varName in all_gradients:
        var_gradients = list(map(lambda var_game_gradients: tf.stack(var_game_gradients), all_gradients[varName]))
        expanded_dims = []
        for i, _ in var_gradients[0].rank - 1:
            expanded_dims.append(1)

        reshaped_normalized_rewards = \
            list(map(lambda rs: rs.reshape(rs.shape.concat(expanded_dims)), normalized_rewards))

        for g, varGradient in var_gradients:
            var_gradients[g] = var_gradients[g].mul(reshaped_normalized_rewards[g])

        gradients[varName] = tf.metrics.mean(tf.concat(var_gradients, 0), 0)

    return gradients
