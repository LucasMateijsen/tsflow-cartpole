import tensorflow as tf
import numpy as np

class PolicyNetwork:
    def __init__(self, hiddenLayerSizesOrModel):
        if isinstance(hiddenLayerSizesOrModel, tf.keras.Model):
            self.model = hiddenLayerSizesOrModel
        else:
            self.createModel(hiddenLayerSizesOrModel)


    def createModel(self, hiddenLayerSizes):
        if not isinstance(hiddenLayerSizes, list):
            hiddenLayerSizes = [hiddenLayerSizes]
        self.model = tf.keras.Sequential()
        for i, hiddenLayerSize in hiddenLayerSizes:
            tf.layers.Dense(
                units = hiddenLayerSize,
                activation = "elu",
                inputShape = [4] if i == 0 else None
            )
        self.model.add(tf.layers.Dense(
            units = 1
        ))

    def train(self, cartPoleSystem, optimizer, discountRate, numGames, maxStepsPerGame):
        allGradients = []
        allRewards = []
        gameSteps = []
        # onGameEnd(0, numGames)

        for i, numGame in numGames:
            cartPoleSystem.setRandomState()
            gameRewards = []
            gameGradients = []
            for j, maxStepPerGame in maxStepsPerGame:

                # tidy verhaal
                inputTensor = cartPoleSystem.getStateTensor()
                gradients = self.getGradientsAndSaveActions(inputTensor).grads

                self.pushGradients(gameGradients, gradients)
                action = self.currentActions_[0]
                isDone = cartPoleSystem.update(action)
                
                # maybeRenderDuringTraining(cartPoleSystem)

                if isDone:
                    gameRewards.append(0)
                else:
                    gameRewards.append(1)
            #onGameEnd(i + 1, numGames)
            gameSteps.append(gameRewards.count())
            self.pushGradients(allGradients, gameGradients)
            allRewards.append(gameRewards)
            # tf.nextFrame();
        normalizedRewards = discountAndNormalizeRewards(allRewards, discountRate)
        optimizer.applyGradients(scaleAndAverageGradients(allGradients, normalizedRewards))
        # tf.dispose(allGradients)
        
        return gameSteps

    def getGradientsAndSaveActions(self, inputTensor):
        [logits, actions] = self.getLogitsAndActions(inputTensor)
        self.currentActions_ = actions.dataSync()
        labels = tf.subtract(1, np.array(self.currentActions_, actions.shape))
        gradients = tf.losses.sigmoid_cross_entropy(labels, logits).asScalar()

        tf.variableGrads(gradients)


    def getCurrentActions(self):
        return self.currentActions_

    def getLogitsAndActions(self, inputs):
        logits = self.model.predict(inputs)

        leftProb = tf.sigmoid(logits)
        leftRightProbs = tf.concat([leftProb, tf.subtract(1, leftProb)], 1)
        actions = tf.multinomial(leftRightProbs, 1, None, True)
        return [logits, actions]

    def getActions(self, inputs):
        return self.getLogitsAndActions(inputs)[1].dataSync()

    def pushGradients(self, record, gradients):
        for key in gradients:
            if key in record:
                record[key].append(gradients[key])
            else:
                record[key] = [gradients[key]]



def discountRewards(rewards, discountRate):
    discountedBuffer = tf.buffer([rewards.count()])
    prev = 0
    reversedRewards = rewards.reverse()

    for i, reward in reversedRewards:
        index = reversedRewards.count() - 1 - i
        current = discountRate * prev + reward
        discountedBuffer.set(current, index)
        prev = current

    return discountedBuffer.toTensor()

def discountAndNormalizeRewards(rewardSequences, discountRate):
    discounted = []
    for sequence in rewardSequences:
        discounted.append(discountRewards(sequence, discountRate))
    # Assumption
    concatenated = tf.concat(discounted, 0)
    mean = tf.metrics.mean(concatenated)

    # subst = 
    cqrroot = tf.square(concatenated.subtract(mean))
    mn = tf.metrics.mean(cqrroot)
    std = tf.sqrt(mn)
    
    normalized = list(map(lambda rs: rs.subtract(mean).divide(std), discounted))
    return normalized

def scaleAndAverageGradients(allGradients, normalizedRewards):
    gradients = {}
    for varName in allGradients:
        varGradients = list(map(lambda varGameGradients: tf.stack(varGameGradients), allGradients[varName]))
        expandedDims = []
        for i, rnk in varGradients[0].rank - 1:
            expandedDims.append(1)

        reshapedNormalizedRewards = list(map(lambda rs: rs.reshape(rs.shape.concat(expandedDims))), normalizedRewards)
        for g, varGradient in varGradients:
            varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g])

        gradients[varName] = tf.metrics.mean(tf.concat(varGradients, 0), 0)
    
    return gradients