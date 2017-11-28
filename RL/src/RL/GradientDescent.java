package RL;

public class GradientDescent
{
    public static float[][] gradientDescentLastLayer(NeuralNetwork network,
                                                     float[][] givenOutputs,
                                                     float[][] networkOutputs)
    {
        /**
         * dJ/d(last set of weights) = -(y - y(hat)) *
         *                              f'(z(last layer of neurons) *
         *                              dz(last layer of neurons)/d(weights of last layer of neurons)
         * f(x) = sigmoid function
         * z(last layer of neurons) = activity of last layer of neurons
         * dz(last layer of neurons)/d(weights of last layer of neurons) = activity of each synapse
         *      = activation (a) of each neurons times weights
         */
        float[][] deltaThis = getNodeDelta(
                network,
                givenOutputs,
                networkOutputs,
                network.neuronLayers.size() - 1);
        float[][] dZdW      = Utility.transpose(new float[][] {network.activations.get(network.activations.size() - 2)});

        float[][] error = Utility.matrixMultiply(dZdW, Utility.transpose(deltaThis));
        return error;
    }

    public static float[][] getNodeDelta(NeuralNetwork network,
                                         float[][] givenOutputs,
                                         float[][] networkOutputs,
                                         int layer)
    {
        /**
         * delta(last layer) = -(y - yhat)(f'(z(last layer of neurons)))
         *
         * delta(other layers) = delta(preceding layer) * transpose(weights(following layer)) x f'(z(this layer))
         *
         * * = matrix multiply
         * x = scalar multiply
         */
        // delta of output layer of neurons
        if (layer == network.neuronLayers.size() - 1)
        {
            float[][] yMinyHat = Utility.scalarSubMat(networkOutputs, givenOutputs);
            float[][] sig      = Utility.sigmoidPrimeMat(networkOutputs);
            float[][] nodeDelta = Utility.scalarMultMat(yMinyHat, sig);
            return nodeDelta;
        }

        // delta of the rest of the layers
        float[][] deltaLast = getNodeDelta(
                network,
                givenOutputs,
                networkOutputs,
                layer + 1);
        float[][] weights          = network.weights.get(layer);
        System.out.println("A" + network.activities.size());
        float[][] sigmoidPrimeThis = Utility.sigmoidPrimeMat(Utility.transpose(new float[][] {network.activities.get(layer)}));
        float[][] deltaWeights     = Utility.matrixMultiply(weights, deltaLast);
        float[][] deltaThis        = Utility.scalarMultMat(deltaWeights, sigmoidPrimeThis);
        return deltaThis;
    }

    public static float[][] gradientDescent(NeuralNetwork network,
                                            float[][] givenOutputs,
                                            float[][] networkOutputs,
                                            int layerNumber)
    {
        if (layerNumber == network.NUM_HIDDEN_LAYERS + 1) return gradientDescentLastLayer(network, givenOutputs, networkOutputs);
        float[][] deltaThis = getNodeDelta(
                network,
                givenOutputs,
                networkOutputs,
                layerNumber);

        float[][] prevLayerActivation = new float[][] {network.activations.get(layerNumber - 1)};

        float[][] error = Utility.matrixMultiply(deltaThis, prevLayerActivation);
        return Utility.transpose(error);
    }
}
