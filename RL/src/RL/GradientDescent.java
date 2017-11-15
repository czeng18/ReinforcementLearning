package RL;

public class GradientDescent
{
    public static float[][] backPropagateLastLayer(NeuralNetwork network, float[][] givenOutputs, float[][] networkOutputs)
    {
        float[][] deltaThis = getDelta(network, givenOutputs, networkOutputs, network.neuronLayers.size() - 1);
        Neuron[] layer      = network.neuronLayers.get(network.neuronLayers.size() - 2);
        float[][] dZdW      = new float[layer.length][network.OUTPUT_LAYER_SIZE];

        for (int i = 0; i < dZdW.length; i++)
        {
            for (int j = 0; j < dZdW[0].length; j++)
            {
                dZdW[i][j] = layer[i].activation * layer[i].weights[j];
            }
        }

        float[][] error = Utility.matrixMultiply(Utility.transpose(dZdW), deltaThis);
        return error;
    }

    public static float[][] getDelta(NeuralNetwork network, float[][] givenOutputs, float[][] networkOutputs, int layer)
    {
        if (layer == network.neuronLayers.size() - 1)
        {
            float[][] yMinyHat = Utility.scalarSubMat(givenOutputs, networkOutputs);
            float[][] delta    = Utility.scalarMultMat(Utility.sigmoidPrimeMat(networkOutputs), yMinyHat);
            return delta;
        }
        float[][] deltaLast        = getDelta(network, givenOutputs, networkOutputs, layer + 1);
        float[][] weights          = network.weights.get(layer);
        float[][] sigmoidPrimeThis = Utility.sigmoidPrimeMat(new float[][] {network.activities.get(layer - 1)});
        float[][] deltaWeights     = Utility.matrixMultiply(deltaLast, Utility.transpose(weights));
        float[][] deltaThis        = Utility.scalarMultMat(deltaWeights, sigmoidPrimeThis);
        return deltaThis;
    }
}
