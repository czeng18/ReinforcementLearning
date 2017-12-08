package RL;

public class GradientDescent
{
    public static double[][] gradientDescentLastLayer(NeuralNetwork network,
                                                      double[][] givenOutputs,
                                                      double[][] networkOutputs)
    {
        double[][] deltaThis = getNodeDelta(
                network,
                givenOutputs,
                networkOutputs,
                network.neuronLayers.size() - 1);

        double[][] activity = new double[][] {network.activations.get(network.activations.size() - 2)};
        double[][] error    = Utility.matrixMultiply(Utility.transpose(activity), deltaThis);
        return error;
    }

    public static double[][] getNodeDelta(NeuralNetwork network,
                                          double[][] givenOutputs,
                                          double[][] networkOutputs,
                                          int layer)
    {
        if (layer == network.neuronLayers.size() - 1)
        {
            double[][] err      = Utility.scalarSubMat(givenOutputs, networkOutputs);
            double[][] activity = new double[][] {network.activities.get(layer - 1)};
            double[][] weights  = network.weights.get(layer - 1);
            double[][] z        = Utility.matrixMultiply(activity, weights);
            double[][] sigmoid  = Utility.sigmoidPrimeMat(z);
            return Utility.scalarMultMat(err, sigmoid);
        }

        double[][] nextNodeDelta = getNodeDelta(network, givenOutputs, networkOutputs, layer + 1);
        double[][] thisWeights   = network.weights.get(layer);
        double[][] lastWeights   = network.weights.get(layer - 1);
        double[][] activity      = new double[][] {network.activities.get(layer - 1)};
        double[][] deltaWeights  = Utility.matrixMultiply(nextNodeDelta, Utility.transpose(thisWeights));
        double[][] z             = Utility.matrixMultiply(activity, lastWeights);
        double[][] sigmoid       = Utility.sigmoidPrimeMat(z);
        return Utility.scalarMultMat(deltaWeights, sigmoid);
    }

    public static double[][] gradientDescent(NeuralNetwork network,
                                             double[][] givenOutputs,
                                             double[][] networkOutputs,
                                             int layerNumber)
    {
        if (layerNumber == network.neuronLayers.size() - 1) return gradientDescentLastLayer(network, givenOutputs, networkOutputs);

        double[][] deltaThis = getNodeDelta(
                network,
                givenOutputs,
                networkOutputs,
                layerNumber);

        double[][] prevLayerActivation = new double[][] {network.activations.get(layerNumber - 1)};
        return Utility.matrixMultiply(Utility.transpose(prevLayerActivation), deltaThis);
    }
}
