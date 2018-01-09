public class GradientDescent
{
    public static double[][] getNodeDelta(Network network,
                                          double[][] givenOutputs,
                                          int layerNumber)
    {
        double[][] yhat = network.activations.get(network.NUM_LAYERS - 1);
        if (layerNumber == network.NUM_LAYERS - 1)
        {
            double[][] err          = Utility.subtract(yhat, givenOutputs);
            double[][] thisActivity = network.activities.get(layerNumber);
            double[][] sigprimez    = Utility.sigmoidPrime(thisActivity);
            return Utility.elementwiseMultiply(err, sigprimez);
        }

        double[][] nextNodeDelta = getNodeDelta(network, givenOutputs, layerNumber + 1);
        double[][] nextWeights   = network.weights.get(layerNumber);
        double[][] thisActivity  = network.activities.get(layerNumber);
        double[][] deltaWeights  = Utility.multiply(nextNodeDelta, Utility.transpose(nextWeights));
        double[][] sigmoid       = Utility.sigmoidPrime(thisActivity);
        return Utility.elementwiseMultiply(deltaWeights, sigmoid);
    }

    public static double[][] gradientDescent(Network network,
                                             double[][] givenOutputs,
                                             int layerNumber)
    {
        double[][] deltaThis = getNodeDelta(network, givenOutputs, layerNumber);

        if (layerNumber == network.neuronLayers.size() - 1)
        {
            double[][] prevLayerActivity = network.activities.get(layerNumber - 1);
            double[][] error             = Utility.multiply(Utility.transpose(prevLayerActivity), deltaThis);
            return error;
        }

        double[][] prevLayerActivation = network.activations.get(layerNumber - 1);
        return Utility.multiply(Utility.transpose(prevLayerActivation), deltaThis);
    }
}
