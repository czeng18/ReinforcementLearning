import java.util.ArrayList;

public class Network
{
    final int NUM_LAYERS;
    /**
     * Sizes of hidden layers of network.
     * index of size = layer # - 1.
     */
    final int[] LAYER_SIZES;
    /**
     * Layers of neurons.
     * Index of Neuron[] = layer #.
     */
    ArrayList<Neuron[]> neuronLayers  = new ArrayList<>();
    /**
     * Weights on all synapses between all neurons.
     * Index of double[] = layer # of preceding layer of neurons.
     * double[n]    = set of weights for Neuron n in current layer.
     * double[n][m] = weight of synapse between Neuron n of current layer and Neuron m of next layer.
     */
    ArrayList<double[][]> weights     = new ArrayList<>();
    /**
     * Activities of all neurons in all neuron layers.
     * Index of double[] = layer # - 1
     */
    ArrayList<double[][]> activities  = new ArrayList<>();
    /**
     * Activations of all neurons in all neuron layers.
     * Index of double[] = layer # - 1
     */
    ArrayList<double[][]> activations = new ArrayList<>();

    public Network()
    {
        NUM_LAYERS = 3;
        LAYER_SIZES = new int[] {2, 3, 2};
        buildNetwork();
    }

    public Network(int numLayers, int[] layers)
    {
        NUM_LAYERS = numLayers;
        LAYER_SIZES = layers;
        buildNetwork();
    }

    public void buildNetwork()
    {
        for (int i = 0; i < LAYER_SIZES.length; i++)
        {
            int layerSize = LAYER_SIZES[i];
            Neuron[] layer = new Neuron[layerSize];
            for (int j = 0; j < layerSize; j++)
            {
                layer[j] = new Neuron(i == LAYER_SIZES.length - 1 ? 0 : LAYER_SIZES[i + 1], i);
            }
            neuronLayers.add(layer);
        }

        for (int i = 0; i < neuronLayers.size() - 1; i++)
        {
            Neuron[] layer = neuronLayers.get(i);
            double[][] weight = new double[layer.length][neuronLayers.get(i + 1).length];
            for (int j = 0; j < layer.length; j++)
            {
                weight[j] = layer[j].weights;
            }
            weights.add(weight);
        }
    }

    public double[][] passForward(double[][] inputs)
    {
        activities.clear();
        activations.clear();
        double[][] temp = inputs.clone();
        for (int i = 0; i < neuronLayers.size(); i++)
        {
            temp = passLayer(temp, i);
        }
        return temp;
    }

    public double[][] passLayer(double[][] in, int layer)
    {
        activities.add(in);
        double[][] activation;
        if (layer == 0)
        {
            activation = in.clone();
            activations.add(in);
        } else
        {
            activation = Utility.sigmoid(in);
            activations.add(activation);
        }

        double[][] out = activation.clone();
        if (layer < neuronLayers.size() - 1)
        {
            out = Utility.multiply(activation, weights.get(layer));
        }

        return out;
    }

    public void setWeights(double[][] newWeights, int layerNumber)
    {
        Neuron[] layer = neuronLayers.get(layerNumber);

        for (int i = 0; i < layer.length; i++)
        {
            layer[i].weights = newWeights[i];
        }
        neuronLayers.set(layerNumber, layer);
        weights.set(layerNumber, newWeights);
    }

    public void subtractFromWeights(double[][] sub, int layerNumber)
    {
        double[][] temp = Utility.multiply(.01, sub);
        double[][] newWeights = Utility.subtract(weights.get(layerNumber), temp);
        setWeights(newWeights, layerNumber);
    }

    public double costFunction(double[][] outputs)
    {
        double[][] diff = Utility.subtract(outputs, activations.get(NUM_LAYERS - 1));
        for (int i = 0; i < diff.length; i++)
        {
            double[] row = diff[i];
            for (int j = 0; j < row.length; j++)
            {
                diff[i][j] = Math.pow(diff[i][j], 2);
            }
        }
        return Utility.sumOfAll(diff);
    }
}
