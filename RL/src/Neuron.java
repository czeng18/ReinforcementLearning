public class Neuron
{
    /**
     * Weights of outgoing synapses.
     */
    double[] weights;
    /**
     * Activity of Neuron (before calling activation function, or sigmoid function).
     * Variable name: z
     */
    double activity,
    /**
     * Activation of Neuron (activation function has been called).
     * Passed to next layer of neurons.
     * Variable name: a
     */
    activation;
    int layerNumber;


    public Neuron(double[] w, int layer)
    {
        this.weights = w;
        layerNumber = layer;
    }

    public Neuron(int outputs, int layer)
    {
        if (outputs != 0)
        {
            weights = new double[outputs];
            for (int i = 0; i < outputs; i++)
            {
                weights[i] = Math.random();
            }
        }
        layerNumber = layer;
    }

    public double[] passForward(double[] inputs)
    {
        activity = Utility.sumOfAll(inputs);
        if (layerNumber == 0 || weights == null)
        {
            activation = activity;
        } else
        {
            activation = Utility.sigmoid(activity);
        }

        if (weights != null)
        {
            return Utility.multiply(activation, weights);
        }
        return new double[] {activity};
    }
}