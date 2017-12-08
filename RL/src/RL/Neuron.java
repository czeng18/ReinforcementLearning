package RL;

public class Neuron {
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

    public Neuron(double[] out, int layer)
    {
        weights     = out;
        layerNumber = layer;
    }

    public Neuron(int outputs, int layer)
    {
        if (outputs != 0)
        {
            weights = new double[outputs];
            for (int i = 0; i < outputs; i++)
            {
                weights[i] = (double) Math.random();
            }
        }
        layerNumber = layer;
    }

    public double[] passThrough(double[] inputs)
    {
        activity = Utility.sumOfAll(inputs);
        if (layerNumber == 0) activation = activity;
        else activation = Utility.sigmoidInd(activity);
        if (weights != null)
        {
            double[] out = new double[weights.length];
            for (int i = 0; i < out.length; i++)
            {
                out[i] = activation * weights[i];
            }
            return out;
        } else return new double[] {activity};
    }
}