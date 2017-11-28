package RL;

public class Neuron {
    /**
     * Weights of outgoing synapses.
     */
    float[] weights;
    /**
     * Activity of Neuron (before calling activation function, or sigmoid function).
     * Variable name: z
     */
    float activity,
    /**
     * Activation of Neuron (activation function has been called).
     * Passed to next layer of neurons.
     * Variable name: a
     */
            activation;
    int layerNumber;

    public Neuron(float[] out, int layer)
    {
        weights     = out;
        layerNumber = layer;
    }

    public Neuron(int outputs, int layer)
    {
        if (outputs != 0)
        {
            weights = new float[outputs];
            for (int i = 0; i < outputs; i++)
            {
                weights[i] = (float) Math.random();
            }
        }
        layerNumber = layer;
    }

    public float[] passThrough(float[] inputs)
    {
        activity = Utility.sumOfAll(inputs);
        if (layerNumber == 0) activation = activity;
        else activation = Utility.sigmoidInd(activity);
        if (weights != null)
        {
            float[] out = new float[weights.length];
            for (int i = 0; i < out.length; i++)
            {
                out[i] = activation * weights[i];
            }
            return out;
        } else return new float[] {activity};
    }
}