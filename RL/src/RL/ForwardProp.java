package RL;

public class ForwardProp {

    /**
     * Calculates activity of next layer given output of layer and weights of synapses
     * @param inputs    output of current layer
     * @param weights   weights of synapses in column matrix
     * @return          activity of next layer
     */
    public static float[][] passForward(float[][] inputs, float[][] weights)
    {
        // inputs:       row = set of inputs for input layer
        // weights:      row = set of weights coming from single input neuron
        // activity (z): row = activities of neurons of output layer corresponding to row of input layer
        //              # rows    = # sets of inputs
        //              # columns = # neurons in next layer
        // a = activation of next layer

        float[][] z = Utility.matrixMultiply(inputs, weights);
        float[][] a = Utility.sigmoidMat(z);
        return a;
    }
}
