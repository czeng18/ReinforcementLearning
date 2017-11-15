package RL;

import java.util.ArrayList;

public class NeuralNetwork {
    final int INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS;
    /**
     * Sizes of hidden layers of network.
     * index of size = layer # - 1.
     */
    int[] hiddenLayerSizes;
    /**
     * Layers of neurons.
     * Index of Neuron[] = layer #.
     */
    ArrayList<Neuron[]> neuronLayers = new ArrayList<>();
    /**
     * Weights on all synapses between all neurons.
     * Index of float[] = layer # of preceding layer of neurons.
     * float[n]    = set of weights for Neuron n in current layer.
     * float[n][m] = weight of synapse between Neuron n of current layer and Neuron m of next layer.
     */
    ArrayList<float[][]> weights = new ArrayList<>();
    /**
     * Activities of all neurons in all neuron layers.
     * Index of float[] = layer # - 1
     */
    ArrayList<float[]> activities = new ArrayList<>();
    /**
     * Activations of all neurons in all neuron layers.
     * Index of float[] = layer # - 1
     */
    ArrayList<float[]> activations = new ArrayList<>();

    public NeuralNetwork(int in, int out, int numHidden, int[] hiddenSizes)
    {
        // Numbers for basic structure of network
        INPUT_LAYER_SIZE  = in;
        OUTPUT_LAYER_SIZE = out;
        NUM_HIDDEN_LAYERS = numHidden;
        hiddenLayerSizes  = hiddenSizes;


        // Create input layer of neurons
        Neuron[] inputLayer = new Neuron[INPUT_LAYER_SIZE];
        int nextLayerSize;
        if (NUM_HIDDEN_LAYERS > 0) nextLayerSize = hiddenLayerSizes[0];
        else nextLayerSize = OUTPUT_LAYER_SIZE;

        float[][] inpWeights = new float[INPUT_LAYER_SIZE][nextLayerSize];
        for (int i = 0; i < INPUT_LAYER_SIZE; i++)
        {
            inputLayer[i] = new Neuron(nextLayerSize);
            inpWeights[i] = inputLayer[i].weights;
        }
        neuronLayers.add(inputLayer);
        weights.add(inpWeights);


        // Create hidden layers of neurons
        if (NUM_HIDDEN_LAYERS != 0)
        {
            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++)
            {
                // i = layer # - 1
                int currentLayerSize = hiddenLayerSizes[i];
                if (i == NUM_HIDDEN_LAYERS - 1) nextLayerSize = OUTPUT_LAYER_SIZE;
                else nextLayerSize = hiddenLayerSizes[i + 1];

                Neuron[] layer  = new Neuron[currentLayerSize];
                float[][] layerWeights = new float[currentLayerSize][nextLayerSize];

                for (int j = 0; j < layer.length; j++)
                {
                    layer[j] = new Neuron(nextLayerSize);
                    layerWeights[j] = layer[j].weights;
                }
                neuronLayers.add(layer);
                weights.add(layerWeights);
            }
        }


        // Create output layer of neurons
        Neuron[] outputLayer = new Neuron[OUTPUT_LAYER_SIZE];
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
            outputLayer[i] = new Neuron(0);
        }
        neuronLayers.add(outputLayer);
    }

    /**
     * Propagates multiple sets of inputs through the neural network
     * @param inputs    set of all inputs; inputs[n] = a set of inputs
     * @return          set of all outputs; out[n] = set of outputs corresponding to inputs[n]
     */
    public float[][] propagateForward(float[][] inputs)
    {
        /**
         * inputs: inputs[n] = a set of inputs
         */
        float[][] out = new float[inputs.length][OUTPUT_LAYER_SIZE];

        for (int i = 0; i < inputs.length; i++)
        {
            out[i] = propagateForward(inputs[i]);
        }

        /**
         * output: out[n] = set of outputs corresponding to inputs[n]
         */
        return out;
    }

    /**
     * Propagates a set of inputs through the neural network
     * @param inputs    set of inputs
     * @return          set of outputs
     */
    public float[] propagateForward(float[] inputs)
    {
        // Propagate through input layer
        Neuron[]  inputLayer = neuronLayers.get(0);
        float[][] store      = new float[inputs.length][INPUT_LAYER_SIZE];

        for (int i = 0; i < inputLayer.length; i++)
        {
            Neuron n = inputLayer[i];
            float in = inputs[i];
            store[i] = n.passThrough(new float[] {in});
        }


        // Propagate through remaining layers
        for (int i = 1; i < neuronLayers.size(); i++)
        {
            Neuron[] layer = neuronLayers.get(i);

            int nextLayerSize;
            if (i == neuronLayers.size() - 1)
            {
                nextLayerSize = 1;
            }
            else nextLayerSize = neuronLayers.get(i + 1).length;

            float[][] temp = new float[layer.length][nextLayerSize];

            for (int j = 0; j < layer.length; j++)
            {
                float[] neuronIn = Utility.getRow(store, j);
                float[] x = layer[j].passThrough(neuronIn);
                temp[j]          = x;
                System.out.println("j" + j);
            }

            store = temp;
        }

        System.out.println(store[0][0]);
        System.out.println(store.length);
        System.out.println(store[0].length);

        return Utility.getRow(store, 0);
    }

    public static void main(String[] args)
    {
        NeuralNetwork n = new NeuralNetwork(2, 2, 1, new int[] {2});
        float[] out = n.propagateForward(new float[] {1, 1});
        float[][] err = GradientDescent.backPropagateLastLayer(n, new float[][] {{1, 1}}, new float[][] {out});
        for (float[] x : err)
        {
            for (float y : x)
            {
                System.out.print(y + " ");
            }
            System.out.println();
        }
    }
}