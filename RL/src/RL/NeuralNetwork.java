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
     * Index of double[] = layer # of preceding layer of neurons.
     * double[n]    = set of weights for Neuron n in current layer.
     * double[n][m] = weight of synapse between Neuron n of current layer and Neuron m of next layer.
     */
    ArrayList<double[][]> weights = new ArrayList<>();
    /**
     * Activities of all neurons in all neuron layers.
     * Index of double[] = layer # - 1
     */
    ArrayList<double[]> activities = new ArrayList<>();
    /**
     * Activations of all neurons in all neuron layers.
     * Index of double[] = layer # - 1
     */
    ArrayList<double[]> activations = new ArrayList<>();

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

        double[][] inpWeights = new double[INPUT_LAYER_SIZE][nextLayerSize];
        for (int i = 0; i < INPUT_LAYER_SIZE; i++)
        {
            inputLayer[i] = new Neuron(nextLayerSize, 0);
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
                double[][] layerWeights = new double[currentLayerSize][nextLayerSize];

                for (int j = 0; j < layer.length; j++)
                {
                    layer[j] = new Neuron(nextLayerSize, i + 1);
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
            outputLayer[i] = new Neuron(0, NUM_HIDDEN_LAYERS + 1);
        }
        neuronLayers.add(outputLayer);
    }

    /**
     * Propagates multiple sets of inputs through the neural network
     * @param inputs    set of all inputs; inputs[n] = a set of inputs
     * @return          set of all outputs; out[n] = set of outputs corresponding to inputs[n]
     */
    public double[][] propagateForward(double[][] inputs)
    {
        /**
         * inputs: inputs[n] = a set of inputs
         */
        double[][] out = new double[inputs.length][OUTPUT_LAYER_SIZE];

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
    public double[] propagateForward(double[] inputs)
    {
        // Propagate through input layer
        Neuron[]  inputLayer = neuronLayers.get(0);
        double[][] store      = new double[inputs.length][INPUT_LAYER_SIZE];

        for (int i = 0; i < inputLayer.length; i++)
        {
            Neuron n = inputLayer[i];
            double in = inputs[i];
            store[i] = n.passThrough(new double[] {in});
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

            double[][] temp = new double[layer.length][nextLayerSize];

            for (int j = 0; j < layer.length; j++)
            {
                double[] neuronIn = Utility.getCol(store, j);
                double[] x        = layer[j].passThrough(neuronIn);
                temp[j]          = x;
            }

            store = temp;
        }

        pullActs();

        return Utility.getCol(store, 0);
    }

    public void pullActs()
    {
        for (int layerNumber = 0; layerNumber < neuronLayers.size(); layerNumber++)
        {
            Neuron[] layer = neuronLayers.get(layerNumber);
            double[] activity = new double[layer.length];
            double[] activation = new double[layer.length];
            for (int i = 0; i < layer.length; i++)
            {
                activity[i] = layer[i].activity;
                activation[i] = layer[i].activation;
            }
            activities.add(activity);
            activations.add(activation);
        }
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

    public void addToWeights(double[][] add, int layerNumber)
    {
        // Scale down change to be more accurate
        add = Utility.scalarMultiply(add, 0.001);
        double[][] newWeights = Utility.scalarAddMat(weights.get(layerNumber), add);
        setWeights(newWeights, layerNumber);
    }

    public static void main(String[] args)
    {
        NeuralNetwork n = new NeuralNetwork(2, 2, 2, new int[] {3, 4});
        double[][] gout = new double[][] {{.75, .5}};
        double[] input  = new double[] {-3, -2};
        double[][] out  = new double[][] {n.propagateForward(input)};

        System.out.println("Given Output:");
        Utility.printMat(gout);
        System.out.println("Network Output:");
        Utility.printMat(out);
        System.out.println();

        double[][] err   = GradientDescent.gradientDescent(n,
                gout,
                out,
                1);

        System.out.println("Layer 1 Weights Error:");
        Utility.printMat(err);

        double[][] err1 = GradientDescent.gradientDescent(n, gout, out, 2);

        System.out.println("Layer 2 Weights Error:");
        Utility.printMat(err1);

        double[][] err2 = GradientDescent.gradientDescent(n, gout, out, 3);

        System.out.println("Layer 3 Weights Error:");
        Utility.printMat(err2);
        System.out.println();

        for (int i = 0; i < n.weights.size(); i++)
        {
            System.out.println("Weights Layer " + i);
            Utility.printMat(n.weights.get(i));
        }

        n.addToWeights(err, 0);
        n.addToWeights(err1, 1);
        n.addToWeights(err2, 2);

        System.out.println();

        err   = GradientDescent.gradientDescent(n,
                gout,
                out,
                1);

        System.out.println("Layer 1 Weights Error:");
        Utility.printMat(err);

        err1 = GradientDescent.gradientDescent(n, gout, out, 2);

        System.out.println("Layer 2 Weights Error:");
        Utility.printMat(err1);

        err2 = GradientDescent.gradientDescent(n, gout, out, 3);

        System.out.println("Layer 3 Weights Error:");
        Utility.printMat(err2);
        System.out.println();

        for (int i = 0; i < n.weights.size(); i++)
        {
            System.out.println("Weights Layer " + i);
            Utility.printMat(n.weights.get(i));
        }

        n.addToWeights(err, 0);
        n.addToWeights(err1, 1);
        n.addToWeights(err2, 2);

        System.out.println();

        for (int i = 0; i < 10000000; i++)
        {
            out = new double[][]{n.propagateForward(input)};
            if (i % 1000000 == 0)
            {
                System.out.println("Iteration: " + i);
                System.out.println("Network Output:");
                Utility.printMat(out);

                err  = GradientDescent.gradientDescent(n, gout, out, 1);
                err1 = GradientDescent.gradientDescent(n, gout, out, 2);
                err2 = GradientDescent.gradientDescent(n, gout, out, 3);

                n.addToWeights(err, 0);
                n.addToWeights(err1, 1);
                n.addToWeights(err2, 2);
            } else
            {
                n = BackPropagation.backPropagate(n, gout, out);
            }
        }

        out = new double[][] {n.propagateForward(input)};

        System.out.println("Network Output:");
        Utility.printMat(out);

        err = GradientDescent.gradientDescent(n, gout, out, 1);

        System.out.println("Layer 1 Weights Error:");
        Utility.printMat(err);

        err1 = GradientDescent.gradientDescent(n, gout, out, 2);

        System.out.println("Layer 2 Weights Error:");
        Utility.printMat(err1);

        err2 = GradientDescent.gradientDescent(n, gout, out, 3);

        System.out.println("Layer 3 Weights Error:");
        Utility.printMat(err2);

        n.addToWeights(err, 0);
        n.addToWeights(err1, 1);
        n.addToWeights(err2, 2);

        for (int i = 0; i < n.weights.size(); i++)
        {
            System.out.println("Weights Layer " + i);
            Utility.printMat(n.weights.get(i));
        }

        out = Utility.transpose(new double[][] {n.propagateForward(input)});

        System.out.println("New Network Output:");
        Utility.printMat(out);
        System.out.println();
    }
}