package RL;

import java.util.ArrayList;

public class NeuralNetwork {
    final int INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS;
    int[] hiddenLayerSizes;
    ArrayList<float[][]> weights = new ArrayList<>();

    public NeuralNetwork(int in, int out, int hidden, int[] hiddensizes)
    {
        INPUT_LAYER_SIZE  = in;
        OUTPUT_LAYER_SIZE = out;
        NUM_HIDDEN_LAYERS = hidden;
        hiddenLayerSizes  = hiddensizes;

        float[][] inpWeights = new float[hiddenLayerSizes[0]][INPUT_LAYER_SIZE];
        for (int i = 0; i < hiddenLayerSizes[0]; i++)
        {
            for (int j = 0; j < INPUT_LAYER_SIZE; j++)
            {
                inpWeights[i][j] = (float) Math.random() * 10;
            }
        }
        weights.add(inpWeights);
        System.out.println(Utility.getX(inpWeights) + " " + Utility.getY(inpWeights));

        for (int i = 0; i < NUM_HIDDEN_LAYERS; i++)
        {
            int nextLayerSize;
            if (i == NUM_HIDDEN_LAYERS - 1) nextLayerSize = OUTPUT_LAYER_SIZE;
            else nextLayerSize = hiddenLayerSizes[i + 1];

            float[][] layerWeights = new float[nextLayerSize][hiddenLayerSizes[i]];
            for (int j = 0; j < nextLayerSize; j++)
            {
                for (int k = 0; k < hiddenLayerSizes[i]; k++)
                {
                    layerWeights[j][k] = (float) Math.random() * 10;
                }
            }
            weights.add(layerWeights);
            System.out.println(Utility.getX(layerWeights) + " " + Utility.getY(layerWeights));
        }
    }

    public float[][] passForwardComplete(float[][] inputs, int layer)
    {
        float[][] a = ForwardProp.passForward(inputs, weights.get(layer));
        if (layer != weights.size() - 1)
        {
            layer++;
            return passForwardComplete(a, layer);
        }
        return a;
    }

    public static void main(String[] args)
    {
        NeuralNetwork n = new NeuralNetwork(2, 2, 2, new int[] {3, 3});
        float[][] inputs = new float[][] {{3, 5, 10}, {5, 1, 2}};
        float[][] x = n.passForwardComplete(inputs, 0);
        Utility.printMat(x);
    }
}
