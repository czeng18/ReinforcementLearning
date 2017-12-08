package RL;

import java.util.ArrayList;

public class BackPropagation
{
    public static NeuralNetwork backPropagate(NeuralNetwork n, double[][] givenOut, double[][] genOut)
    {
        ArrayList<double[][]> weights = n.weights;
        ArrayList<double[][]> errors = new ArrayList<>();

        for (int layerNumber = 1; layerNumber < weights.size(); layerNumber++)
        {
            double[][] err = GradientDescent.gradientDescent(n, givenOut, genOut, layerNumber);
            errors.add(err);
        }

        for (int i = 0; i < errors.size(); i ++)
        {
            n.addToWeights(errors.get(i), i);
        }

        return n;
    }

}
